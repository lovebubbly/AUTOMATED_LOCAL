"""
AI Director - Automated Image/Video Production
ComfyUI Local Integration Version

This module orchestrates AI-powered image and video generation using
local ComfyUI API instead of web-based services.
"""

import time
import os
import copy
import pandas as pd
import random
import shutil
import threading
import re
import sys
from datetime import datetime
from pathlib import Path

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional, will use defaults

# ComfyUI Integration
from comfyui_client import ComfyUIClient
from workflows import (
    load_workflow,
    set_text_prompt,
    set_image_input,
    randomize_seed,
    validate_node_ids,
    clone_workflow,
    get_node_map,
    DEFAULT_NEGATIVE_PROMPT
)

# ==========================================
# ⚙️ CONFIGURATION (Environment Variables)
# ==========================================

# 🌐 ComfyUI Connection Settings
COMFYUI_HOST = os.getenv("COMFYUI_HOST", "127.0.0.1")
COMFYUI_PORT = int(os.getenv("COMFYUI_PORT", "8188"))

# 📁 Workflow Files
WORKFLOW_IMAGE = os.getenv("WORKFLOW_IMAGE", "workflows/image_gen.json")
WORKFLOW_VIDEO = os.getenv("WORKFLOW_VIDEO", "workflows/video_gen.json")

# 📁 Path Settings
DOWNLOAD_DIR = os.getenv("OUTPUT_DIR", os.path.join(os.getcwd(), "assets", "images"))
PRODUCTION_TABLE = os.getenv("PRODUCTION_TABLE", os.path.join(os.getcwd(), "assets", "production_table.csv"))
CHAR_SHEET_PATH = os.getenv("CHAR_SHEET_PATH", os.path.join(os.getcwd(), "assets", "char_sheet.png"))

# ⏱️ Timing Settings
GENERATION_TIMEOUT = int(os.getenv("GENERATION_TIMEOUT", "600"))
WS_RECONNECT_ATTEMPTS = int(os.getenv("WS_RECONNECT_ATTEMPTS", "3"))
WS_RECONNECT_DELAY = int(os.getenv("WS_RECONNECT_DELAY", "5"))

# 🛑 Video Batch Settings
MAX_CONCURRENT_VIDEOS = int(os.getenv("MAX_CONCURRENT_VIDEOS", "4"))
VIDEO_BATCH_WAIT_TIME = int(os.getenv("VIDEO_BATCH_WAIT_TIME", "200"))

# 📝 Logging
MAX_LOG_ENTRIES = int(os.getenv("MAX_LOG_ENTRIES", "500"))


# ==========================================
# 🛠️ HELPER FUNCTIONS
# ==========================================

def ensure_dir(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)


# ==========================================
# 🎬 DIRECTOR CLASS (Stateful)
# ==========================================

class Director:
    """
    Main orchestration class for AI-powered image/video production.
    
    Uses ComfyUI local API for generation instead of web-based services.
    Maintains state for frontend dashboard integration.
    """
    
    def __init__(self):
        self.is_running = False
        self.stop_requested = False
        self.status = "IDLE"  # IDLE, IMAGING, VIDEO, COMPLETED, STOPPED, ERROR
        self.current_block = "-"
        self.logs = []
        self.generated_images = []
        self._thread = None
        
        # ComfyUI client instance
        self._client = None

    def log(self, message):
        """Adds a log message with timestamp."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        entry = f"[{timestamp}] {message}"
        print(entry)
        self.logs.append(entry)
        # Keep logs manageable
        if len(self.logs) > MAX_LOG_ENTRIES:
            self.logs.pop(0)

    def start(self, mode="full", start_block=1):
        """Starts the automation in a separate thread."""
        if self.is_running:
            self.log("⚠️ Director is already running.")
            return False
        
        self.is_running = True
        self.stop_requested = False
        self.status = "STARTING"
        self._thread = threading.Thread(target=self._run_process, args=(mode, start_block))
        self._thread.start()
        return True

    def stop(self):
        """Requests the automation to stop."""
        if not self.is_running:
            return
        self.log("🛑 Stop requested...")
        self.stop_requested = True
        self.status = "STOPPING"
        
        # Attempt to interrupt ComfyUI
        if self._client:
            try:
                self._client.interrupt_current()
            except:
                pass

    def _run_process(self, mode, start_block=1):
        """Internal runner to handle the selected mode."""
        try:
            # Initialize ComfyUI client
            self._client = ComfyUIClient(COMFYUI_HOST, COMFYUI_PORT)
            
            # Test connection
            if not self._client.test_connection():
                self.log(f"❌ Cannot connect to ComfyUI at {COMFYUI_HOST}:{COMFYUI_PORT}")
                self.log("💡 Make sure ComfyUI is running!")
                self.status = "ERROR"
                self.is_running = False
                return
            
            self.log(f"✅ Connected to ComfyUI at {COMFYUI_HOST}:{COMFYUI_PORT}")
            
            if mode in ["image", "full"]:
                self.status = "IMAGING"
                self._run_image_production(start_block)
            
            if self.stop_requested:
                self.status = "STOPPED"
                self.is_running = False
                return

            if mode in ["video", "full"] and not self.stop_requested:
                if mode == "full":
                    time.sleep(2)
                self.status = "VIDEO"
                self._run_video_production(start_block)

            if not self.stop_requested:
                self.status = "COMPLETED"
                self.log("🎉 All tasks completed successfully.")

        except Exception as e:
            self.status = "ERROR"
            self.log(f"❌ Critical Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.is_running = False
            self._client = None

    def _run_image_production(self, start_block=1):
        """
        Image production using ComfyUI API.
        
        Replaces the Playwright-based Nano Banana automation with
        direct ComfyUI HTTP/WebSocket API calls.
        """
        ensure_dir(DOWNLOAD_DIR)
        df = pd.read_csv(PRODUCTION_TABLE)
        self.log(f"🚀 Starting Image Production ({len(df)} blocks, Starting from {start_block})...")
        
        # Load workflow template
        try:
            workflow_path = os.path.join(os.getcwd(), WORKFLOW_IMAGE)
            workflow_template = load_workflow(workflow_path)
            self.log(f"   📋 Loaded workflow: {WORKFLOW_IMAGE}")
        except FileNotFoundError as e:
            self.log(f"❌ Workflow file not found: {e}")
            self.log("💡 Please export your workflow from ComfyUI and save it to workflows/image_gen.json")
            return
        except Exception as e:
            self.log(f"❌ Failed to load workflow: {e}")
            return
        
        # Get node mappings
        try:
            node_map = get_node_map("image_gen")
        except KeyError as e:
            self.log(f"❌ Node mapping error: {e}")
            return
        
        # Validate node IDs against workflow
        missing_nodes = validate_node_ids(workflow_template, node_map)
        if missing_nodes:
            self.log(f"⚠️ Warning: Some node IDs not found in workflow: {missing_nodes}")
            self.log("💡 Update workflows/config.py with correct Node IDs from your workflow")
        
        # Upload character sheet (fixed reference)
        char_sheet_uploaded = None
        if os.path.exists(CHAR_SHEET_PATH):
            try:
                char_sheet_uploaded = self._client.upload_image(CHAR_SHEET_PATH)
                self.log(f"   📎 Char Sheet uploaded: {char_sheet_uploaded}")
            except Exception as e:
                self.log(f"   ⚠️ Char Sheet upload failed: {e}")
        
        # Rolling reference tracking
        current_reference_path = CHAR_SHEET_PATH
        
        for index, row in df.iterrows():
            if self.stop_requested:
                break

            block_id = str(row['Block']).zfill(2)
            
            # SKIP Logic
            if int(block_id) < start_block:
                continue

            self.current_block = f"Block {block_id} (Image)"
            self.log(f"🎬 Processing Image Block {block_id}...")

            start_prompt = str(row['Nano Banana (Start Frame)'])
            
            # ============================================================
            # LOOP BANK: Handle '[Loop Bank A/B/C]' placeholders
            # ============================================================
            loop_match = re.search(r'\[Loop Bank ([A-C])\]', start_prompt)
            if loop_match:
                loop_key = loop_match.group(1)
                loop_file = os.path.join(DOWNLOAD_DIR, f"loop_bank_{loop_key}.png")
                
                if os.path.exists(loop_file):
                    # Reuse existing Loop Bank image
                    self.log(f"   ♻️ Reusing Loop Bank {loop_key}...")
                    filename = f"block_{block_id}_start.png"
                    target_path = os.path.join(DOWNLOAD_DIR, filename)
                    shutil.copy(loop_file, target_path)
                    self.log(f"   ✅ Copied Loop Bank {loop_key} to: {filename}")
                    current_reference_path = target_path
                    self.generated_images.append(filename)
                    continue  # Skip to next block
                else:
                    # First time: Generate and save as Loop Bank
                    self.log(f"   🎬 First Loop Bank {loop_key} - will generate and cache...")
                    # Extract the actual prompt after the placeholder
                    actual_prompt = re.sub(r'\[Loop Bank [A-C]\]\s*', '', start_prompt).strip()
                    if actual_prompt:
                        start_prompt = actual_prompt
            
            # ============================================================
            # PROTOCOL: Handle '[Input: Last Frame]' / '[Input: Last Frame of XX]'
            # ============================================================
            skip_start_generation = False
            if "[Input: Last Frame" in start_prompt:
                self.log(f"   ⏭️ Detected Last Frame Placeholder...")
                
                # Parse block reference: [Input: Last Frame of 02] -> block 02
                ref_match = re.search(r'\[Input: Last Frame of (\d+)\]', start_prompt)
                
                if ref_match:
                    ref_block_id = ref_match.group(1).zfill(2)
                    # Try End Frame first, then Start Frame
                    ref_end_path = os.path.join(DOWNLOAD_DIR, f"block_{ref_block_id}_end.png")
                    ref_start_path = os.path.join(DOWNLOAD_DIR, f"block_{ref_block_id}_start.png")
                    
                    if os.path.exists(ref_end_path):
                        source_path = ref_end_path
                        self.log(f"   📎 Using End Frame of Block {ref_block_id}")
                    elif os.path.exists(ref_start_path):
                        source_path = ref_start_path
                        self.log(f"   📎 Using Start Frame of Block {ref_block_id} (no End Frame)")
                    else:
                        self.log(f"   ⚠️ Block {ref_block_id} images not found! Using current reference...")
                        source_path = current_reference_path
                else:
                    # Fallback: [Input: Last Frame] without block number -> use current reference
                    source_path = current_reference_path
                    self.log(f"   📎 Using current rolling reference")
                
                # Copy the source to this block's output
                filename = f"block_{block_id}_start.png"
                target_path = os.path.join(DOWNLOAD_DIR, filename)
                
                if source_path and os.path.exists(source_path):
                    shutil.copy(source_path, target_path)
                    self.log(f"   ✅ Copied to: {filename}")
                    current_reference_path = target_path
                    self.generated_images.append(filename)
                    skip_start_generation = True
                else:
                    self.log("   ⚠️ No source image found for Last Frame placeholder!")
                    skip_start_generation = True

            # ============================================================
            # GENERATION: Create image using ComfyUI
            # ============================================================
            if not skip_start_generation:
                saved_path = self._generate_image(
                    prompt=start_prompt,
                    reference_path=current_reference_path,
                    char_sheet_name=char_sheet_uploaded,
                    workflow_template=workflow_template,
                    node_map=node_map,
                    output_filename=f"block_{block_id}_start.png"
                )
                
                if saved_path:
                    current_reference_path = saved_path
                    self.generated_images.append(f"block_{block_id}_start.png")
                    
                    # LOOP BANK: Cache if this was a Loop Bank block
                    if loop_match:
                        loop_cache_file = os.path.join(DOWNLOAD_DIR, f"loop_bank_{loop_match.group(1)}.png")
                        shutil.copy(saved_path, loop_cache_file)
                        self.log(f"   💾 Cached as Loop Bank {loop_match.group(1)}")
                else:
                    self.log(f"   ❌ Failed to generate Start Frame for Block {block_id}")
                    continue  # Skip End Frame if Start Frame failed

            # ============================================================
            # PROTOCOL A: End Frame Generation
            # ============================================================
            protocol = str(row['Protocol']).strip().upper()
            is_protocol_a = protocol.startswith("A") or "KEYFRAME" in protocol
            
            if is_protocol_a and not self.stop_requested:
                end_prompt = row['Nano Banana (End Frame)']
                
                # Skip if End Frame is empty, NaN, or a placeholder
                skip_end_frame = (
                    pd.isna(end_prompt) or 
                    str(end_prompt).lower().strip() in ["nan", "", "(auto)"] or
                    "[Input: Start Frame]" in str(end_prompt)
                )
                
                if "[Input: Start Frame]" in str(end_prompt):
                    # Copy Start Frame as End Frame
                    start_file = os.path.join(DOWNLOAD_DIR, f"block_{block_id}_start.png")
                    end_file = os.path.join(DOWNLOAD_DIR, f"block_{block_id}_end.png")
                    if os.path.exists(start_file):
                        shutil.copy(start_file, end_file)
                        self.log(f"   ✅ Copied Start Frame as End Frame: block_{block_id}_end.png")
                        self.generated_images.append(f"block_{block_id}_end.png")
                    skip_end_frame = True
                
                if not skip_end_frame:
                    self.log(f"   🔄 [Protocol A] Generating End Frame...")
                    
                    saved_path_end = self._generate_image(
                        prompt=str(end_prompt),
                        reference_path=current_reference_path,
                        char_sheet_name=char_sheet_uploaded,
                        workflow_template=workflow_template,
                        node_map=node_map,
                        output_filename=f"block_{block_id}_end.png"
                    )
                    
                    if saved_path_end:
                        current_reference_path = saved_path_end
                        self.generated_images.append(f"block_{block_id}_end.png")
                    else:
                        self.log(f"   ⚠️ Failed to generate End Frame for Block {block_id}")
            
            # Small delay between blocks
            time.sleep(random.uniform(1.0, 2.0))

    def _generate_image(
        self,
        prompt: str,
        reference_path: str,
        char_sheet_name: str,
        workflow_template: dict,
        node_map: dict,
        output_filename: str
    ) -> str:
        """
        Generate a single image using ComfyUI.
        
        Args:
            prompt: Positive prompt text
            reference_path: Path to reference image
            char_sheet_name: Uploaded character sheet filename (or None)
            workflow_template: Base workflow dictionary
            node_map: Node ID mappings
            output_filename: Filename for saved output
            
        Returns:
            Path to saved image, or None if failed
        """
        try:
            # Clone workflow for modification
            workflow = clone_workflow(workflow_template)
            
            # Set prompts
            set_text_prompt(workflow, node_map["positive_prompt_node_id"], prompt)
            set_text_prompt(workflow, node_map["negative_prompt_node_id"], DEFAULT_NEGATIVE_PROMPT)
            
            # Randomize seed
            randomize_seed(workflow, node_map["sampler_node_id"], node_map.get("seed_input_name", "seed"))
            
            # Upload and set reference image
            if reference_path and os.path.exists(reference_path):
                try:
                    ref_name = self._client.upload_image(reference_path)
                    set_image_input(workflow, node_map["reference_image_node_id"], ref_name)
                    self.log(f"   📎 Ref: {os.path.basename(reference_path)}")
                except Exception as e:
                    self.log(f"   ⚠️ Reference upload failed: {e}")
            
            # Set character sheet if available and node exists
            if char_sheet_name and node_map.get("char_sheet_node_id"):
                try:
                    set_image_input(workflow, node_map["char_sheet_node_id"], char_sheet_name)
                except KeyError:
                    pass  # Node doesn't exist in workflow, skip
            
            # Queue the prompt
            self.log("   👉 Generating...")
            prompt_id = self._client.queue_prompt(workflow)
            
            # Wait for completion
            result = self._client.wait_for_completion(
                prompt_id,
                timeout=GENERATION_TIMEOUT,
                log_callback=self.log,
                reconnect_attempts=WS_RECONNECT_ATTEMPTS,
                reconnect_delay=WS_RECONNECT_DELAY
            )
            
            if not result:
                self.log("   ⚠️ Generation timed out or failed")
                return None
            
            # Download result
            output_path = os.path.join(DOWNLOAD_DIR, output_filename)
            self._client.download_image(
                filename=result['filename'],
                output_path=output_path,
                subfolder=result.get('subfolder', ''),
                image_type=result.get('type', 'output')
            )
            
            self.log(f"   ✅ Saved: {output_filename}")
            return output_path
            
        except Exception as e:
            self.log(f"   ❌ Generation error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _run_video_production(self, start_block=1):
        """
        Video production using ComfyUI API with WAN 2.2 VACE workflow.
        
        Uses Start Frame and End Frame images from image production
        to generate videos with motion prompts.
        """
        self.log("🎥 Starting Video Production (WAN 2.2 VACE)...")
        
        # Load workflow template
        try:
            workflow_path = os.path.join(os.getcwd(), WORKFLOW_VIDEO)
            workflow_template = load_workflow(workflow_path)
            self.log(f"   📋 Loaded workflow: {WORKFLOW_VIDEO}")
        except FileNotFoundError as e:
            self.log(f"❌ Workflow file not found: {e}")
            self.log("💡 Please export your video workflow from ComfyUI and save it to workflows/video_gen.json")
            return
        except Exception as e:
            self.log(f"❌ Failed to load workflow: {e}")
            return
        
        # Get node mappings
        try:
            node_map = get_node_map("video_gen")
        except KeyError as e:
            self.log(f"❌ Node mapping error: {e}")
            return
        
        # Validate node IDs
        missing_nodes = validate_node_ids(workflow_template, node_map)
        if missing_nodes:
            self.log(f"⚠️ Warning: Some node IDs not found in workflow: {missing_nodes}")
        
        df = pd.read_csv(PRODUCTION_TABLE)
        video_count = 0
        generated_videos = []
        
        for index, row in df.iterrows():
            if self.stop_requested:
                break
            
            block_id = str(row['Block']).zfill(2)
            if int(block_id) < start_block:
                continue
            
            self.current_block = f"Block {block_id} (Video)"
            
            # Check for required images
            start_image_path = os.path.join(DOWNLOAD_DIR, f"block_{block_id}_start.png")
            end_image_path = os.path.join(DOWNLOAD_DIR, f"block_{block_id}_end.png")
            
            if not os.path.exists(start_image_path):
                self.log(f"   ⚠️ Block {block_id}: Start frame not found, skipping...")
                continue
            
            # Get motion prompt
            motion_prompt = str(row.get('Kling O1 (Motion Prompt)', ''))
            if not motion_prompt or motion_prompt.lower() == 'nan':
                motion_prompt = "Smooth camera motion, cinematic movement"
            
            self.log(f"🎬 Processing Video Block {block_id}...")
            self.log(f"   📝 Prompt: {motion_prompt[:60]}...")
            
            try:
                # Clone workflow
                workflow = clone_workflow(workflow_template)
                
                # Set positive prompt
                if node_map.get("positive_prompt_node_id"):
                    set_text_prompt(workflow, node_map["positive_prompt_node_id"], motion_prompt)
                
                # Set negative prompt (empty for CFG=1)
                if node_map.get("negative_prompt_node_id"):
                    set_text_prompt(workflow, node_map["negative_prompt_node_id"], "")
                
                # Upload and set start frame
                if node_map.get("start_frame_node_id"):
                    try:
                        start_name = self._client.upload_image(start_image_path)
                        set_image_input(workflow, node_map["start_frame_node_id"], start_name)
                        self.log(f"   📎 Start Frame uploaded")
                    except Exception as e:
                        self.log(f"   ⚠️ Start frame upload failed: {e}")
                
                # Upload and set end frame (if exists)
                if node_map.get("end_frame_node_id") and os.path.exists(end_image_path):
                    try:
                        end_name = self._client.upload_image(end_image_path)
                        set_image_input(workflow, node_map["end_frame_node_id"], end_name)
                        self.log(f"   📎 End Frame uploaded")
                    except Exception as e:
                        self.log(f"   ⚠️ End frame upload failed: {e}")
                
                # Randomize seed
                if node_map.get("sampler_node_id"):
                    randomize_seed(workflow, node_map["sampler_node_id"], node_map.get("seed_input_name", "seed"))
                
                # Queue the prompt
                self.log("   👉 Generating video...")
                prompt_id = self._client.queue_prompt(workflow)
                
                # Wait for completion (video takes longer)
                result = self._client.wait_for_completion(
                    prompt_id,
                    timeout=GENERATION_TIMEOUT * 3,  # 3x timeout for video
                    log_callback=self.log,
                    reconnect_attempts=WS_RECONNECT_ATTEMPTS,
                    reconnect_delay=WS_RECONNECT_DELAY
                )
                
                if result:
                    # Download video result
                    output_filename = f"block_{block_id}_video.mp4"
                    output_path = os.path.join(DOWNLOAD_DIR, output_filename)
                    
                    try:
                        self._client.download_image(
                            filename=result['filename'],
                            output_path=output_path,
                            subfolder=result.get('subfolder', ''),
                            image_type=result.get('type', 'output')
                        )
                        self.log(f"   ✅ Saved: {output_filename}")
                        video_count += 1
                        generated_videos.append(output_path)
                    except Exception as e:
                        self.log(f"   ⚠️ Video download failed: {e}")
                else:
                    self.log(f"   ❌ Video generation failed for Block {block_id}")
                
                # Delay between videos
                time.sleep(random.uniform(2.0, 4.0))
                
            except Exception as e:
                self.log(f"   ❌ Error generating video for Block {block_id}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        self.log(f"🎉 Video Production Complete: {video_count} videos generated")
        
        # Offer to concatenate all videos
        if len(generated_videos) > 1:
            self.log(f"💡 To concatenate all videos, use:")
            self.log(f"   from upscale import concat_directory")
            self.log(f"   concat_directory('{DOWNLOAD_DIR}', 'block_*_video.mp4', 'final_video.mp4')")


# ==========================================
# 🏁 MAIN (Legacy CLI Support)
# ==========================================
if __name__ == "__main__":
    director = Director()
    print("========================================")
    print("   AI DIRECTOR v7.0 - ComfyUI Edition   ")
    print("========================================")
    print("1. 🖼️  Image Generation")
    print("2. 🎥 Video Generation (Coming Soon)")
    print("3. 🚀 Full Pipeline")
    
    choice = input("\nSelect Mode (1/2/3): ").strip()
    
    sb = input("Start Block? (Default 1): ")
    try:
        sb_int = int(sb)
    except:
        sb_int = 1

    if choice == "1":
        director.start("image", start_block=sb_int)
    elif choice == "2":
        director.start("video", start_block=sb_int)
    elif choice == "3":
        director.start("full", start_block=sb_int)
    else:
        print("❌ Invalid Choice.")
    
    # Keep main thread alive while director runs
    while director.is_running:
        try:
            time.sleep(1)
        except KeyboardInterrupt:
            director.stop()
            break