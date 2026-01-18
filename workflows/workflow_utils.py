"""
Workflow Utilities - JSON Manipulation for ComfyUI Workflows

이 모듈은 ComfyUI 워크플로우 JSON을 로드하고 동적으로 값을 수정하는 유틸리티를 제공합니다.
"""

import json
import random
import copy
from pathlib import Path
from typing import Any, Optional, Union


def load_workflow(workflow_path: str) -> dict:
    """
    Load a workflow JSON file.
    
    Args:
        workflow_path: Path to the workflow JSON file
        
    Returns:
        Workflow dictionary in ComfyUI API format
        
    Raises:
        FileNotFoundError: If workflow file doesn't exist
        json.JSONDecodeError: If file is not valid JSON
    """
    path = Path(workflow_path)
    if not path.exists():
        raise FileNotFoundError(f"Workflow file not found: {workflow_path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        workflow = json.load(f)
    
    return workflow


def save_workflow(workflow: dict, output_path: str) -> None:
    """
    Save a workflow dictionary to JSON file.
    
    Args:
        workflow: Workflow dictionary
        output_path: Path to save the JSON file
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(workflow, f, indent=2, ensure_ascii=False)


def get_node(workflow: dict, node_id: str) -> Optional[dict]:
    """
    Get a node from the workflow by its ID.
    
    Args:
        workflow: Workflow dictionary
        node_id: Node ID (string)
        
    Returns:
        Node dictionary or None if not found
    """
    return workflow.get(str(node_id))


def set_node_input(
    workflow: dict, 
    node_id: str, 
    input_name: str, 
    value: Any
) -> dict:
    """
    Set an input value for a specific node.
    
    Args:
        workflow: Workflow dictionary (will be modified in place)
        node_id: Target node ID
        input_name: Name of the input to modify
        value: New value to set
        
    Returns:
        Modified workflow dictionary
        
    Raises:
        KeyError: If node_id doesn't exist in workflow
    """
    node_id = str(node_id)
    if node_id not in workflow:
        raise KeyError(f"Node ID '{node_id}' not found in workflow. Available nodes: {list(workflow.keys())[:10]}...")
    
    node = workflow[node_id]
    
    # Handle different node structures
    if "inputs" in node:
        node["inputs"][input_name] = value
    else:
        # Some nodes have inputs at top level
        node[input_name] = value
    
    return workflow


def remove_node_input(
    workflow: dict, 
    node_id: str, 
    input_name: str
) -> dict:
    """
    Remove an input from a specific node (for optional inputs).
    
    This is useful for removing optional inputs like 'end_image' in 
    WanVideoVACEStartToEndFrame when Protocol B (Open Run) is used.
    
    Args:
        workflow: Workflow dictionary (will be modified in place)
        node_id: Target node ID
        input_name: Name of the input to remove
        
    Returns:
        Modified workflow dictionary
    """
    node_id = str(node_id)
    if node_id not in workflow:
        return workflow  # Node doesn't exist, skip silently
    
    node = workflow[node_id]
    
    # Handle different node structures
    if "inputs" in node and input_name in node["inputs"]:
        del node["inputs"][input_name]
    elif input_name in node:
        del node[input_name]
    
    return workflow


def set_text_prompt(workflow: dict, node_id: str, prompt: str) -> dict:
    """
    Set text prompt for a CLIP Text Encode node.
    
    Args:
        workflow: Workflow dictionary
        node_id: CLIP Text Encode node ID
        prompt: Text prompt to set
        
    Returns:
        Modified workflow
    """
    return set_node_input(workflow, node_id, "text", prompt)


def set_image_input(workflow: dict, node_id: str, filename: str, subfolder: str = "") -> dict:
    """
    Set image filename for a LoadImage node.
    
    The filename should be the name returned by ComfyUIClient.upload_image()
    
    Args:
        workflow: Workflow dictionary
        node_id: LoadImage node ID
        filename: Uploaded image filename (in ComfyUI's input folder)
        subfolder: Optional subfolder (if image was uploaded to subfolder)
        
    Returns:
        Modified workflow
    """
    node_id = str(node_id)
    if node_id not in workflow:
        raise KeyError(f"Node ID '{node_id}' not found in workflow")
    
    node = workflow[node_id]
    
    if "inputs" in node:
        # Standard LoadImage node structure
        if subfolder:
            node["inputs"]["image"] = f"{subfolder}/{filename}"
        else:
            node["inputs"]["image"] = filename
    
    return workflow


def randomize_seed(workflow: dict, node_id: str, input_name: str = "seed") -> dict:
    """
    Set a random seed value for a sampler node.
    
    Args:
        workflow: Workflow dictionary
        node_id: Sampler node ID (KSampler, etc.)
        input_name: Name of the seed input (default: "seed")
        
    Returns:
        Modified workflow with random seed
    """
    random_seed = random.randint(0, 2**32 - 1)
    return set_node_input(workflow, node_id, input_name, random_seed)


def set_seed(workflow: dict, node_id: str, seed: int, input_name: str = "seed") -> dict:
    """
    Set a specific seed value for reproducibility.
    
    Args:
        workflow: Workflow dictionary
        node_id: Sampler node ID
        seed: Seed value to set
        input_name: Name of the seed input
        
    Returns:
        Modified workflow
    """
    return set_node_input(workflow, node_id, input_name, seed)


def validate_node_ids(workflow: dict, node_map: dict) -> list:
    """
    Validate that all node IDs in the mapping exist in the workflow.
    
    Args:
        workflow: Workflow dictionary
        node_map: Node ID mapping from config.py
        
    Returns:
        List of missing node IDs (empty if all valid)
    """
    missing = []
    workflow_nodes = set(str(k) for k in workflow.keys())
    
    for key, node_id in node_map.items():
        if node_id == "TBD":
            continue  # Skip unconfigured nodes
        if not node_id:
            continue  # Skip empty/None values
        if str(node_id) not in workflow_nodes:
            missing.append(f"{key} (ID: {node_id})")
    
    return missing


def get_workflow_info(workflow: dict) -> dict:
    """
    Get summary information about a workflow.
    
    Args:
        workflow: Workflow dictionary
        
    Returns:
        Dictionary with workflow statistics and node types
    """
    node_types = {}
    for node_id, node in workflow.items():
        if isinstance(node, dict):
            class_type = node.get("class_type", "unknown")
            if class_type not in node_types:
                node_types[class_type] = []
            node_types[class_type].append(node_id)
    
    return {
        "total_nodes": len(workflow),
        "node_types": node_types,
        "node_ids": list(workflow.keys())
    }


def clone_workflow(workflow: dict) -> dict:
    """
    Create a deep copy of a workflow for safe modification.
    
    Args:
        workflow: Original workflow dictionary
        
    Returns:
        Deep copy of the workflow
    """
    return copy.deepcopy(workflow)


def find_nodes_by_type(workflow: dict, class_type: str) -> list:
    """
    Find all nodes of a specific class type.
    
    Args:
        workflow: Workflow dictionary
        class_type: Node class type (e.g., "KSampler", "CLIPTextEncode")
        
    Returns:
        List of (node_id, node_dict) tuples
    """
    found = []
    for node_id, node in workflow.items():
        if isinstance(node, dict) and node.get("class_type") == class_type:
            found.append((str(node_id), node))
    return found


def auto_detect_node_ids(workflow: dict) -> dict:
    """
    Attempt to auto-detect common node IDs based on class types.
    
    This is a helper function to assist users in setting up their config.py.
    
    Args:
        workflow: Workflow dictionary
        
    Returns:
        Dictionary of suggested node ID mappings
    """
    suggestions = {}
    
    # Find CLIP Text Encode nodes
    clip_nodes = find_nodes_by_type(workflow, "CLIPTextEncode")
    if len(clip_nodes) >= 1:
        suggestions["positive_prompt_node_id"] = clip_nodes[0][0]
    if len(clip_nodes) >= 2:
        suggestions["negative_prompt_node_id"] = clip_nodes[1][0]
    
    # Find KSampler nodes
    sampler_nodes = find_nodes_by_type(workflow, "KSampler")
    if not sampler_nodes:
        sampler_nodes = find_nodes_by_type(workflow, "SamplerCustom")
    if sampler_nodes:
        suggestions["sampler_node_id"] = sampler_nodes[0][0]
    
    # Find LoadImage nodes
    load_image_nodes = find_nodes_by_type(workflow, "LoadImage")
    if len(load_image_nodes) >= 1:
        suggestions["reference_image_node_id"] = load_image_nodes[0][0]
    if len(load_image_nodes) >= 2:
        suggestions["char_sheet_node_id"] = load_image_nodes[1][0]
    
    # Find SaveImage nodes
    save_nodes = find_nodes_by_type(workflow, "SaveImage")
    if not save_nodes:
        save_nodes = find_nodes_by_type(workflow, "PreviewImage")
    if save_nodes:
        suggestions["save_image_node_id"] = save_nodes[0][0]
    
    return suggestions


# ============================================================
# VIDEO WORKFLOW UTILITIES
# ============================================================

def set_video_start_frame(workflow: dict, node_id: str, filename: str) -> dict:
    """
    Set the start frame image for video generation.
    
    Args:
        workflow: Workflow dictionary
        node_id: Start frame LoadImage node ID
        filename: Uploaded image filename
        
    Returns:
        Modified workflow
    """
    return set_image_input(workflow, node_id, filename)


def set_video_end_frame(workflow: dict, node_id: str, filename: str) -> dict:
    """
    Set the end frame image for video generation (optional for I2V).
    
    Args:
        workflow: Workflow dictionary
        node_id: End frame LoadImage node ID
        filename: Uploaded image filename
        
    Returns:
        Modified workflow
    """
    return set_image_input(workflow, node_id, filename)


def set_num_frames(workflow: dict, node_id: str, num_frames: int) -> dict:
    """
    Set the number of frames for video generation.
    
    Args:
        workflow: Workflow dictionary
        node_id: PrimitiveInt node ID for frame count
        num_frames: Number of frames to generate (e.g., 81 for ~5 seconds at 16fps)
        
    Returns:
        Modified workflow
    """
    node_id = str(node_id)
    if node_id not in workflow:
        raise KeyError(f"Node ID '{node_id}' not found in workflow")
    
    node = workflow[node_id]
    
    if "inputs" in node:
        node["inputs"]["value"] = num_frames
    
    return workflow


def set_video_output_prefix(workflow: dict, node_id: str, prefix: str) -> dict:
    """
    Set the output filename prefix for VHS_VideoCombine node.
    
    Args:
        workflow: Workflow dictionary
        node_id: VHS_VideoCombine node ID
        prefix: Filename prefix (e.g., "block_001/video")
        
    Returns:
        Modified workflow
    """
    node_id = str(node_id)
    if node_id not in workflow:
        raise KeyError(f"Node ID '{node_id}' not found in workflow")
    
    node = workflow[node_id]
    
    if "inputs" in node:
        node["inputs"]["filename_prefix"] = prefix
    
    return workflow


def configure_video_workflow(
    workflow: dict,
    node_map: dict,
    prompt: str,
    start_frame_filename: Optional[str] = None,
    end_frame_filename: Optional[str] = None,
    num_frames: int = 81,
    output_prefix: str = "output/video",
    seed: Optional[int] = None
) -> dict:
    """
    Configure a video workflow with all necessary parameters.
    
    Args:
        workflow: Workflow dictionary (will be cloned)
        node_map: Node ID mapping from config.py
        prompt: Motion/action prompt
        start_frame_filename: Start frame image filename (optional)
        end_frame_filename: End frame image filename (optional)
        num_frames: Number of frames to generate (default: 81 = ~5 sec at 16fps)
        output_prefix: Output filename prefix
        seed: Random seed (None for random)
        
    Returns:
        Configured workflow copy
    """
    wf = clone_workflow(workflow)
    
    # Set positive prompt
    if "positive_prompt_node_id" in node_map:
        set_text_prompt(wf, node_map["positive_prompt_node_id"], prompt)
    
    # Set negative prompt (usually blank for CFG 1)
    if "negative_prompt_node_id" in node_map:
        set_text_prompt(wf, node_map["negative_prompt_node_id"], "")
    
    # Set start frame
    if start_frame_filename and "start_frame_node_id" in node_map:
        set_video_start_frame(wf, node_map["start_frame_node_id"], start_frame_filename)
    
    # Set end frame
    if end_frame_filename and "end_frame_node_id" in node_map:
        set_video_end_frame(wf, node_map["end_frame_node_id"], end_frame_filename)
    
    # Set number of frames
    if "num_frames_node_id" in node_map:
        set_num_frames(wf, node_map["num_frames_node_id"], num_frames)
    
    # Set output prefix
    if "video_combine_node_id" in node_map:
        set_video_output_prefix(wf, node_map["video_combine_node_id"], output_prefix)
    
    # Set seed
    if "sampler_node_id" in node_map:
        seed_input = node_map.get("seed_input_name", "seed")
        if seed is not None:
            set_seed(wf, node_map["sampler_node_id"], seed, seed_input)
        else:
            randomize_seed(wf, node_map["sampler_node_id"], seed_input)
    
    return wf


# CLI helper for workflow analysis
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python workflow_utils.py <workflow.json>")
        print("       Analyzes a workflow and suggests node ID mappings")
        sys.exit(1)
    
    workflow_path = sys.argv[1]
    
    try:
        wf = load_workflow(workflow_path)
        info = get_workflow_info(wf)
        
        print(f"\n📋 Workflow Analysis: {workflow_path}")
        print(f"   Total Nodes: {info['total_nodes']}")
        print(f"\n🔧 Node Types Found:")
        for node_type, ids in sorted(info['node_types'].items()):
            print(f"   - {node_type}: {ids}")
        
        print(f"\n💡 Suggested Node ID Mappings for config.py:")
        suggestions = auto_detect_node_ids(wf)
        for key, value in suggestions.items():
            print(f'   "{key}": "{value}",')
        
        if not suggestions:
            print("   (Could not auto-detect - please configure manually)")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
