import pandas as pd
import os
import shutil

# Configuration
IMAGE_DIR = os.path.join(os.getcwd(), "assets", "images")
CSV_PATH = os.path.join(os.getcwd(), "assets", "production_table.csv")

def cleanup():
    print(f"🧹 Starting Cleanup in {IMAGE_DIR}...")
    
    if not os.path.exists(CSV_PATH):
        print("❌ CSV not found!")
        return

    df = pd.read_csv(CSV_PATH)
    
    # We need to track the 'previous valid output' to copy from
    # Logic: simple iteration. If current is placeholder, copy prev.
    # If current is normal, update prev.
    
    last_valid_image = None
    
    # Sort just in case
    # df = df.sort_values(by="Block") 

    for index, row in df.iterrows():
        block_id = str(row['Block']).zfill(2)
        start_prompt = str(row['Nano Banana (Start Frame)'])
        
        # Expected filenames for this block
        current_start_img = os.path.join(IMAGE_DIR, f"block_{block_id}_start.png")
        current_end_img = os.path.join(IMAGE_DIR, f"block_{block_id}_end.png")
        
        protocol = str(row['Protocol']).strip()
        
        # 1. Check if this block was a "Last Frame" placeholder
        if "[Input: Last Frame]" in start_prompt:
            print(f"🔍 Block {block_id}: Detect '[Input: Last Frame]'...")
            
            if last_valid_image and os.path.exists(last_valid_image):
                print(f"   ♻️  Overwriting garbage image with: {os.path.basename(last_valid_image)}")
                shutil.copy(last_valid_image, current_start_img)
                
                # Now this 'new' header image becomes the valid one for this block (effectively same as prev)
                last_valid_image = current_start_img 
            else:
                print(f"   ⚠️ No previous image found to copy from! Skipping.")
        else:
            # Normal Block - Just update tracking logic
            # If Protocol A, expecting End Frame to be the final output of this block
            # If Protocol B, expecting Start Frame to be the final output
            
            # Check what actually exists on disk to be safe
            if os.path.exists(current_end_img):
                 last_valid_image = current_end_img
            elif os.path.exists(current_start_img):
                 last_valid_image = current_start_img
            else:
                # If neither exists (maybe not generated yet), retain previous or set None?
                # Probably keep previous if this block hasn't run. 
                # But if we assume this is a cleanup of *badly generated* stuff, 
                # we should probably trust that files exist if rows are processed.
                pass

    print("✅ Cleanup Finished.")

if __name__ == "__main__":
    cleanup()
