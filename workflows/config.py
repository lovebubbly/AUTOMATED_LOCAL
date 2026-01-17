"""
Workflow Configuration - Node ID Mappings

이 파일에서 ComfyUI 워크플로우의 각 노드 ID를 정의합니다.
워크플로우 JSON을 ComfyUI에서 Export한 후, 아래 매핑을 해당 워크플로우에 맞게 업데이트하세요.

Node ID 확인 방법:
1. ComfyUI에서 워크플로우를 열기
2. 노드 우클릭 → "Copy Node ID" 또는
3. API Format으로 Export하여 JSON에서 직접 확인
"""

# ============================================================
# NODE ID MAPPINGS
# ============================================================

NODE_MAP = {
    # -----------------------------------------------------------
    # Image Generation Workflow (image_gen.json)
    # -----------------------------------------------------------
    "image_gen": {
        # Text Encoding Nodes
        "positive_prompt_node_id": "6",      # CLIP Text Encode (Positive Prompt)
        "negative_prompt_node_id": "7",      # CLIP Text Encode (Negative Prompt)
        
        # Sampler / Generator Node
        "sampler_node_id": "3",              # KSampler, SamplerCustom, etc.
        "seed_input_name": "seed",           # Input name for seed (usually "seed")
        
        # Image Input Nodes
        "reference_image_node_id": "10",     # LoadImage - Rolling Reference
        "char_sheet_node_id": "12",          # LoadImage - Character Sheet (Optional)
        
        # Output Node
        "save_image_node_id": "9",           # SaveImage or PreviewImage
    },
    
    # -----------------------------------------------------------
    # Video Generation Workflow (video_gen.json) - WAN 2.2 VACE
    # -----------------------------------------------------------
    "video_gen": {
        # WAN 2.2 VACE T2V/I2V Workflow (API Format)
        "positive_prompt_node_id": "9",       # CLIPTextEncode - Positive Prompt
        "negative_prompt_node_id": "10",      # CLIPTextEncode - Negative Prompt (blank for CFG 1)
        "start_frame_node_id": None,          # LoadImage - Start Frame (not in current workflow)
        "end_frame_node_id": None,            # LoadImage - End Frame (not in current workflow)
        "num_frames_node_id": "48",           # PrimitiveInt - Number of Frames
        "vace_to_video_node_id": "28",        # WanVaceToVideo (width, height, length, strength)
        "sampler_node_id": "8",               # KSampler
        "seed_input_name": "seed",
        "checkpoint_node_id": "26",           # CheckpointLoaderSimple
        "vae_decode_node_id": "11",           # VAEDecode
        "video_combine_node_id": "39",        # VHS_VideoCombine - Output
    }
}


# ============================================================
# DEFAULT PROMPTS
# ============================================================

# Default negative prompt for image generation
DEFAULT_NEGATIVE_PROMPT = (
    "blurry, low quality, watermark, text, signature, "
    "jpeg artifacts, pixelated, noise, grain, "
    "deformed, disfigured, bad anatomy, extra limbs"
)

# Default negative prompt for video generation
DEFAULT_VIDEO_NEGATIVE_PROMPT = (
    "static, frozen, no motion, blurry, low quality, "
    "watermark, text, jittery, stuttering"
)


# ============================================================
# WORKFLOW FILE PATHS (relative to project root)
# ============================================================

WORKFLOW_PATHS = {
    "image_gen": "workflows/image_gen.json",
    "video_gen": "workflows/video_gen.json",
}


# ============================================================
# VALIDATION HELPERS
# ============================================================

def get_node_map(workflow_type: str) -> dict:
    """
    Get node mapping for a specific workflow type.
    
    Args:
        workflow_type: "image_gen" or "video_gen"
        
    Returns:
        Node ID mapping dictionary
        
    Raises:
        KeyError: If workflow_type is not defined
    """
    if workflow_type not in NODE_MAP:
        raise KeyError(f"Unknown workflow type: {workflow_type}. Available: {list(NODE_MAP.keys())}")
    return NODE_MAP[workflow_type]


def validate_node_map(workflow_type: str) -> list:
    """
    Check for any TBD (undefined) node IDs in a mapping.
    
    Args:
        workflow_type: "image_gen" or "video_gen"
        
    Returns:
        List of node keys that are still "TBD"
    """
    node_map = get_node_map(workflow_type)
    undefined = []
    for key, value in node_map.items():
        if value == "TBD":
            undefined.append(key)
    return undefined
