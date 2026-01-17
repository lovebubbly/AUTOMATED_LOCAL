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
    # Image Generation Workflow (image_gen.json) - Qwen Image Edit
    # -----------------------------------------------------------
    "image_gen": {
        # Qwen Image Edit 2511 Workflow (API Format)
        # Text Encoding Nodes
        "positive_prompt_node_id": "91:68",   # TextEncodeQwenImageEditPlus (Positive)
        "negative_prompt_node_id": "91:69",   # TextEncodeQwenImageEditPlus (Negative)
        
        # Sampler / Generator Node
        "sampler_node_id": "91:65",           # KSampler
        "seed_input_name": "seed",
        
        # Image Input Nodes
        "source_image_node_id": "41",         # LoadImage - Source Image (image1)
        "reference_image_node_id": "83",      # LoadImage - Reference/Material Image (image2)
        
        # Output Node
        "save_image_node_id": "92",           # SaveImage
        
        # Model Nodes (for reference)
        "clip_node_id": "91:61",              # CLIPLoader (qwen_2.5_vl_7b)
        "vae_node_id": "91:10",               # VAELoader
        "unet_node_id": "91:89",              # UnetLoaderGGUF
        "lora_node_id": "91:74",              # LoraLoaderModelOnly
    },
    
    # -----------------------------------------------------------
    # Video Generation Workflow (video_gen.json) - WAN 2.2 VACE
    # -----------------------------------------------------------
    "video_gen": {
        # WAN 2.2 VACE I2V Workflow (API Format)
        "positive_prompt_node_id": "9",       # CLIPTextEncode - Positive Prompt
        "negative_prompt_node_id": "10",      # CLIPTextEncode - Negative Prompt (blank for CFG 1)
        "start_frame_node_id": "16",          # LoadImage - Start Frame
        "end_frame_node_id": "37",            # LoadImage - End Frame
        "num_frames_node_id": "48",           # PrimitiveInt - Number of Frames
        "vace_node_id": "34",                 # WanVideoVACEStartToEndFrame
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
# VIDEO UPSCALE SETTINGS
# ============================================================

UPSCALE_CONFIG = {
    "enabled": True,                    # 업스케일 활성화 여부
    "scale": 2,                         # 배율 (2 = 720p→1440p, 4 = 720p→4K)
    "model": "realesr-animevideov3",    # 모델명 (AI 생성 영상에 최적화)
    "output_suffix": "_1440p",          # 출력 파일 접미사
    "gpu_id": 0,                        # GPU ID
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
