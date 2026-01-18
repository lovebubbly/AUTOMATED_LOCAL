# Workflows Package
# This package contains workflow JSON files and configuration for ComfyUI

from .config import NODE_MAP, DEFAULT_NEGATIVE_PROMPT, get_node_map, validate_node_map
from .workflow_utils import (
    load_workflow,
    save_workflow,
    set_node_input,
    remove_node_input,
    set_text_prompt,
    set_image_input,
    randomize_seed,
    set_seed,
    validate_node_ids,
    clone_workflow,
    find_nodes_by_type,
    auto_detect_node_ids,
    get_workflow_info
)

__all__ = [
    # Config
    'NODE_MAP',
    'DEFAULT_NEGATIVE_PROMPT',
    'get_node_map',
    'validate_node_map',
    # Utils
    'load_workflow',
    'save_workflow',
    'set_node_input',
    'remove_node_input',
    'set_text_prompt',
    'set_image_input',
    'randomize_seed',
    'set_seed',
    'validate_node_ids',
    'clone_workflow',
    'find_nodes_by_type',
    'auto_detect_node_ids',
    'get_workflow_info'
]
