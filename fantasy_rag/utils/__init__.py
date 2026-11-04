"""
Utilities module for Fantasy Assets RAG
"""

from . import file_utils
from . import mesh_utils  
from . import image_utils

# Import commonly used functions
from .file_utils import (
    get_stl_files,
    get_file_info,
    create_asset_id,
    safe_filename,
    ensure_directory,
    save_json,
    load_json
)

from .mesh_utils import (
    load_stl_mesh,
    extract_mesh_metadata,
    validate_stl_file,
    MeshMetadata
)

from .image_utils import (
    load_image,
    save_image,
    resize_image,
    create_thumbnail,
    analyze_image_properties,
    create_image_grid
)

__all__ = [
    # Modules
    "file_utils",
    "mesh_utils", 
    "image_utils",
    
    # File utilities
    "get_stl_files",
    "get_file_info",
    "create_asset_id",
    "safe_filename",
    "ensure_directory",
    "save_json",
    "load_json",
    
    # Mesh utilities
    "load_stl_mesh",
    "extract_mesh_metadata", 
    "validate_stl_file",
    "MeshMetadata",
    
    # Image utilities
    "load_image",
    "save_image",
    "resize_image",
    "create_thumbnail",
    "analyze_image_properties",
    "create_image_grid"
]
