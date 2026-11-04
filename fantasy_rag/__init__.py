"""
Fantasy Assets RAG System

CLIP-based visual search for fantasy 3D assets (STL files).
Pipeline: STL -> Blender render -> CLIP embedding -> ChromaDB -> search.
"""

__version__ = "0.2.0"

from .config import settings
from .utils import file_utils, mesh_utils, image_utils

__all__ = [
    "settings",
    "file_utils",
    "mesh_utils",
    "image_utils",
]
