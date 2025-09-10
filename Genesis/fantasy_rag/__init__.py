"""
Fantasy Assets RAG System
========================

A comprehensive RAG system for fantasy 3D assets with multilingual descriptions,
visual embeddings, and MeshFormer integration.

Author: Genesis Project
Date: January 2025
"""

__version__ = "0.1.0"
__author__ = "Genesis Project"

# Core modules
from .config import settings
from .utils import file_utils, mesh_utils, image_utils

__all__ = [
    "settings",
    "file_utils", 
    "mesh_utils",
    "image_utils"
]
