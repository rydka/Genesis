"""
Configuration module for Fantasy Assets RAG
"""

from .settings import (
    settings,
    FantasyRAGConfig,
    RenderingConfig,
    LLMConfig,
    EmbeddingConfig,
    DatabaseConfig,
    BlenderConfig,
    create_directories,
    # Path constants
    PROJECT_ROOT,
    RESOURCES_DIR,
    DATA_DIR,
    ASSETS_DIR,
    RENDERS_DIR,
    CACHE_DIR,
    DATABASE_DIR,
    LOGS_DIR
)

__all__ = [
    "settings",
    "FantasyRAGConfig",
    "RenderingConfig", 
    "LLMConfig",
    "EmbeddingConfig",
    "DatabaseConfig",
    "BlenderConfig",
    "create_directories",
    "PROJECT_ROOT",
    "RESOURCES_DIR",
    "DATA_DIR",
    "ASSETS_DIR",
    "RENDERS_DIR",
    "CACHE_DIR",
    "DATABASE_DIR",
    "LOGS_DIR"
]
