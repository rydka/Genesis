#!/usr/bin/env python3
"""
Fantasy RAG Database Management Module

This module provides unified database management for fantasy assets,
integrating rendering, descriptions, embeddings, and search capabilities.
"""

from .schema_definitions import (
    AssetRecord, DescriptionRecord, RenderRecord, EmbeddingRecord,
    AssetStatus, AssetCategory, ProcessingStage
)
from .chromadb_manager import ChromaDBManager, DatabaseConfig
from .indexing_pipeline import IndexingPipeline, PipelineConfig, ProcessingResult

__all__ = [
    'AssetRecord',
    'DescriptionRecord', 
    'RenderRecord',
    'EmbeddingRecord',
    'AssetStatus',
    'AssetCategory',
    'ProcessingStage',
    'ChromaDBManager',
    'DatabaseConfig',
    'IndexingPipeline',
    'PipelineConfig',
    'ProcessingResult'
]

__version__ = "1.0.0"
