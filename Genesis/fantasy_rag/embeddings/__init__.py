#!/usr/bin/env python3
"""
Fantasy RAG Embeddings Module

This module handles text and multimodal embeddings for fantasy asset descriptions,
enabling semantic search and similarity matching.
"""

from .text_embedder import TextEmbedder, EmbeddingConfig
from .similarity_search import SimilaritySearchEngine, SearchResult, SearchQuery
from .multilingual_search import MultilingualSearchEngine

__all__ = [
    'TextEmbedder',
    'EmbeddingConfig', 
    'SimilaritySearchEngine',
    'SearchResult',
    'SearchQuery',
    'MultilingualSearchEngine'
]

__version__ = "1.0.0"
