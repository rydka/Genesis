#!/usr/bin/env python3
"""
Schema Definitions for Fantasy Asset Database

This module defines the data structures and schemas for storing
fantasy assets, descriptions, renders, and embeddings.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from pathlib import Path
import time
import uuid

class AssetStatus(Enum):
    """Status of asset processing"""
    DISCOVERED = "discovered"
    RENDERING = "rendering"
    RENDERED = "rendered"
    DESCRIBING = "describing"
    DESCRIBED = "described"
    EMBEDDING = "embedding"
    INDEXED = "indexed"
    FAILED = "failed"
    ARCHIVED = "archived"

class AssetCategory(Enum):
    """Categories of fantasy assets"""
    WEAPON = "weapon"
    ARMOR = "armor"
    CHARACTER = "character"
    CREATURE = "creature"
    MAGICAL_ITEM = "magical_item"
    ENVIRONMENT = "environment"
    VEHICLE = "vehicle"
    UNKNOWN = "unknown"

class ProcessingStage(Enum):
    """Stages of asset processing pipeline"""
    DISCOVERY = "discovery"
    RENDERING = "rendering"
    DESCRIPTION = "description"
    EMBEDDING = "embedding"
    INDEXING = "indexing"
    COMPLETE = "complete"

@dataclass
class AssetRecord:
    """Complete record for a fantasy asset"""
    
    # Core identification
    asset_id: str
    original_filename: str
    file_path: Path
    file_hash: str
    
    # Asset metadata
    category: AssetCategory = AssetCategory.UNKNOWN
    status: AssetStatus = AssetStatus.DISCOVERED
    current_stage: ProcessingStage = ProcessingStage.DISCOVERY
    
    # File information
    file_size: int = 0
    file_format: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    
    # Processing metadata
    processing_attempts: int = 0
    last_error: Optional[str] = None
    processing_time: float = 0.0
    
    # Custom metadata
    tags: List[str] = field(default_factory=list)
    custom_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization processing"""
        if isinstance(self.file_path, str):
            self.file_path = Path(self.file_path)
        
        if not self.asset_id:
            self.asset_id = str(uuid.uuid4())
    
    def update_status(self, status: AssetStatus, stage: ProcessingStage = None):
        """Update asset status and stage"""
        self.status = status
        if stage:
            self.current_stage = stage
        self.updated_at = time.time()
    
    def add_error(self, error_message: str):
        """Add error information"""
        self.last_error = error_message
        self.status = AssetStatus.FAILED
        self.processing_attempts += 1
        self.updated_at = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        # Convert lists to strings for ChromaDB compatibility
        tags_str = ",".join(self.tags) if self.tags else ""

        result = {
            "asset_id": self.asset_id,
            "original_filename": self.original_filename,
            "file_path": str(self.file_path),
            "file_hash": self.file_hash,
            "category": self.category.value,
            "status": self.status.value,
            "current_stage": self.current_stage.value,
            "file_size": self.file_size,
            "file_format": self.file_format,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "processing_attempts": self.processing_attempts,
            "last_error": self.last_error or "",
            "processing_time": self.processing_time,
            "tags": tags_str
        }

        # Add custom metadata as individual fields (flatten)
        for key, value in self.custom_metadata.items():
            # Ensure ChromaDB compatible types
            if isinstance(value, (str, int, float, bool)) or value is None:
                result[f"custom_{key}"] = value
            else:
                result[f"custom_{key}"] = str(value)

        return result

@dataclass
class RenderRecord:
    """Record for rendered views of an asset"""
    
    render_id: str
    asset_id: str
    view_name: str  # front, back, left, right, top, iso_3d
    image_path: Path
    
    # Render settings
    resolution: tuple = (1024, 1024)
    samples: int = 128
    render_engine: str = "cycles"
    
    # Processing info
    render_time: float = 0.0
    file_size: int = 0
    created_at: float = field(default_factory=time.time)
    
    # Quality metrics
    image_quality_score: Optional[float] = None
    has_transparency: bool = False
    dominant_colors: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Post-initialization processing"""
        if isinstance(self.image_path, str):
            self.image_path = Path(self.image_path)
        
        if not self.render_id:
            self.render_id = f"{self.asset_id}_{self.view_name}_{uuid.uuid4().hex[:8]}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        # Convert tuple to string for ChromaDB compatibility
        resolution_str = f"{self.resolution[0]}x{self.resolution[1]}"
        colors_str = ",".join(self.dominant_colors) if self.dominant_colors else ""

        return {
            "render_id": self.render_id,
            "asset_id": self.asset_id,
            "view_name": self.view_name,
            "image_path": str(self.image_path),
            "resolution": resolution_str,
            "samples": self.samples,
            "render_engine": self.render_engine,
            "render_time": self.render_time,
            "file_size": self.file_size,
            "created_at": self.created_at,
            "image_quality_score": self.image_quality_score or 0.0,
            "has_transparency": self.has_transparency,
            "dominant_colors": colors_str
        }

@dataclass
class DescriptionRecord:
    """Record for generated descriptions"""
    
    description_id: str
    asset_id: str
    language: str
    description_text: str
    
    # Generation metadata
    model_name: str = ""
    generation_time: float = 0.0
    word_count: int = 0
    target_length: int = 200
    
    # Quality metrics
    quality_score: Optional[float] = None
    similarity_to_target: Optional[float] = None
    language_confidence: Optional[float] = None
    
    # Processing info
    created_at: float = field(default_factory=time.time)
    prompt_used: Optional[str] = None
    views_analyzed: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Post-initialization processing"""
        if not self.description_id:
            self.description_id = f"{self.asset_id}_{self.language}_{uuid.uuid4().hex[:8]}"
        
        if not self.word_count and self.description_text:
            self.word_count = len(self.description_text.split())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        # Convert list to string for ChromaDB compatibility
        views_str = ",".join(self.views_analyzed) if self.views_analyzed else ""

        return {
            "description_id": self.description_id,
            "asset_id": self.asset_id,
            "language": self.language,
            "description_text": self.description_text,
            "model_name": self.model_name,
            "generation_time": self.generation_time,
            "word_count": self.word_count,
            "target_length": self.target_length,
            "quality_score": self.quality_score or 0.0,
            "similarity_to_target": self.similarity_to_target or 0.0,
            "language_confidence": self.language_confidence or 0.0,
            "created_at": self.created_at,
            "prompt_used": self.prompt_used or "",
            "views_analyzed": views_str
        }

@dataclass
class EmbeddingRecord:
    """Record for text embeddings"""
    
    embedding_id: str
    description_id: str
    asset_id: str
    
    # Embedding data
    embedding_vector: List[float]
    embedding_model: str
    embedding_dimension: int
    
    # Processing info
    generation_time: float = 0.0
    created_at: float = field(default_factory=time.time)
    normalized: bool = True
    
    # Metadata for search
    language: str = ""
    text_length: int = 0
    
    def __post_init__(self):
        """Post-initialization processing"""
        if not self.embedding_id:
            self.embedding_id = f"{self.description_id}_emb_{uuid.uuid4().hex[:8]}"
        
        if not self.embedding_dimension and self.embedding_vector:
            self.embedding_dimension = len(self.embedding_vector)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "embedding_id": self.embedding_id,
            "description_id": self.description_id,
            "asset_id": self.asset_id,
            "embedding_vector": self.embedding_vector,
            "embedding_model": self.embedding_model,
            "embedding_dimension": self.embedding_dimension,
            "generation_time": self.generation_time,
            "created_at": self.created_at,
            "normalized": self.normalized,
            "language": self.language,
            "text_length": self.text_length
        }

@dataclass
class ProcessingLog:
    """Log entry for processing pipeline"""
    
    log_id: str
    asset_id: str
    stage: ProcessingStage
    status: str  # started, completed, failed
    
    # Timing
    started_at: float
    completed_at: Optional[float] = None
    duration: Optional[float] = None
    
    # Details
    message: str = ""
    error_details: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization processing"""
        if not self.log_id:
            self.log_id = f"{self.asset_id}_{self.stage.value}_{uuid.uuid4().hex[:8]}"
    
    def complete(self, message: str = "", metadata: Dict[str, Any] = None):
        """Mark log entry as completed"""
        self.completed_at = time.time()
        self.duration = self.completed_at - self.started_at
        self.status = "completed"
        if message:
            self.message = message
        if metadata:
            self.metadata.update(metadata)
    
    def fail(self, error_message: str, error_details: str = None):
        """Mark log entry as failed"""
        self.completed_at = time.time()
        self.duration = self.completed_at - self.started_at
        self.status = "failed"
        self.message = error_message
        self.error_details = error_details
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "log_id": self.log_id,
            "asset_id": self.asset_id,
            "stage": self.stage.value,
            "status": self.status,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration": self.duration,
            "message": self.message,
            "error_details": self.error_details,
            "metadata": self.metadata
        }
