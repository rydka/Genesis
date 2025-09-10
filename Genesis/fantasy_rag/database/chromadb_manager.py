#!/usr/bin/env python3
"""
ChromaDB Manager for Fantasy Asset Database

This module provides a unified interface for managing all fantasy asset data
in ChromaDB with proper collections, schemas, and relationships.
"""

import logging
import json
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import time

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

from .schema_definitions import (
    AssetRecord, RenderRecord, DescriptionRecord, EmbeddingRecord, ProcessingLog,
    AssetStatus, AssetCategory, ProcessingStage
)

logger = logging.getLogger(__name__)

@dataclass
class DatabaseConfig:
    """Configuration for ChromaDB manager"""
    db_path: str = "ressources/data/fantasy_rag_db"
    
    # Collection names
    assets_collection: str = "fantasy_assets"
    renders_collection: str = "asset_renders"
    descriptions_collection: str = "asset_descriptions"
    embeddings_collection: str = "description_embeddings"
    logs_collection: str = "processing_logs"
    
    # Settings
    enable_telemetry: bool = False
    allow_reset: bool = True
    backup_enabled: bool = True

class ChromaDBManager:
    """
    Unified ChromaDB manager for fantasy asset database
    """
    
    def __init__(self, config: DatabaseConfig = None):
        """
        Initialize ChromaDB manager
        
        Args:
            config: Database configuration
        """
        self.config = config or DatabaseConfig()
        self.client = None
        self.collections = {}
        self.initialized = False
        
        logger.info(f"ChromaDBManager initialized:")
        logger.info(f"  Database path: {self.config.db_path}")
    
    def initialize(self) -> bool:
        """Initialize ChromaDB client and collections"""
        
        if not CHROMADB_AVAILABLE:
            logger.error("ChromaDB not available. Install with: pip install chromadb")
            return False
        
        try:
            # Create database directory
            db_path = Path(self.config.db_path)
            db_path.mkdir(parents=True, exist_ok=True)
            
            # Initialize ChromaDB client
            logger.info("Initializing ChromaDB client...")
            self.client = chromadb.PersistentClient(
                path=str(db_path),
                settings=Settings(
                    anonymized_telemetry=self.config.enable_telemetry,
                    allow_reset=self.config.allow_reset
                )
            )
            
            # Initialize collections
            self._initialize_collections()
            
            self.initialized = True
            logger.info("✅ ChromaDB manager initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB manager: {str(e)}")
            return False
    
    def _initialize_collections(self):
        """Initialize all required collections"""
        
        collection_configs = [
            (self.config.assets_collection, "Fantasy asset metadata and file information"),
            (self.config.renders_collection, "Rendered views and images for assets"),
            (self.config.descriptions_collection, "Generated descriptions with embeddings"),
            (self.config.embeddings_collection, "Text embeddings for semantic search"),
            (self.config.logs_collection, "Processing pipeline logs and status")
        ]
        
        for collection_name, description in collection_configs:
            try:
                # Try to get existing collection
                collection = self.client.get_collection(collection_name)
                logger.info(f"✅ Loaded existing collection: {collection_name} ({collection.count()} docs)")
            except:
                # Create new collection
                collection = self.client.create_collection(
                    name=collection_name,
                    metadata={"description": description}
                )
                logger.info(f"✅ Created new collection: {collection_name}")
            
            self.collections[collection_name] = collection
    
    # Asset Management
    def add_asset(self, asset: AssetRecord) -> bool:
        """Add or update an asset record"""
        
        if not self.initialized and not self.initialize():
            return False
        
        try:
            collection = self.collections[self.config.assets_collection]
            
            # Convert asset to storage format
            asset_data = asset.to_dict()
            
            # Use asset_id as document ID
            collection.upsert(
                ids=[asset.asset_id],
                documents=[asset.original_filename],  # Searchable filename
                metadatas=[asset_data]
            )
            
            logger.debug(f"Added/updated asset: {asset.asset_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding asset {asset.asset_id}: {str(e)}")
            return False
    
    def get_asset(self, asset_id: str) -> Optional[AssetRecord]:
        """Get an asset record by ID"""
        
        if not self.initialized:
            return None
        
        try:
            collection = self.collections[self.config.assets_collection]
            
            result = collection.get(
                ids=[asset_id],
                include=["metadatas"]
            )
            
            if result['metadatas'] and result['metadatas'][0]:
                metadata = result['metadatas'][0]

                # Parse tags from string back to list
                tags_str = metadata.get('tags', '')
                tags = tags_str.split(',') if tags_str else []

                # Extract custom metadata
                custom_metadata = {}
                for key, value in metadata.items():
                    if key.startswith('custom_'):
                        custom_key = key[7:]  # Remove 'custom_' prefix
                        custom_metadata[custom_key] = value

                # Reconstruct AssetRecord
                return AssetRecord(
                    asset_id=metadata['asset_id'],
                    original_filename=metadata['original_filename'],
                    file_path=Path(metadata['file_path']),
                    file_hash=metadata['file_hash'],
                    category=AssetCategory(metadata['category']),
                    status=AssetStatus(metadata['status']),
                    current_stage=ProcessingStage(metadata['current_stage']),
                    file_size=metadata['file_size'],
                    file_format=metadata['file_format'],
                    created_at=metadata['created_at'],
                    updated_at=metadata['updated_at'],
                    processing_attempts=metadata['processing_attempts'],
                    last_error=metadata.get('last_error', ''),
                    processing_time=metadata['processing_time'],
                    tags=tags,
                    custom_metadata=custom_metadata
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting asset {asset_id}: {str(e)}")
            return None
    
    def get_assets_by_status(self, status: AssetStatus) -> List[AssetRecord]:
        """Get all assets with a specific status"""
        
        if not self.initialized:
            return []
        
        try:
            collection = self.collections[self.config.assets_collection]
            
            result = collection.get(
                where={"status": status.value},
                include=["metadatas"]
            )
            
            assets = []
            if result['metadatas']:
                for metadata in result['metadatas']:
                    # Parse tags from string back to list
                    tags_str = metadata.get('tags', '')
                    tags = tags_str.split(',') if tags_str else []

                    # Extract custom metadata
                    custom_metadata = {}
                    for key, value in metadata.items():
                        if key.startswith('custom_'):
                            custom_key = key[7:]  # Remove 'custom_' prefix
                            custom_metadata[custom_key] = value

                    asset = AssetRecord(
                        asset_id=metadata['asset_id'],
                        original_filename=metadata['original_filename'],
                        file_path=Path(metadata['file_path']),
                        file_hash=metadata['file_hash'],
                        category=AssetCategory(metadata['category']),
                        status=AssetStatus(metadata['status']),
                        current_stage=ProcessingStage(metadata['current_stage']),
                        file_size=metadata['file_size'],
                        file_format=metadata['file_format'],
                        created_at=metadata['created_at'],
                        updated_at=metadata['updated_at'],
                        processing_attempts=metadata['processing_attempts'],
                        last_error=metadata.get('last_error', ''),
                        processing_time=metadata['processing_time'],
                        tags=tags,
                        custom_metadata=custom_metadata
                    )
                    assets.append(asset)
            
            return assets
            
        except Exception as e:
            logger.error(f"Error getting assets by status {status}: {str(e)}")
            return []
    
    # Render Management
    def add_render(self, render: RenderRecord) -> bool:
        """Add a render record"""
        
        if not self.initialized and not self.initialize():
            return False
        
        try:
            collection = self.collections[self.config.renders_collection]
            
            render_data = render.to_dict()
            
            collection.upsert(
                ids=[render.render_id],
                documents=[f"{render.asset_id}_{render.view_name}"],
                metadatas=[render_data]
            )
            
            logger.debug(f"Added render: {render.render_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding render {render.render_id}: {str(e)}")
            return False
    
    def get_asset_renders(self, asset_id: str) -> List[RenderRecord]:
        """Get all renders for an asset"""
        
        if not self.initialized:
            return []
        
        try:
            collection = self.collections[self.config.renders_collection]
            
            result = collection.get(
                where={"asset_id": asset_id},
                include=["metadatas"]
            )
            
            renders = []
            if result['metadatas']:
                for metadata in result['metadatas']:
                    # Parse resolution from string back to tuple
                    resolution_str = metadata.get('resolution', '1024x1024')
                    resolution = tuple(map(int, resolution_str.split('x')))

                    # Parse colors from string back to list
                    colors_str = metadata.get('dominant_colors', '')
                    colors = colors_str.split(',') if colors_str else []

                    render = RenderRecord(
                        render_id=metadata['render_id'],
                        asset_id=metadata['asset_id'],
                        view_name=metadata['view_name'],
                        image_path=Path(metadata['image_path']),
                        resolution=resolution,
                        samples=metadata['samples'],
                        render_engine=metadata['render_engine'],
                        render_time=metadata['render_time'],
                        file_size=metadata['file_size'],
                        created_at=metadata['created_at'],
                        image_quality_score=metadata.get('image_quality_score', 0.0),
                        has_transparency=metadata['has_transparency'],
                        dominant_colors=colors
                    )
                    renders.append(render)
            
            return renders
            
        except Exception as e:
            logger.error(f"Error getting renders for asset {asset_id}: {str(e)}")
            return []
    
    # Description Management
    def add_description(self, description: DescriptionRecord, embedding_vector: List[float] = None) -> bool:
        """Add a description record with optional embedding"""
        
        if not self.initialized and not self.initialize():
            return False
        
        try:
            collection = self.collections[self.config.descriptions_collection]
            
            description_data = description.to_dict()
            
            # Prepare for ChromaDB
            collection.upsert(
                ids=[description.description_id],
                documents=[description.description_text],
                metadatas=[description_data],
                embeddings=[embedding_vector] if embedding_vector else None
            )
            
            logger.debug(f"Added description: {description.description_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding description {description.description_id}: {str(e)}")
            return False
    
    def get_asset_descriptions(self, asset_id: str) -> List[DescriptionRecord]:
        """Get all descriptions for an asset"""
        
        if not self.initialized:
            return []
        
        try:
            collection = self.collections[self.config.descriptions_collection]
            
            result = collection.get(
                where={"asset_id": asset_id},
                include=["metadatas", "documents"]
            )
            
            descriptions = []
            if result['metadatas']:
                for i, metadata in enumerate(result['metadatas']):
                    # Parse views from string back to list
                    views_str = metadata.get('views_analyzed', '')
                    views = views_str.split(',') if views_str else []

                    description = DescriptionRecord(
                        description_id=metadata['description_id'],
                        asset_id=metadata['asset_id'],
                        language=metadata['language'],
                        description_text=result['documents'][i],
                        model_name=metadata['model_name'],
                        generation_time=metadata['generation_time'],
                        word_count=metadata['word_count'],
                        target_length=metadata['target_length'],
                        quality_score=metadata.get('quality_score', 0.0),
                        similarity_to_target=metadata.get('similarity_to_target', 0.0),
                        language_confidence=metadata.get('language_confidence', 0.0),
                        created_at=metadata['created_at'],
                        prompt_used=metadata.get('prompt_used', ''),
                        views_analyzed=views
                    )
                    descriptions.append(description)
            
            return descriptions
            
        except Exception as e:
            logger.error(f"Error getting descriptions for asset {asset_id}: {str(e)}")
            return []
    
    # Statistics and Management
    def get_database_stats(self) -> Dict[str, Any]:
        """Get comprehensive database statistics"""
        
        if not self.initialized:
            return {"initialized": False}
        
        try:
            stats = {
                "initialized": True,
                "database_path": self.config.db_path,
                "collections": {}
            }
            
            for collection_name, collection in self.collections.items():
                count = collection.count()
                stats["collections"][collection_name] = {
                    "document_count": count,
                    "name": collection_name
                }
            
            # Asset status breakdown
            if self.config.assets_collection in self.collections:
                asset_collection = self.collections[self.config.assets_collection]
                
                # Get all assets to analyze status distribution
                all_assets = asset_collection.get(include=["metadatas"])
                
                status_counts = {}
                category_counts = {}
                stage_counts = {}
                
                if all_assets['metadatas']:
                    for metadata in all_assets['metadatas']:
                        status = metadata.get('status', 'unknown')
                        category = metadata.get('category', 'unknown')
                        stage = metadata.get('current_stage', 'unknown')
                        
                        status_counts[status] = status_counts.get(status, 0) + 1
                        category_counts[category] = category_counts.get(category, 0) + 1
                        stage_counts[stage] = stage_counts.get(stage, 0) + 1
                
                stats["asset_analysis"] = {
                    "status_distribution": status_counts,
                    "category_distribution": category_counts,
                    "stage_distribution": stage_counts
                }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting database stats: {str(e)}")
            return {"initialized": True, "error": str(e)}
    
    def clear_collection(self, collection_name: str) -> bool:
        """Clear a specific collection"""
        
        if not self.initialized:
            return False
        
        try:
            if collection_name in self.collections:
                # Delete and recreate collection
                self.client.delete_collection(collection_name)
                
                # Recreate with same metadata
                collection = self.client.create_collection(
                    name=collection_name,
                    metadata={"description": f"Recreated {collection_name} collection"}
                )
                
                self.collections[collection_name] = collection
                logger.info(f"✅ Cleared collection: {collection_name}")
                return True
            else:
                logger.warning(f"Collection not found: {collection_name}")
                return False
                
        except Exception as e:
            logger.error(f"Error clearing collection {collection_name}: {str(e)}")
            return False
