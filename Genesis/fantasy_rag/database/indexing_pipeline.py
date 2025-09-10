#!/usr/bin/env python3
"""
Indexing Pipeline for Fantasy Assets

This module provides a complete pipeline for processing fantasy assets
from STL files through rendering, description generation, and indexing.
"""

import logging
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import time

from .chromadb_manager import ChromaDBManager, DatabaseConfig
from .schema_definitions import (
    AssetRecord, RenderRecord, DescriptionRecord, EmbeddingRecord, ProcessingLog,
    AssetStatus, AssetCategory, ProcessingStage
)

from ..rendering.blender_renderer import BlenderRenderer
from ..description.llm_interface import OllamaVisionInterface
from ..description.prompt_templates import PromptContext, DescriptionLanguage
from ..embeddings.text_embedder import TextEmbedder, EmbeddingConfig

logger = logging.getLogger(__name__)

@dataclass
class PipelineConfig:
    """Configuration for the indexing pipeline"""
    
    # Database
    database_config: DatabaseConfig = field(default_factory=DatabaseConfig)
    
    # Rendering
    enable_rendering: bool = True
    render_resolution: Tuple[int, int] = (1024, 1024)
    render_samples: int = 128
    
    # Description generation
    enable_description: bool = True
    description_languages: List[DescriptionLanguage] = field(
        default_factory=lambda: [DescriptionLanguage.FRENCH, DescriptionLanguage.ENGLISH]
    )
    target_description_length: int = 200
    ollama_model: str = "llama3.2-vision:latest"
    
    # Embeddings
    enable_embeddings: bool = True
    embedding_config: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    
    # Processing
    max_retries: int = 3
    skip_existing: bool = True
    parallel_processing: bool = False

@dataclass
class ProcessingResult:
    """Result of processing a single asset"""
    
    asset_id: str
    success: bool
    stages_completed: List[ProcessingStage]
    total_time: float
    
    # Stage-specific results
    renders_created: int = 0
    descriptions_created: int = 0
    embeddings_created: int = 0
    
    # Error information
    errors: List[str] = field(default_factory=list)
    last_stage: Optional[ProcessingStage] = None

class IndexingPipeline:
    """
    Complete indexing pipeline for fantasy assets
    """
    
    def __init__(self, config: PipelineConfig = None):
        """
        Initialize indexing pipeline
        
        Args:
            config: Pipeline configuration
        """
        self.config = config or PipelineConfig()
        
        # Initialize components
        self.db_manager = ChromaDBManager(self.config.database_config)
        self.renderer = None
        self.vision_interface = None
        self.text_embedder = None
        
        self.initialized = False
        
        logger.info("IndexingPipeline initialized")
    
    def initialize(self) -> bool:
        """Initialize all pipeline components"""
        
        logger.info("Initializing indexing pipeline components...")
        
        try:
            # Initialize database
            if not self.db_manager.initialize():
                logger.error("Failed to initialize database manager")
                return False
            
            # Initialize renderer
            if self.config.enable_rendering:
                self.renderer = BlenderRenderer()
                logger.info("✅ Renderer initialized")
            
            # Initialize vision interface
            if self.config.enable_description:
                self.vision_interface = OllamaVisionInterface(
                    model_name=self.config.ollama_model
                )
                if not self.vision_interface.verify_connection():
                    logger.error("Failed to connect to Ollama vision interface")
                    return False
                logger.info("✅ Vision interface initialized")
            
            # Initialize text embedder
            if self.config.enable_embeddings:
                self.text_embedder = TextEmbedder(self.config.embedding_config)
                logger.info("✅ Text embedder initialized")
            
            self.initialized = True
            logger.info("🎉 All pipeline components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {str(e)}")
            return False
    
    def process_stl_file(self, stl_path: Path) -> ProcessingResult:
        """
        Process a single STL file through the complete pipeline
        
        Args:
            stl_path: Path to STL file
            
        Returns:
            ProcessingResult with processing details
        """
        if not self.initialized and not self.initialize():
            return ProcessingResult(
                asset_id="",
                success=False,
                stages_completed=[],
                total_time=0.0,
                errors=["Pipeline not initialized"]
            )
        
        start_time = time.time()
        asset_id = self._generate_asset_id(stl_path)
        
        logger.info(f"Processing STL file: {stl_path.name}")
        logger.info(f"Asset ID: {asset_id}")
        
        result = ProcessingResult(
            asset_id=asset_id,
            success=False,
            stages_completed=[],
            total_time=0.0
        )
        
        try:
            # Stage 1: Asset Discovery
            asset_record = self._discover_asset(stl_path, asset_id)
            if not asset_record:
                result.errors.append("Failed to discover asset")
                return result
            
            result.stages_completed.append(ProcessingStage.DISCOVERY)
            result.last_stage = ProcessingStage.DISCOVERY
            
            # Check if we should skip existing
            if self.config.skip_existing:
                existing_asset = self.db_manager.get_asset(asset_id)
                if existing_asset and existing_asset.status == AssetStatus.INDEXED:
                    logger.info(f"Skipping already indexed asset: {asset_id}")
                    result.success = True
                    result.total_time = time.time() - start_time
                    return result
            
            # Stage 2: Rendering
            renders = []
            if self.config.enable_rendering:
                renders = self._render_asset(asset_record)
                if renders:
                    result.stages_completed.append(ProcessingStage.RENDERING)
                    result.last_stage = ProcessingStage.RENDERING
                    result.renders_created = len(renders)
                else:
                    result.errors.append("Failed to render asset")
                    return result
            
            # Stage 3: Description Generation
            descriptions = []
            if self.config.enable_description and renders:
                descriptions = self._generate_descriptions(asset_record, renders)
                if descriptions:
                    result.stages_completed.append(ProcessingStage.DESCRIPTION)
                    result.last_stage = ProcessingStage.DESCRIPTION
                    result.descriptions_created = len(descriptions)
                else:
                    result.errors.append("Failed to generate descriptions")
                    return result
            
            # Stage 4: Embedding Generation
            embeddings = []
            if self.config.enable_embeddings and descriptions:
                embeddings = self._generate_embeddings(descriptions)
                if embeddings:
                    result.stages_completed.append(ProcessingStage.EMBEDDING)
                    result.last_stage = ProcessingStage.EMBEDDING
                    result.embeddings_created = len(embeddings)
                else:
                    result.errors.append("Failed to generate embeddings")
                    return result
            
            # Stage 5: Final Indexing
            if self._finalize_indexing(asset_record, descriptions, embeddings):
                result.stages_completed.append(ProcessingStage.INDEXING)
                result.last_stage = ProcessingStage.INDEXING
                result.success = True
            else:
                result.errors.append("Failed to finalize indexing")
                return result
            
            result.total_time = time.time() - start_time
            logger.info(f"✅ Successfully processed asset {asset_id} in {result.total_time:.1f}s")
            
            return result
            
        except Exception as e:
            result.errors.append(f"Pipeline error: {str(e)}")
            result.total_time = time.time() - start_time
            logger.error(f"Error processing asset {asset_id}: {str(e)}")
            return result
    
    def _generate_asset_id(self, stl_path: Path) -> str:
        """Generate unique asset ID from file path and content"""
        
        # Use file path and size for ID generation
        path_str = str(stl_path.absolute())
        file_size = stl_path.stat().st_size if stl_path.exists() else 0
        
        # Create hash
        hash_input = f"{path_str}_{file_size}".encode('utf-8')
        file_hash = hashlib.md5(hash_input).hexdigest()
        
        return f"stl_{file_hash[:12]}"
    
    def _discover_asset(self, stl_path: Path, asset_id: str) -> Optional[AssetRecord]:
        """Discover and register an asset"""
        
        try:
            if not stl_path.exists():
                logger.error(f"STL file not found: {stl_path}")
                return None
            
            # Calculate file hash
            with open(stl_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            
            # Create asset record
            asset = AssetRecord(
                asset_id=asset_id,
                original_filename=stl_path.name,
                file_path=stl_path,
                file_hash=file_hash,
                file_size=stl_path.stat().st_size,
                file_format="stl",
                status=AssetStatus.DISCOVERED,
                current_stage=ProcessingStage.DISCOVERY
            )
            
            # Add to database
            if self.db_manager.add_asset(asset):
                logger.info(f"✅ Asset discovered: {asset_id}")
                return asset
            else:
                logger.error(f"Failed to add asset to database: {asset_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error discovering asset {asset_id}: {str(e)}")
            return None
    
    def _render_asset(self, asset: AssetRecord) -> List[RenderRecord]:
        """Render views for an asset"""
        
        logger.info(f"Rendering asset: {asset.asset_id}")
        
        try:
            # Update asset status
            asset.update_status(AssetStatus.RENDERING, ProcessingStage.RENDERING)
            self.db_manager.add_asset(asset)
            
            # Render views
            render_result = self.renderer.render_asset_views(
                stl_path=asset.file_path,
                asset_id=asset.asset_id,
                asset_name=asset.original_filename
            )
            
            if not render_result.success:
                logger.error(f"Rendering failed: {render_result.error_message}")
                asset.add_error(f"Rendering failed: {render_result.error_message}")
                self.db_manager.add_asset(asset)
                return []

            # Create render records
            renders = []
            for view_name, image_path in render_result.views.items():
                render = RenderRecord(
                    render_id="",  # Will be auto-generated
                    asset_id=asset.asset_id,
                    view_name=view_name,
                    image_path=image_path,
                    resolution=self.config.render_resolution,
                    samples=self.config.render_samples,
                    render_engine="cycles",
                    render_time=render_result.render_time / len(render_result.views),
                    file_size=image_path.stat().st_size if image_path.exists() else 0
                )
                
                # Add to database
                if self.db_manager.add_render(render):
                    renders.append(render)
            
            # Update asset status
            asset.update_status(AssetStatus.RENDERED, ProcessingStage.RENDERING)
            self.db_manager.add_asset(asset)
            
            logger.info(f"✅ Created {len(renders)} renders for asset {asset.asset_id}")
            return renders
            
        except Exception as e:
            logger.error(f"Error rendering asset {asset.asset_id}: {str(e)}")
            asset.add_error(f"Rendering error: {str(e)}")
            self.db_manager.add_asset(asset)
            return []
    
    def _generate_descriptions(self, asset: AssetRecord, renders: List[RenderRecord]) -> List[DescriptionRecord]:
        """Generate descriptions for an asset"""
        
        logger.info(f"Generating descriptions for asset: {asset.asset_id}")
        
        try:
            # Update asset status
            asset.update_status(AssetStatus.DESCRIBING, ProcessingStage.DESCRIPTION)
            self.db_manager.add_asset(asset)
            
            # Prepare view paths
            view_paths = {}
            for render in renders:
                view_paths[render.view_name] = render.image_path
            
            descriptions = []
            
            # Generate descriptions for each language
            for language in self.config.description_languages:
                context = PromptContext(
                    asset_name=asset.original_filename,
                    view_count=len(view_paths),
                    target_length=self.config.target_description_length,
                    language=language
                )
                
                # Generate description
                response = self.vision_interface.generate_description(view_paths, context)
                
                if response.success:
                    description = DescriptionRecord(
                        description_id="",  # Will be auto-generated
                        asset_id=asset.asset_id,
                        language=language.value,
                        description_text=response.description,
                        model_name=self.config.ollama_model,
                        generation_time=response.processing_time,
                        word_count=response.word_count,
                        target_length=self.config.target_description_length,
                        views_analyzed=list(view_paths.keys())
                    )
                    
                    # Add to database
                    if self.db_manager.add_description(description):
                        descriptions.append(description)
                        logger.info(f"✅ Generated {language.value} description for {asset.asset_id}")
                else:
                    logger.error(f"Failed to generate {language.value} description: {response.error_message}")
            
            # Update asset status
            if descriptions:
                asset.update_status(AssetStatus.DESCRIBED, ProcessingStage.DESCRIPTION)
            else:
                asset.add_error("No descriptions generated")
            
            self.db_manager.add_asset(asset)
            
            return descriptions
            
        except Exception as e:
            logger.error(f"Error generating descriptions for {asset.asset_id}: {str(e)}")
            asset.add_error(f"Description error: {str(e)}")
            self.db_manager.add_asset(asset)
            return []
    
    def _generate_embeddings(self, descriptions: List[DescriptionRecord]) -> List[EmbeddingRecord]:
        """Generate embeddings for descriptions"""
        
        logger.info(f"Generating embeddings for {len(descriptions)} descriptions")
        
        try:
            embeddings = []
            
            for description in descriptions:
                # Generate embedding
                embedding_result = self.text_embedder.embed_texts([description.description_text])
                
                if embedding_result.embeddings is not None:
                    embedding = EmbeddingRecord(
                        embedding_id="",  # Will be auto-generated
                        description_id=description.description_id,
                        asset_id=description.asset_id,
                        embedding_vector=embedding_result.embeddings[0].tolist(),
                        embedding_model=embedding_result.model_name,
                        embedding_dimension=embedding_result.embedding_dim,
                        generation_time=embedding_result.processing_time,
                        language=description.language,
                        text_length=len(description.description_text)
                    )
                    
                    embeddings.append(embedding)
                    logger.debug(f"✅ Generated embedding for description {description.description_id}")
            
            logger.info(f"✅ Generated {len(embeddings)} embeddings")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            return []
    
    def _finalize_indexing(self, asset: AssetRecord, descriptions: List[DescriptionRecord], embeddings: List[EmbeddingRecord]) -> bool:
        """Finalize indexing by updating asset status"""
        
        try:
            # Update asset to indexed status
            asset.update_status(AssetStatus.INDEXED, ProcessingStage.COMPLETE)
            
            # Update processing time
            asset.processing_time = time.time() - asset.created_at
            
            # Add final asset update
            if self.db_manager.add_asset(asset):
                logger.info(f"✅ Asset {asset.asset_id} successfully indexed")
                return True
            else:
                logger.error(f"Failed to finalize asset {asset.asset_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error finalizing indexing for {asset.asset_id}: {str(e)}")
            return False
    
    def process_directory(self, directory: Path) -> List[ProcessingResult]:
        """Process all STL files in a directory"""
        
        logger.info(f"Processing directory: {directory}")
        
        if not directory.exists():
            logger.error(f"Directory not found: {directory}")
            return []
        
        # Find all STL files
        stl_files = list(directory.rglob("*.stl"))
        logger.info(f"Found {len(stl_files)} STL files")
        
        results = []
        
        for stl_file in stl_files:
            logger.info(f"Processing {stl_file.name}...")
            result = self.process_stl_file(stl_file)
            results.append(result)
            
            if result.success:
                logger.info(f"✅ {stl_file.name} processed successfully")
            else:
                logger.error(f"❌ {stl_file.name} failed: {result.errors}")
        
        # Summary
        successful = sum(1 for r in results if r.success)
        logger.info(f"📊 Processing complete: {successful}/{len(results)} successful")
        
        return results
