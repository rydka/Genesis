import logging
import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from .chromadb_manager import ChromaDBManager, DatabaseConfig
from .schema_definitions import (
    AssetRecord, RenderRecord, AssetStatus, ProcessingStage
)
from ..rendering.blender_renderer import BlenderRenderer
from ..embeddings.clip_embedder import CLIPConfig
from ..embeddings.similarity_search import SimilaritySearchEngine

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    database_config: DatabaseConfig = field(default_factory=DatabaseConfig)

    enable_rendering: bool = True
    render_resolution: Tuple[int, int] = (1024, 1024)
    render_samples: int = 128

    enable_embeddings: bool = True
    clip_config: CLIPConfig = field(default_factory=CLIPConfig)

    max_retries: int = 3
    skip_existing: bool = True


@dataclass
class ProcessingResult:
    asset_id: str
    success: bool
    stages_completed: List[ProcessingStage]
    total_time: float

    renders_created: int = 0
    embeddings_created: int = 0

    errors: List[str] = field(default_factory=list)
    last_stage: Optional[ProcessingStage] = None


class IndexingPipeline:

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.db_manager = ChromaDBManager(self.config.database_config)
        self.renderer: Optional[BlenderRenderer] = None
        self.search_engine: Optional[SimilaritySearchEngine] = None
        self.initialized = False

    def initialize(self) -> bool:
        try:
            if not self.db_manager.initialize():
                logger.error("Failed to initialize database manager")
                return False

            if self.config.enable_rendering:
                self.renderer = BlenderRenderer()
                logger.info("Renderer initialized")

            if self.config.enable_embeddings:
                self.search_engine = SimilaritySearchEngine(
                    db_path=str(self.config.database_config.persist_directory),
                    clip_config=self.config.clip_config,
                )
                if not self.search_engine.initialize():
                    logger.error("Failed to initialize search engine")
                    return False
                logger.info("CLIP search engine initialized")

            self.initialized = True
            logger.info("Pipeline initialized")
            return True

        except Exception:
            logger.exception("Failed to initialize pipeline")
            return False

    def process_stl_file(self, stl_path: Path) -> ProcessingResult:
        if not self.initialized and not self.initialize():
            return ProcessingResult(
                asset_id="", success=False, stages_completed=[],
                total_time=0.0, errors=["Pipeline not initialized"],
            )

        start_time = time.time()
        asset_id = self._generate_asset_id(stl_path)

        result = ProcessingResult(
            asset_id=asset_id, success=False,
            stages_completed=[], total_time=0.0,
        )

        try:
            # Stage 1: Discovery
            asset_record = self._discover_asset(stl_path, asset_id)
            if not asset_record:
                result.errors.append("Failed to discover asset")
                return result

            result.stages_completed.append(ProcessingStage.DISCOVERY)
            result.last_stage = ProcessingStage.DISCOVERY

            if self.config.skip_existing:
                existing = self.db_manager.get_asset(asset_id)
                if existing and existing.status == AssetStatus.INDEXED:
                    logger.info("Skipping already indexed asset: %s", asset_id)
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

            # Stage 3: CLIP Embedding + Indexing
            if self.config.enable_embeddings and renders:
                view_paths = {r.view_name: Path(r.image_path) for r in renders}
                metadata = {
                    "original_filename": asset_record.original_filename,
                    "file_size": asset_record.file_size,
                }

                if self.search_engine.index_asset(asset_id, view_paths, metadata):
                    result.stages_completed.append(ProcessingStage.EMBEDDING)
                    result.last_stage = ProcessingStage.EMBEDDING
                    result.embeddings_created = 1
                else:
                    result.errors.append("Failed to generate CLIP embedding")
                    return result

            # Finalize
            if self._finalize_indexing(asset_record):
                result.stages_completed.append(ProcessingStage.INDEXING)
                result.last_stage = ProcessingStage.INDEXING
                result.success = True

            result.total_time = time.time() - start_time
            logger.info("Processed asset %s in %.1fs", asset_id, result.total_time)
            return result

        except Exception as e:
            result.errors.append(f"Pipeline error: {e}")
            result.total_time = time.time() - start_time
            logger.exception("Error processing asset %s", asset_id)
            return result

    def _generate_asset_id(self, stl_path: Path) -> str:
        path_str = str(stl_path.absolute())
        file_size = stl_path.stat().st_size if stl_path.exists() else 0
        hash_input = f"{path_str}_{file_size}".encode("utf-8")
        return f"stl_{hashlib.md5(hash_input).hexdigest()[:12]}"

    def _discover_asset(self, stl_path: Path, asset_id: str) -> Optional[AssetRecord]:
        try:
            if not stl_path.exists():
                logger.error("STL file not found: %s", stl_path)
                return None

            with open(stl_path, "rb") as f:
                file_hash = hashlib.md5(f.read()).hexdigest()

            asset = AssetRecord(
                asset_id=asset_id,
                original_filename=stl_path.name,
                file_path=stl_path,
                file_hash=file_hash,
                file_size=stl_path.stat().st_size,
                file_format="stl",
                status=AssetStatus.DISCOVERED,
                current_stage=ProcessingStage.DISCOVERY,
            )

            if self.db_manager.add_asset(asset):
                return asset
            return None

        except Exception:
            logger.exception("Error discovering asset %s", asset_id)
            return None

    def _render_asset(self, asset: AssetRecord) -> List[RenderRecord]:
        try:
            asset.update_status(AssetStatus.RENDERING, ProcessingStage.RENDERING)
            self.db_manager.add_asset(asset)

            render_result = self.renderer.render_asset_views(
                stl_path=asset.file_path,
                asset_id=asset.asset_id,
                asset_name=asset.original_filename,
            )

            if not render_result.success:
                logger.error("Rendering failed: %s", render_result.error_message)
                return []

            renders = []
            for view_name, image_path in render_result.views.items():
                render = RenderRecord(
                    render_id="",
                    asset_id=asset.asset_id,
                    view_name=view_name,
                    image_path=image_path,
                    resolution=self.config.render_resolution,
                    samples=self.config.render_samples,
                    render_engine="cycles",
                    render_time=render_result.render_time / max(len(render_result.views), 1),
                    file_size=image_path.stat().st_size if Path(image_path).exists() else 0,
                )
                if self.db_manager.add_render(render):
                    renders.append(render)

            asset.update_status(AssetStatus.RENDERED, ProcessingStage.RENDERING)
            self.db_manager.add_asset(asset)

            logger.info("Created %d renders for %s", len(renders), asset.asset_id)
            return renders

        except Exception:
            logger.exception("Error rendering asset %s", asset.asset_id)
            return []

    def _finalize_indexing(self, asset: AssetRecord) -> bool:
        try:
            asset.update_status(AssetStatus.INDEXED, ProcessingStage.COMPLETE)
            asset.processing_time = time.time() - asset.created_at
            return self.db_manager.add_asset(asset)
        except Exception:
            logger.exception("Error finalizing %s", asset.asset_id)
            return False

    def process_directory(self, directory: Path) -> List[ProcessingResult]:
        if not directory.exists():
            logger.error("Directory not found: %s", directory)
            return []

        stl_files = list(directory.rglob("*.stl"))
        logger.info("Found %d STL files in %s", len(stl_files), directory)

        results = []
        for stl_file in stl_files:
            result = self.process_stl_file(stl_file)
            results.append(result)

        successful = sum(1 for r in results if r.success)
        logger.info("Processing complete: %d/%d successful", successful, len(results))
        return results
