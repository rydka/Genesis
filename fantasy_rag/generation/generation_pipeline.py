import hashlib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from .shape_generator import ShapeGenerator, ShapeConfig, ShapeResult
from ..config.settings import GENERATED_DIR

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    asset_id: str
    success: bool
    generation_time: float
    mesh_path: Optional[Path] = None
    vertices_count: int = 0
    faces_count: int = 0
    stages_completed: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class GenerationPipeline:

    def __init__(self, output_dir: Optional[Path] = None,
                 shape_config: Optional[ShapeConfig] = None):
        self.output_dir = Path(output_dir or GENERATED_DIR)
        self.shape_generator = ShapeGenerator(shape_config)

    def generate_from_image(self, image_path: Path,
                            asset_name: Optional[str] = None,
                            export_format: str = "glb") -> GenerationResult:
        image_path = Path(image_path)
        asset_id = self._generate_asset_id(image_path)
        asset_name = asset_name or image_path.stem

        result = GenerationResult(
            asset_id=asset_id, success=False, generation_time=0.0,
        )
        start = time.time()

        if not image_path.exists():
            result.errors.append(f"Image not found: {image_path}")
            return result

        # Generate mesh
        shape_result = self.shape_generator.generate(image_path)
        result.generation_time = shape_result.generation_time

        if not shape_result.success:
            result.errors.append(shape_result.error_message or "Generation failed")
            return result

        result.stages_completed.append("shape_generation")
        result.vertices_count = shape_result.vertices_count
        result.faces_count = shape_result.faces_count

        # Export
        asset_dir = self.output_dir / asset_id
        asset_dir.mkdir(parents=True, exist_ok=True)
        output_file = asset_dir / f"{asset_name}.{export_format}"

        try:
            exported = self.shape_generator.export_mesh(
                shape_result.mesh, output_file, fmt=export_format,
            )
            result.mesh_path = exported
            result.stages_completed.append("export")
        except Exception as e:
            result.errors.append(f"Export failed: {e}")
            return result

        result.success = True
        result.generation_time = time.time() - start
        logger.info("Generated %s: %d verts, %d faces in %.1fs -> %s",
                     asset_id, result.vertices_count, result.faces_count,
                     result.generation_time, result.mesh_path)
        return result

    def unload(self):
        self.shape_generator.unload()

    @staticmethod
    def _generate_asset_id(image_path: Path) -> str:
        path_str = str(image_path.absolute())
        file_size = image_path.stat().st_size if image_path.exists() else 0
        hash_input = f"{path_str}_{file_size}".encode("utf-8")
        return f"gen_{hashlib.md5(hash_input).hexdigest()[:12]}"
