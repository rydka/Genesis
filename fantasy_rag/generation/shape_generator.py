import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

HY3D_PATH = Path(__file__).parent.parent.parent / "third_party" / "Hunyuan3D-2.1" / "hy3dshape"

try:
    if str(HY3D_PATH) not in sys.path:
        sys.path.insert(0, str(HY3D_PATH))
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
    HY3D_AVAILABLE = True
except ImportError:
    HY3D_AVAILABLE = False
    logger.warning("Hunyuan3D not installed — 3D generation unavailable")

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

from .background import BackgroundRemover


@dataclass
class ShapeConfig:
    model_id: str = "tencent/Hunyuan3D-2.1"
    device: str = "cuda"
    num_inference_steps: int = 50
    guidance_scale: float = 5.5
    octree_resolution: int = 384
    seed: Optional[int] = None
    remove_background: bool = True
    bg_model: str = "birefnet-general"
    refine_mask: bool = True
    remove_floaters: bool = True
    # Components smaller than this fraction of the total face count are dropped
    floater_ratio: float = 0.005


@dataclass
class ShapeResult:
    success: bool
    generation_time: float
    mesh: object = None
    output_path: Optional[Path] = None
    vertices_count: int = 0
    faces_count: int = 0
    error_message: Optional[str] = None


class ShapeGenerator:

    def __init__(self, config: Optional[ShapeConfig] = None):
        self.config = config or ShapeConfig()
        self.pipeline = None
        self._loaded = False
        self.bg_remover = BackgroundRemover()

    def _load_pipeline(self) -> bool:
        if self._loaded:
            return True
        if not HY3D_AVAILABLE:
            logger.error("Hunyuan3D not available — install from third_party/Hunyuan3D-2.1")
            return False

        start = time.time()
        try:
            self.pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                self.config.model_id
            )
            self._loaded = True
            elapsed = time.time() - start
            logger.info("Hunyuan3D pipeline loaded in %.1fs", elapsed)
            return True
        except Exception:
            logger.exception("Failed to load Hunyuan3D pipeline")
            return False

    def _remove_background(self, image_path: Path) -> Path:
        if not self.bg_remover.is_available or not PIL_AVAILABLE:
            logger.warning("rembg or PIL not available, skipping background removal")
            return image_path

        self.bg_remover.model = self.config.bg_model
        self.bg_remover.refine = self.config.refine_mask
        img_no_bg = self.bg_remover(Image.open(image_path))
        # onnxruntime keeps its GPU arena while the session lives: release it before Hunyuan3D runs,
        # otherwise VRAM saturates and generation slows down several-fold
        self.bg_remover.unload()

        out_path = image_path.parent / f"{image_path.stem}_nobg.png"
        img_no_bg.save(out_path, "PNG")
        logger.info("Background removed (%s, refine=%s): %s",
                    self.config.bg_model, self.config.refine_mask, out_path)
        return out_path

    def _remove_floaters(self, mesh):
        if not TRIMESH_AVAILABLE or not isinstance(mesh, trimesh.Trimesh):
            return mesh
        parts = mesh.split(only_watertight=False)
        if len(parts) <= 1:
            return mesh
        min_faces = len(mesh.faces) * self.config.floater_ratio
        kept = [p for p in parts if len(p.faces) >= min_faces]
        if not kept:
            return mesh
        logger.info("Removed %d floating parts (kept %d)", len(parts) - len(kept), len(kept))
        return trimesh.util.concatenate(kept)

    def generate(self, image_path: Path) -> ShapeResult:
        image_path = Path(image_path)
        if not image_path.exists():
            return ShapeResult(
                success=False, generation_time=0.0,
                error_message=f"Image not found: {image_path}",
            )

        # Before loading Hunyuan3D so both models never share VRAM on a cold start
        if self.config.remove_background:
            image_path = self._remove_background(image_path)

        if not self._load_pipeline():
            return ShapeResult(
                success=False, generation_time=0.0,
                error_message="Hunyuan3D pipeline not available",
            )

        generator = None
        if self.config.seed is not None and TORCH_AVAILABLE:
            generator = torch.Generator().manual_seed(int(self.config.seed))

        start = time.time()
        try:
            mesh = self.pipeline(
                image=str(image_path),
                num_inference_steps=self.config.num_inference_steps,
                guidance_scale=self.config.guidance_scale,
                octree_resolution=self.config.octree_resolution,
                generator=generator,
            )[0]

            if self.config.remove_floaters:
                mesh = self._remove_floaters(mesh)

            elapsed = time.time() - start

            vertices_count = 0
            faces_count = 0
            if hasattr(mesh, "vertices"):
                vertices_count = len(mesh.vertices)
            if hasattr(mesh, "faces"):
                faces_count = len(mesh.faces)

            logger.info("Generated mesh: %d vertices, %d faces in %.1fs",
                        vertices_count, faces_count, elapsed)

            return ShapeResult(
                success=True,
                generation_time=elapsed,
                mesh=mesh,
                vertices_count=vertices_count,
                faces_count=faces_count,
            )

        except Exception as e:
            elapsed = time.time() - start
            logger.exception("Generation failed")
            return ShapeResult(
                success=False, generation_time=elapsed,
                error_message=str(e),
            )

    def export_mesh(self, mesh, output_path: Path, fmt: str = "stl") -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not str(output_path).endswith(f".{fmt}"):
            output_path = output_path.with_suffix(f".{fmt}")

        mesh.export(str(output_path))
        logger.info("Exported mesh to %s", output_path)
        return output_path

    def unload(self):
        self.pipeline = None
        self._loaded = False
        self.bg_remover.unload()
        if TORCH_AVAILABLE:
            torch.cuda.empty_cache()
        logger.info("Hunyuan3D pipeline unloaded, VRAM freed")

    @property
    def is_available(self) -> bool:
        return HY3D_AVAILABLE

    @property
    def is_loaded(self) -> bool:
        return self._loaded
