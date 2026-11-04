import logging
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

STDGEN_DIR = Path(__file__).parent.parent.parent / "third_party" / "StdGEN"
CONDA_ENV = "stdgen"
CONDA_EXE = Path.home() / "anaconda3" / "Scripts" / "conda.exe"


@dataclass
class CharacterConfig:
    num_views: int = 6
    num_levels: int = 3
    low_vram: bool = True
    outside_ratio: float = 0.20
    decompose: bool = True


@dataclass
class CharacterResult:
    success: bool
    generation_time: float
    output_dir: Optional[Path] = None
    glb_paths: list = None
    error_message: Optional[str] = None

    def __post_init__(self):
        if self.glb_paths is None:
            self.glb_paths = []


class CharacterGenerator:

    def __init__(self, config: Optional[CharacterConfig] = None):
        self.config = config or CharacterConfig()

    @property
    def is_available(self) -> bool:
        return STDGEN_DIR.exists() and CONDA_EXE.exists()

    def _run_stage(self, script: str, args: list, cwd: Path = None) -> subprocess.CompletedProcess:
        cmd = [
            str(CONDA_EXE), "run", "-n", CONDA_ENV, "--no-banner",
            "python", script, *args,
        ]
        logger.info("Running: %s", " ".join(cmd))
        return subprocess.run(
            cmd,
            cwd=str(cwd or STDGEN_DIR),
            capture_output=True,
            text=True,
            timeout=1200,
        )

    def generate(self, image_path: Path, output_dir: Path) -> CharacterResult:
        if not self.is_available:
            return CharacterResult(
                success=False, generation_time=0.0,
                error_message="StdGEN not installed or conda env missing",
            )

        image_path = Path(image_path)
        output_dir = Path(output_dir)

        if not image_path.exists():
            return CharacterResult(
                success=False, generation_time=0.0,
                error_message=f"Image not found: {image_path}",
            )

        input_dir = output_dir / "input"
        input_dir.mkdir(parents=True, exist_ok=True)

        import shutil
        dest = input_dir / f"character{image_path.suffix}"
        shutil.copy2(image_path, dest)

        start = time.time()

        # Stage 1: Canonicalize pose to A-pose
        apose_dir = output_dir / "apose"
        r = self._run_stage("infer_canonicalize.py", [
            "--input_dir", str(input_dir),
            "--output_dir", str(apose_dir),
        ])
        if r.returncode != 0:
            logger.error("Canonicalize failed: %s", r.stderr[-500:] if r.stderr else "")
            return CharacterResult(
                success=False, generation_time=time.time() - start,
                error_message=f"Stage 1 (canonicalize) failed: {r.stderr[-200:] if r.stderr else 'unknown'}",
            )
        logger.info("Stage 1 (canonicalize) done")

        # Stage 2: Multi-view generation
        mv_dir = output_dir / "multiview"
        mv_args = [
            "--input_dir", str(apose_dir),
            "--output_dir", str(mv_dir),
            "--num_views", str(self.config.num_views),
            "--num_levels", str(self.config.num_levels),
        ]
        if self.config.low_vram:
            mv_args.append("--low_vram")

        r = self._run_stage("infer_multiview.py", mv_args)
        if r.returncode != 0:
            logger.error("Multiview failed: %s", r.stderr[-500:] if r.stderr else "")
            return CharacterResult(
                success=False, generation_time=time.time() - start,
                error_message=f"Stage 2 (multiview) failed: {r.stderr[-200:] if r.stderr else 'unknown'}",
            )
        logger.info("Stage 2 (multiview) done")

        # Stage 3: 3D Reconstruction (S-LRM)
        slrm_dir = output_dir / "slrm"
        r = self._run_stage("infer_slrm.py", [
            "--input_dir", str(mv_dir),
            "--output_dir", str(slrm_dir),
        ])
        if r.returncode != 0:
            logger.error("SLRM failed: %s", r.stderr[-500:] if r.stderr else "")
            return CharacterResult(
                success=False, generation_time=time.time() - start,
                error_message=f"Stage 3 (reconstruction) failed: {r.stderr[-200:] if r.stderr else 'unknown'}",
            )
        logger.info("Stage 3 (reconstruction) done")

        # Stage 4: Mesh refinement
        refined_dir = output_dir / "refined"
        refine_args = [
            "--input_mv_dir", str(mv_dir),
            "--input_obj_dir", str(slrm_dir),
            "--output_dir", str(refined_dir),
            "--outside_ratio", str(self.config.outside_ratio),
        ]
        if not self.config.decompose:
            refine_args.append("--no_decompose")

        r = self._run_stage("infer_refine.py", refine_args)
        if r.returncode != 0:
            logger.error("Refine failed: %s", r.stderr[-500:] if r.stderr else "")
            return CharacterResult(
                success=False, generation_time=time.time() - start,
                error_message=f"Stage 4 (refinement) failed: {r.stderr[-200:] if r.stderr else 'unknown'}",
            )
        logger.info("Stage 4 (refinement) done")

        elapsed = time.time() - start
        glb_files = sorted(refined_dir.rglob("*.glb"))

        logger.info("Character generated in %.1fs — %d GLB files", elapsed, len(glb_files))

        return CharacterResult(
            success=True,
            generation_time=elapsed,
            output_dir=refined_dir,
            glb_paths=glb_files,
        )
