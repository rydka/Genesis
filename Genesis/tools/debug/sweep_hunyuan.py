"""Sweep Hunyuan3D hyperparameters on one background-removed image.

Usage:
    python tools/debug/sweep_hunyuan.py data/bg_comparison/<stem>/e2e_input_nobg.png \
        --octree 384 512 --seeds 0 1 2

Each run is saved as <out>/oct{octree}_seed{seed}.glb, with a summary.json.
The pipeline is loaded once; the image is used as-is (no background removal).
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fantasy_rag.generation.shape_generator import ShapeGenerator, ShapeConfig  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("sweep")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", type=Path)
    parser.add_argument("--octree", type=int, nargs="+", default=[384, 512])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=5.5)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    out_dir = args.out or args.image.parent / "sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    gen = ShapeGenerator(ShapeConfig(
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        remove_background=False,
    ))

    summary = []
    for octree in args.octree:
        for seed in args.seeds:
            name = f"oct{octree}_seed{seed}"
            gen.config.octree_resolution = octree
            gen.config.seed = seed
            logger.info("Run %s ...", name)
            start = time.time()
            result = gen.generate(args.image)
            entry = {"name": name, "octree": octree, "seed": seed, "success": result.success,
                     "time_s": round(time.time() - start, 1)}
            if result.success:
                path = gen.export_mesh(result.mesh, out_dir / f"{name}.glb", fmt="glb")
                entry.update(vertices=result.vertices_count, faces=result.faces_count,
                             size_mb=round(path.stat().st_size / 1e6, 1))
            else:
                entry["error"] = result.error_message
            logger.info("DONE %s", entry)
            summary.append(entry)
            (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    gen.unload()


if __name__ == "__main__":
    main()
