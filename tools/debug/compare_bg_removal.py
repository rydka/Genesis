"""Compare background-removal models and mask refinement before Hunyuan3D generation.

Usage:
    python tools/debug/compare_bg_removal.py data/7a277384da63c083defb731ceccbff74.jpg
    python tools/debug/compare_bg_removal.py IMAGE --models isnet-anime birefnet-general
    python tools/debug/compare_bg_removal.py IMAGE --generate u2net:raw isnet-anime:refined

Outputs go to data/bg_comparison/<image stem>/.
"""
import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "third_party" / "Hunyuan3D-2.1" / "hy3dshape"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
from fantasy_rag.generation.background import BackgroundRemover, refine_mask  # noqa: E402

logger = logging.getLogger("compare_bg")

DEFAULT_MODELS = ["u2net", "isnet-anime", "birefnet-general"]
OUTPUT_ROOT = PROJECT_ROOT / "data" / "bg_comparison"
TILE = 384


def remove_background(image: Image.Image, model: str) -> np.ndarray:
    return np.asarray(BackgroundRemover(model, refine=False)(image))


def hunyuan_view(rgba: np.ndarray) -> np.ndarray:
    """What Hunyuan3D actually sees: recentered on its alpha bbox, composited on white, 512px."""
    try:
        from hy3dshape.preprocessors import ImageProcessorV2
        image, _ = ImageProcessorV2(size=512).load_image(Image.fromarray(rgba), to_tensor=False)
        return image
    except ImportError:
        logger.warning("Hunyuan3D preprocessor not importable, showing plain composite")
        a = rgba[..., 3:4] / 255.0
        return (rgba[..., :3] * a + 255 * (1 - a)).astype(np.uint8)


def mask_stats(rgba: np.ndarray) -> dict:
    a = rgba[..., 3]
    ys, xs = np.nonzero(a)
    bbox_area = (np.ptp(ys) + 1) * (np.ptp(xs) + 1) / a.size if len(ys) else 0.0
    return {
        "nonzero_pct": (a > 0).mean() * 100,
        "semi_pct": ((a > 0) & (a < 255)).mean() * 100,
        "opaque_pct": (a == 255).mean() * 100,
        "bbox_pct": bbox_area * 100,
    }


def on_checker(rgba: np.ndarray, cell: int = 16) -> np.ndarray:
    h, w = rgba.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    checker = np.where(((yy // cell + xx // cell) % 2)[..., None] == 0, 200, 150).astype(np.float32)
    a = rgba[..., 3:4] / 255.0
    return (rgba[..., :3] * a + checker * (1 - a)).astype(np.uint8)


def fit_tile(img: np.ndarray) -> Image.Image:
    im = Image.fromarray(img).convert("RGB")
    im.thumbnail((TILE, TILE))
    tile = Image.new("RGB", (TILE, TILE), (60, 60, 60))
    tile.paste(im, ((TILE - im.width) // 2, (TILE - im.height) // 2))
    return tile


def contact_sheet(rows: list, out_path: Path):
    cols = ["detourage brut (damier)", "brut -> vu par Hunyuan",
            "affine (damier)", "affine -> vu par Hunyuan"]
    label_w, header_h = 170, 28
    sheet = Image.new("RGB", (label_w + TILE * len(cols), header_h + TILE * len(rows)), (30, 30, 30))
    draw = ImageDraw.Draw(sheet)
    for c, name in enumerate(cols):
        draw.text((label_w + c * TILE + 8, 8), name, fill=(230, 230, 230))
    for r, (model, tiles, stats_raw, stats_ref, secs) in enumerate(rows):
        y = header_h + r * TILE
        draw.text((8, y + 8), model, fill=(255, 210, 120))
        draw.text((8, y + 30), f"{secs:.1f}s", fill=(200, 200, 200))
        draw.text((8, y + 60), "brut:", fill=(200, 200, 200))
        draw.text((8, y + 76), f" alpha>0 {stats_raw['nonzero_pct']:.0f}%", fill=(200, 200, 200))
        draw.text((8, y + 92), f" semi   {stats_raw['semi_pct']:.0f}%", fill=(200, 200, 200))
        draw.text((8, y + 108), f" bbox   {stats_raw['bbox_pct']:.0f}%", fill=(200, 200, 200))
        draw.text((8, y + 136), "affine:", fill=(200, 200, 200))
        draw.text((8, y + 152), f" alpha>0 {stats_ref['nonzero_pct']:.0f}%", fill=(200, 200, 200))
        draw.text((8, y + 168), f" bbox   {stats_ref['bbox_pct']:.0f}%", fill=(200, 200, 200))
        for c, tile in enumerate(tiles):
            sheet.paste(tile, (label_w + c * TILE, y))
    sheet.save(out_path)
    logger.info("Contact sheet: %s", out_path)


def run_comparison(image_path: Path, models: list, out_dir: Path):
    source = Image.open(image_path).convert("RGB")
    source_rgb = np.asarray(source)
    rows = []
    for model in models:
        logger.info("Model %s ...", model)
        start = time.time()
        try:
            raw = remove_background(source, model)
        except Exception:
            logger.exception("Model %s failed", model)
            continue
        secs = time.time() - start
        refined = refine_mask(raw, source_rgb)

        Image.fromarray(raw).save(out_dir / f"{model}_raw.png")
        Image.fromarray(refined).save(out_dir / f"{model}_refined.png")

        s_raw, s_ref = mask_stats(raw), mask_stats(refined)
        logger.info("  %s raw: %s", model, {k: round(v, 1) for k, v in s_raw.items()})
        logger.info("  %s refined: %s", model, {k: round(v, 1) for k, v in s_ref.items()})

        tiles = [fit_tile(on_checker(raw)), fit_tile(hunyuan_view(raw)),
                 fit_tile(on_checker(refined)), fit_tile(hunyuan_view(refined))]
        rows.append((model, tiles, s_raw, s_ref, secs))

    if rows:
        contact_sheet(rows, out_dir / "comparison.png")


def run_generation(out_dir: Path, variants: list, steps: int):
    from fantasy_rag.generation.shape_generator import ShapeGenerator, ShapeConfig

    gen = ShapeGenerator(ShapeConfig(num_inference_steps=steps, remove_background=False))
    for variant in variants:
        model, kind = variant.split(":")
        png = out_dir / f"{model}_{kind}.png"
        if not png.exists():
            logger.error("Missing %s — run the comparison first", png)
            continue
        logger.info("Generating mesh from %s ...", png.name)
        result = gen.generate(png)
        if not result.success:
            logger.error("Generation failed for %s: %s", variant, result.error_message)
            continue
        out = gen.export_mesh(result.mesh, out_dir / f"{model}_{kind}.glb", fmt="glb")
        logger.info("  %s: %d verts, %d faces, %.0fs -> %s", variant, result.vertices_count,
                    result.faces_count, result.generation_time, out)
    gen.unload()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", type=Path)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--generate", nargs="+", metavar="MODEL:raw|refined",
                        help="Run Hunyuan3D on these variants (skips the comparison step)")
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()

    out_dir = OUTPUT_ROOT / args.image.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.generate:
        run_generation(out_dir, args.generate, args.steps)
    else:
        run_comparison(args.image, args.models, out_dir)


if __name__ == "__main__":
    main()
