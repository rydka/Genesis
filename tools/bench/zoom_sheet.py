"""Zoom sheets: the same regions (face, hands...) of several versions of a mesh, side by side.

Silhouette measures do not see a dented face or a spike on a thumb; these sheets are the visual check
of the shape-fitting tools (fit_silhouette, fit_flexicubes). Every version is rendered by Blender
(tools/debug/render_glb_views.py) at high resolution, all framed on the bounding box of the first
version, so that a crop at fixed image coordinates shows the same region of every mesh.

Usage (from the repository root; outputs go under data/, never in git):
    .venv\\Scripts\\python.exe tools/bench/zoom_sheet.py --out data/bench/<campaign>/<stem>/zooms
        --row input=A.glb --row old=B.glb --row new=C.glb
        --crop face:front:0.30,0.05,0.55,0.30 --crop hand:three_quarter:0.2,0.5,0.4,0.7
        [--res 2048]

A crop is name:view:x0,y0,x1,y1 in fractions of the render. Writes <out>/<name>.png (one strip per
crop, versions left to right) and <out>/zooms.png (all crops, one row per version). Renders already
present in <out>/renders/<label>/ are reused (delete them, or --force, after a mesh changed). Blender
renders on the GPU: the GPU lock of the main repository is taken while rendering (tools/bench/gpu_lock.py).
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.bench.gpu_lock import GpuLock  # noqa: E402

BLENDER = Path(r"C:\Program Files\Blender Foundation\Blender 4.5\blender.exe")

Crop = Tuple[str, str, Tuple[float, float, float, float]]


def parse_crop(text: str) -> Crop:
    name, view, box = text.split(":")
    x0, y0, x1, y1 = (float(v) for v in box.split(","))
    return name, view, (x0, y0, x1, y1)


def render_rows(rows: Sequence[Tuple[str, Path]], views: Sequence[str], render_dir: Path, res: int,
                force: bool = False):
    frame = rows[0][1]
    for label, glb in rows:
        d = render_dir / label
        if not force and all((d / f"{glb.stem}__{v}.png").exists() for v in views):
            continue
        cmd = [str(BLENDER), "-b", "-P", str(ROOT / "tools/debug/render_glb_views.py"), "--", str(d),
               str(glb), "--res", str(res), "--views", ",".join(views), "--frame", str(frame)]
        print("blender:", label, glb, flush=True)
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)


def crop_tile(path: Path, box, tile: int) -> Image.Image:
    im = Image.open(path).convert("RGB")
    w, h = im.size
    c = im.crop((int(box[0] * w), int(box[1] * h), int(box[2] * w), int(box[3] * h)))
    c.thumbnail((tile, tile), Image.LANCZOS)
    return c


def make_sheets(rows: Sequence[Tuple[str, Path]], crops: Sequence[Crop], render_dir: Path, out: Path,
                tile: int = 420) -> List[Path]:
    written = []
    bg, fg = (30, 30, 30), (255, 210, 120)
    tiles = {(label, name): crop_tile(render_dir / label / f"{glb.stem}__{view}.png", box, tile)
             for label, glb in rows for name, view, box in crops}
    # one strip per crop: versions left to right
    for name, view, _ in crops:
        sheet = Image.new("RGB", (tile * len(rows), tile + 24), bg)
        d = ImageDraw.Draw(sheet)
        for i, (label, _) in enumerate(rows):
            t = tiles[(label, name)]
            d.text((i * tile + 6, 4), f"{label} ({name}, {view})", fill=fg)
            sheet.paste(t, (i * tile + (tile - t.width) // 2, 24 + (tile - t.height) // 2))
        path = out / f"{name}.png"
        sheet.save(path)
        written.append(path)
    # all crops: one row per version
    lw = 150
    sheet = Image.new("RGB", (lw + tile * len(crops), 24 + tile * len(rows)), bg)
    d = ImageDraw.Draw(sheet)
    for j, (name, view, _) in enumerate(crops):
        d.text((lw + j * tile + 6, 4), f"{name} ({view})", fill=fg)
    for i, (label, _) in enumerate(rows):
        d.text((6, 24 + i * tile + tile // 2), label, fill=fg)
        for j, (name, _, _) in enumerate(crops):
            t = tiles[(label, name)]
            sheet.paste(t, (lw + j * tile + (tile - t.width) // 2, 24 + i * tile + (tile - t.height) // 2))
    path = out / "zooms.png"
    sheet.save(path)
    written.append(path)
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--row", action="append", required=True, help="label=path.glb (first one frames)")
    parser.add_argument("--crop", action="append", required=True, help="name:view:x0,y0,x1,y1")
    parser.add_argument("--res", type=int, default=2048)
    parser.add_argument("--tile", type=int, default=420)
    parser.add_argument("--force", action="store_true", help="render again even if present")
    args = parser.parse_args()
    rows = [(r.split("=", 1)[0], Path(r.split("=", 1)[1])) for r in args.row]
    for label, glb in rows:
        if not glb.exists():
            sys.exit(f"missing mesh for {label}: {glb}")
    crops = [parse_crop(c) for c in args.crop]
    views = sorted({v for _, v, _ in crops})
    args.out.mkdir(parents=True, exist_ok=True)
    if not BLENDER.exists():
        sys.exit(f"Blender not found at {BLENDER}")
    lock = GpuLock("zoom_sheet", f"Blender renders for {args.out}")
    if not lock.acquire():
        sys.exit(f"GPU lock held ({lock.describe_holder()}); try again later")
    try:
        render_rows(rows, views, args.out / "renders", args.res, args.force)
    finally:
        lock.release()
    for p in make_sheets(rows, crops, args.out / "renders", args.out, args.tile):
        print(p)


if __name__ == "__main__":
    main()
