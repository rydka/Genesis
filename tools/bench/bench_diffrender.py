"""Time and memory of the in-house differentiable rasterizer (fantasy_rag.diffrender).

Usage (from the repository root):
    .venv\\Scripts\\python.exe tools/bench/bench_diffrender.py --device cpu --faces 1000000 --res 512
    .venv\\Scripts\\python.exe tools/bench/bench_diffrender.py --device cuda --faces 1000000 --res 512

A GPU run must respect the lock data/.gpu-lock of the main repository (rule gpu-et-environnement;
see tools/bench/gpu_lock.py, also from a worktree): the script takes it on CUDA, releases it at the
end, and refuses to start while another live process holds it. Nothing else is written to disk.

Scene: a UV sphere with about --faces faces filling most of the image, i.e. a dense mesh with
sub-pixel triangles as produced by the image-to-3D generators. Measured:
  - visibility only (rasterize under no_grad);
  - a full optimisation step: rasterize + interpolate (normals, depth) + silhouette gradients
    (--mode analytic: `antialias`; --mode micro_edge: `edge_gradients`, applied to the mask and to
    the normal map), then backward to the vertex positions.
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from fantasy_rag.diffrender import (  # noqa: E402
    Camera, antialias, edge_gradients, interpolate, look_at, perspective, rasterize, uv_sphere,
    vertex_normals,
)
from tools.bench.gpu_lock import GpuLock  # noqa: E402


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def _timed(fn, device, repeat):
    times = []
    out = None
    for _ in range(repeat):
        _sync(device)
        t0 = time.perf_counter()
        out = fn()
        _sync(device)
        times.append(time.perf_counter() - t0)
    return min(times), out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--faces", type=int, default=1_000_000)
    parser.add_argument("--res", type=int, default=512)
    parser.add_argument("--max-fragments", type=int, default=1 << 22)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--mode", choices=["analytic", "micro_edge"], default="analytic")
    parser.add_argument("--gpu-lock", type=Path, default=None,
                        help="lock file (default: <main repository>/data/.gpu-lock)")
    parser.add_argument("--take-stale-lock", action="store_true",
                        help="take over a lock whose PID no longer runs (check it first)")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type != "cuda":
        run(args, device)
        return
    # Always the lock of the main repository, even when run from a worktree. A GPU benchmark that
    # silently ran on the CPU would be mislabelled, so this tool refuses instead of falling back.
    lock = GpuLock("bench_diffrender", f"diffrender benchmark {args.faces} faces {args.res}px",
                   args.gpu_lock, args.take_stale_lock)
    if not lock.acquire():
        sys.exit(f"{lock.path} held ({lock.describe_holder()}): another GPU job is running, "
                 "retry later")
    try:
        run(args, device)
    finally:
        lock.release()


def run(args, device):
    # 2 (rings - 1) segments faces with segments = 2 rings
    rings = max(4, int(round(math.sqrt(args.faces / 4.0))))
    v, f = uv_sphere(rings, 2 * rings, radius=1.0)
    v, f = v.to(device), f.to(device)
    cam = Camera(look_at((0.3, 0.2, 2.6)), perspective(55.0), (args.res, args.res))
    mvp = cam.mvp.to(device)
    print(f"device={device} faces={f.shape[0]} vertices={v.shape[0]} res={args.res} "
          f"max_fragments={args.max_fragments} mode={args.mode} torch={torch.__version__}")

    def to_clip(x):
        ones = torch.ones_like(x[:, :1])
        return torch.cat([x, ones], 1) @ mvp.T

    def visibility():
        with torch.no_grad():
            return rasterize(to_clip(v), f, args.res, max_fragments=args.max_fragments)

    def step():
        verts = v.clone().requires_grad_()
        clip = to_clip(verts)
        rast = rasterize(clip, f, args.res, max_fragments=args.max_fragments)
        normals = interpolate(vertex_normals(verts, f), rast, f)
        mask = rast.mask.to(clip.dtype).unsqueeze(-1)
        if args.mode == "analytic":
            alpha = antialias(mask, rast, clip, f)
        else:
            alpha = edge_gradients(mask, rast, clip, f)
            normals = edge_gradients(normals, rast, clip, f)
        loss = alpha.mean() + rast.depth.mean() + normals.square().mean()
        loss.backward()
        return verts.grad

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t_vis, rast = _timed(visibility, device, args.repeat)
    covered = int(rast.mask.sum())
    mem_vis = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t_step, grad = _timed(step, device, args.repeat)
    mem_step = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else None

    print(f"covered pixels: {covered} ({covered / args.res ** 2:.0%}), "
          f"vertices with gradient: {int((grad.abs().sum(1) > 0).sum())}")
    print(f"visibility (rasterize, no grad): {t_vis * 1000:.0f} ms"
          + (f", peak {mem_vis:.2f} GiB" if mem_vis is not None else ""))
    print(f"optimisation step (forward + backward): {t_step * 1000:.0f} ms"
          + (f", peak {mem_step:.2f} GiB" if mem_step is not None else ""))
    if device.type == "cpu":
        try:
            import psutil
            rss = psutil.Process(os.getpid()).memory_info().peak_wset / 2**30
            print(f"process peak working set: {rss:.2f} GiB")
        except (ImportError, AttributeError):
            pass


if __name__ == "__main__":
    main()
