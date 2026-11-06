"""Fit an existing mesh to its source image by the silhouette (in-house differentiable rasterizer).

First application of `fantasy_rag.diffrender` (phase 1 of docs/conception/optimisation-par-zones.md):
  1. camera only: orthographic projection (w = 1), scale, 2D translation and a small rotation on top
     of a discrete front-view orientation chosen among 6 candidates by overlap;
  2. mesh deformation, camera frozen: vertex displacements restricted to the image plane (the depth
     of every vertex is unchanged), smoothed parametrisation, Laplacian and anchor regularisation;
  3. measures before / after (IoU global, head band, edge band, flipped faces, displacements),
     comparison plate, loss curve, summary.json, fitted GLB; optional Blender renders.

Both optimisations go coarse to fine (resolution pyramid) and combine a long-range loss (rendered
coverage times the signed distance to the target boundary, whose minimum is the target itself) with
an L2 silhouette loss whose gradients come from `edge_gradients` (micro-edges, the recommended mode).
Silhouette gradients only exist within about one pixel of the edges, hence the pyramid and the
signed-distance term.

Usage (from the repository root; outputs go under data/, never in git):
    .venv\\Scripts\\python.exe tools/bench/fit_silhouette.py --mesh M.glb --mask MASK.png
        [--image SOURCE.jpg] --out data/bench/<date>_fit-silhouette/<stem> [--device cpu] [--blender]

The mask is the alpha channel of an RGBA PNG (or a greyscale image) with the same framing as the
source image. The script is resumable: the camera is cached in camera.json, and a finished run
(summary.json present) is skipped unless --force.
"""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fantasy_rag.diffrender import (  # noqa: E402
    edge_gradients, face_normals, rasterize, vertex_normals,
)

try:
    from scipy import ndimage
    SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover - scipy is in config/requirements/generation.txt
    SCIPY_AVAILABLE = False

BLENDER = Path(r"C:\Program Files\Blender Foundation\Blender 4.5\blender.exe")

# Candidate front views: rows are the camera axes (right, up, towards the camera) in mesh coordinates.
# "+z" is the glTF front of Hunyuan3D meshes (Y up, the figure faces +Z).
ORIENTATIONS: Dict[str, Tuple[Tuple[float, ...], ...]] = {
    "+z": ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    "-z": ((-1, 0, 0), (0, 1, 0), (0, 0, -1)),
    "+x": ((0, 0, -1), (0, 1, 0), (1, 0, 0)),
    "-x": ((0, 0, 1), (0, 1, 0), (-1, 0, 0)),
    "+y": ((1, 0, 0), (0, 0, -1), (0, 1, 0)),
    "-y": ((1, 0, 0), (0, 0, 1), (0, -1, 0)),
}


# --------------------------------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------------------------------

def load_mesh(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Vertices [V, 3] float32 and faces [F, 3] int64 (scene transforms applied, no re-processing)."""
    import trimesh
    mesh = trimesh.load(str(path), force="mesh", process=False)
    return np.asarray(mesh.vertices, np.float32), np.asarray(mesh.faces, np.int64)


def load_mask(path: Path) -> np.ndarray:
    """Target coverage [H, W] in [0, 1]: alpha channel if present, else luminance."""
    im = Image.open(path)
    if im.mode in ("RGBA", "LA") or "transparency" in im.info:
        a = np.asarray(im.convert("RGBA"))[..., 3]
    else:
        a = np.asarray(im.convert("L"))
    return a.astype(np.float32) / 255.0


def signed_distance(binary: np.ndarray) -> np.ndarray:
    """Signed distance in pixels to the boundary of `binary`: > 0 outside, < 0 inside."""
    if not SCIPY_AVAILABLE:
        raise RuntimeError("scipy is required for the signed distance (pip install scipy)")
    b = binary.astype(bool)
    return (ndimage.distance_transform_edt(~b) - ndimage.distance_transform_edt(b)).astype(np.float32)


def level_size(full_hw: Tuple[int, int], long_side: int) -> Tuple[int, int]:
    """Resolution of a pyramid level with the aspect ratio of the target."""
    h, w = full_hw
    k = min(1.0, long_side / max(h, w))
    return max(8, int(round(h * k))), max(8, int(round(w * k)))


def resample(img: np.ndarray, hw: Tuple[int, int]) -> np.ndarray:
    """Area-like resampling of a float image to (h, w)."""
    if img.shape == tuple(hw):
        return img.astype(np.float32)
    pil = Image.fromarray(img.astype(np.float32), mode="F")
    return np.asarray(pil.resize((hw[1], hw[0]), Image.BOX), np.float32)


# --------------------------------------------------------------------------------------------------
# Camera
# --------------------------------------------------------------------------------------------------

def rodrigues(r: torch.Tensor) -> torch.Tensor:
    """Rotation matrix from an axis-angle vector (differentiable, stable at 0)."""
    theta2 = (r * r).sum()
    theta = torch.sqrt(theta2 + 1e-12)
    k = r / theta
    zero = torch.zeros((), dtype=r.dtype, device=r.device)
    kx = torch.stack([torch.stack([zero, -k[2], k[1]]),
                      torch.stack([k[2], zero, -k[0]]),
                      torch.stack([-k[1], k[0], zero])])
    eye = torch.eye(3, dtype=r.dtype, device=r.device)
    return eye + torch.sin(theta) * kx + (1 - torch.cos(theta)) * (kx @ kx)


@dataclass
class OrthoCamera:
    """Orthographic camera in units of the target height H.

    A point p of the mesh goes to camera axes q = R p with R = rodrigues(rotation) @ R0(orientation),
    then to the image: u = cx + scale * q.x, v = cy - scale * q.y (u, v in units of H, origin at the
    top-left corner, v down). Depth is -q.z (closer = smaller NDC z).
    """

    orientation: str = "+z"
    scale: float = 1.0
    cx: float = 0.5
    cy: float = 0.5
    rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["rotation_deg"] = [math.degrees(a) for a in self.rotation]
        return d

    @staticmethod
    def from_dict(d: dict) -> "OrthoCamera":
        return OrthoCamera(d["orientation"], d["scale"], d["cx"], d["cy"], tuple(d["rotation"]))


class CameraParams(torch.nn.Module):
    """Differentiable version of OrthoCamera (log scale, translation, axis-angle rotation)."""

    def __init__(self, cam: OrthoCamera):
        super().__init__()
        self.orientation = cam.orientation
        self.register_buffer("r0", torch.tensor(ORIENTATIONS[cam.orientation], dtype=torch.float32))
        self.log_scale = torch.nn.Parameter(torch.tensor(math.log(cam.scale)))
        self.center = torch.nn.Parameter(torch.tensor([cam.cx, cam.cy]))
        self.rotation = torch.nn.Parameter(torch.tensor(list(cam.rotation), dtype=torch.float32))

    def rot(self) -> torch.Tensor:
        return rodrigues(self.rotation) @ self.r0

    def export(self) -> OrthoCamera:
        c = self.center.detach().cpu().tolist()
        return OrthoCamera(self.orientation, float(self.log_scale.exp()), c[0], c[1],
                           tuple(float(a) for a in self.rotation.detach().cpu()))


def to_clip(q: torch.Tensor, scale, center, aspect_wh: float, zscale: float) -> torch.Tensor:
    """Camera-axis points q [V, 3] -> clip [V, 4] with w = 1. aspect_wh = W / H of the target."""
    u = center[0] + scale * q[:, 0]
    v = center[1] - scale * q[:, 1]
    x = 2.0 * u / aspect_wh - 1.0
    y = 1.0 - 2.0 * v
    z = -q[:, 2] * zscale
    return torch.stack([x, y, z, torch.ones_like(x)], dim=-1)


def render_alpha(clip: torch.Tensor, faces: torch.Tensor, hw: Tuple[int, int],
                 supersample: int = 1, grad: bool = True) -> torch.Tensor:
    """Silhouette [h, w]; micro-edge gradients; `supersample` x `supersample` samples per pixel."""
    h, w = hw
    s = supersample
    rast = rasterize(clip, faces, (h * s, w * s))
    mask = rast.mask.to(clip.dtype).unsqueeze(-1)
    if grad:
        mask = edge_gradients(mask, rast, clip, faces)
    if s > 1:
        mask = mask.view(1, h, s, w, s, 1).mean(dim=(2, 4))
    return mask[0, ..., 0]


@dataclass
class Target:
    """Target coverage and signed distance at one pyramid level."""

    alpha: torch.Tensor   # [h, w] in [0, 1]
    sdf: torch.Tensor     # [h, w] signed distance in units of the full target height H

    @staticmethod
    def make(alpha_full: np.ndarray, sdf_full_px: np.ndarray, hw, device) -> "Target":
        H = alpha_full.shape[0]
        a = resample(alpha_full, hw)
        d = resample(sdf_full_px, hw) / H
        return Target(torch.tensor(a, device=device), torch.tensor(d, device=device))


def silhouette_losses(alpha: torch.Tensor, target: Target) -> Tuple[torch.Tensor, torch.Tensor]:
    """(long-range, L2) silhouette losses, both normalised per pixel of the level.

    long-range: mean(alpha * sdf) - its minimum over all images is reached at alpha = target, and the
    gradient on a boundary pixel is proportional to its distance to the target boundary.
    """
    return (alpha * target.sdf).mean(), ((alpha - target.alpha) ** 2).mean()


def init_camera(verts: torch.Tensor, orientation: str, alpha_full: np.ndarray) -> OrthoCamera:
    """Bounding-box alignment of the projected mesh with the target (orthographic: exact bbox)."""
    H, _ = alpha_full.shape
    ys, xs = np.nonzero(alpha_full > 0.5)
    tx0, tx1, ty0, ty1 = xs.min() / H, (xs.max() + 1) / H, ys.min() / H, (ys.max() + 1) / H
    q = verts @ torch.tensor(ORIENTATIONS[orientation], dtype=verts.dtype, device=verts.device).T
    lo, hi = q.min(0).values.cpu().numpy(), q.max(0).values.cpu().numpy()
    scale = math.sqrt((tx1 - tx0) / (hi[0] - lo[0]) * (ty1 - ty0) / (hi[1] - lo[1]))
    mx, my = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2
    return OrthoCamera(orientation, float(scale), float((tx0 + tx1) / 2 - scale * mx),
                       float((ty0 + ty1) / 2 + scale * my))


def iou(a: np.ndarray, b: np.ndarray, region: Optional[np.ndarray] = None) -> float:
    a, b = a.astype(bool), b.astype(bool)
    if region is not None:
        a, b = a & region, b & region
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else 1.0


@dataclass
class FitConfig:
    device: str = "cpu"
    # camera phase: (long side, supersample, iterations) per level
    camera_levels: Sequence[Tuple[int, int, int]] = ((128, 1, 60), (256, 1, 60), (512, 1, 40))
    camera_probe_iters: int = 30        # short fit of each orientation at the first level
    camera_lr: float = 0.01
    camera_w_sdf: float = 1.0
    camera_w_l2: float = 1.0
    camera_w_rot: float = 0.01          # keeps the rotation small (radians^2)
    # deformation phase
    deform_levels: Sequence[Tuple[int, int, int]] = ((256, 2, 80), (512, 2, 120))
    deform_lr_px: float = 0.5           # largest step, in pixels of the current level
    smooth_iters: int = 30              # displacement = S^k u (neighbour averaging)
    # "free": 2 in-plane components per vertex; "normal": one scalar per vertex along the image-plane
    # part of the (smoothed) vertex normal, so that faces seen from the front barely move and the
    # contour moves across itself rather than sliding along
    direction: str = "free"
    w_sdf: float = 0.5
    w_l2: float = 1.0
    # mean ||L d||^2 with d in units of the target height; tuned on meshes of 0.2 to 0.6 M vertices
    # (it grows roughly with the square of the vertex count for the same smoothness)
    w_lap: float = 2e5
    w_anchor: float = 2.0               # ||d||^2
    head_fraction: float = 0.2          # top band of the target bbox used as "head" zone
    edge_band_px: float = 4.0           # band around the target boundary for the edge IoU
    log_every: int = 10


# --------------------------------------------------------------------------------------------------
# Phase 1: camera
# --------------------------------------------------------------------------------------------------

def _zscale(verts: torch.Tensor) -> float:
    return 0.9 / float(verts.norm(dim=1).max())


def fit_camera_levels(verts, faces, alpha_full, sdf_full, cam: OrthoCamera, levels, cfg: FitConfig,
                      history: Optional[list] = None, tag: str = "camera") -> OrthoCamera:
    device = verts.device
    H, W = alpha_full.shape
    params = CameraParams(cam).to(device)
    opt = torch.optim.Adam(params.parameters(), lr=cfg.camera_lr)
    zs = _zscale(verts)
    for long_side, ss, iters in levels:
        hw = level_size((H, W), long_side)
        tgt = Target.make(alpha_full, sdf_full, hw, device)
        for g in opt.param_groups:
            g["lr"] = cfg.camera_lr * 128.0 / long_side
        for it in range(iters):
            opt.zero_grad()
            q = verts @ params.rot().T
            clip = to_clip(q, params.log_scale.exp(), params.center, W / H, zs)
            alpha = render_alpha(clip, faces, hw, ss)
            l_sdf, l_l2 = silhouette_losses(alpha, tgt)
            loss = (cfg.camera_w_sdf * l_sdf + cfg.camera_w_l2 * l_l2
                    + cfg.camera_w_rot * (params.rotation ** 2).sum())
            loss.backward()
            opt.step()
            if history is not None:
                history.append({"phase": tag, "level": long_side, "iter": it, "loss": float(loss),
                                "sdf": float(l_sdf), "l2": float(l_l2)})
    return params.export()


def hard_silhouette(verts, faces, cam: OrthoCamera, hw, full_hw) -> np.ndarray:
    """Binary silhouette [h, w] rendered at one sample per pixel (no gradients)."""
    with torch.no_grad():
        p = CameraParams(cam).to(verts.device)
        q = verts @ p.rot().T
        clip = to_clip(q, p.log_scale.exp(), p.center, full_hw[1] / full_hw[0], _zscale(verts))
        return render_alpha(clip, faces, hw, 1, grad=False).cpu().numpy() > 0.5


def fit_camera(verts, faces, alpha_full, sdf_full, cfg: FitConfig, history: Optional[list] = None,
               orientations: Optional[Sequence[str]] = None, log=print) -> Tuple[OrthoCamera, dict]:
    """Choose the front orientation by overlap after a short fit, then refine through the pyramid."""
    H, W = alpha_full.shape
    first = cfg.camera_levels[0]
    hw0 = level_size((H, W), first[0])
    tgt0 = resample(alpha_full, hw0) > 0.5
    scores = {}
    best = None
    for name in orientations or ORIENTATIONS:
        cam0 = init_camera(verts, name, alpha_full)
        cam = fit_camera_levels(verts, faces, alpha_full, sdf_full, cam0,
                                [(first[0], first[1], cfg.camera_probe_iters)], cfg)
        score = iou(hard_silhouette(verts, faces, cam, hw0, (H, W)), tgt0)
        scores[name] = {"iou_bbox_init": iou(hard_silhouette(verts, faces, cam0, hw0, (H, W)), tgt0),
                        "iou_probe": score}
        log(f"  orientation {name}: IoU bbox {scores[name]['iou_bbox_init']:.3f}, "
            f"after {cfg.camera_probe_iters} it {score:.3f}")
        if best is None or score > best[0]:
            best = (score, name, cam)
    _, name, cam = best
    cam = fit_camera_levels(verts, faces, alpha_full, sdf_full, cam, cfg.camera_levels, cfg, history)
    return cam, {"orientation_scores": scores, "orientation": name}


# --------------------------------------------------------------------------------------------------
# Phase 2: deformation
# --------------------------------------------------------------------------------------------------

def adjacency(faces: torch.Tensor, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Row-normalised neighbour averaging A [V, V] (sparse CSR) and the degree [V]."""
    e = torch.cat([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    e = torch.cat([e, e.flip(1)])
    e = torch.unique(e, dim=0)
    deg = torch.bincount(e[:, 0], minlength=n).clamp(min=1).to(torch.float32)
    vals = 1.0 / deg[e[:, 0]]
    a = torch.sparse_coo_tensor(e.T, vals, (n, n)).coalesce().to_sparse_csr()
    return a, deg


class UniformAdam(torch.optim.Optimizer):
    """Adam whose denominator is the largest second moment over the whole tensor (as in Large
    Steps, Nicolet et al. 2021). Keeps the relative size of the per-vertex gradients, so vertices
    with only a weak regulariser gradient barely move, unlike per-coordinate Adam."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-12):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            b1, b2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if not st:
                    st["t"] = 0
                    st["m"] = torch.zeros_like(p)
                    st["v"] = torch.zeros_like(p)
                st["t"] += 1
                st["m"].mul_(b1).add_(p.grad, alpha=1 - b1)
                st["v"].mul_(b2).addcmul_(p.grad, p.grad, value=1 - b2)
                m_hat = st["m"] / (1 - b1 ** st["t"])
                v_max = (st["v"].max() / (1 - b2 ** st["t"])).sqrt()
                p.add_(m_hat / (v_max + group["eps"]), alpha=-group["lr"])


def deform(verts, faces, alpha_full, sdf_full, cam: OrthoCamera, cfg: FitConfig,
           history: Optional[list] = None, log=print, time_budget_s: Optional[float] = None
           ) -> torch.Tensor:
    """In-plane vertex displacements against the silhouette, camera frozen. Returns new vertices."""
    device = verts.device
    H, W = alpha_full.shape
    n = verts.shape[0]
    with torch.no_grad():
        p = CameraParams(cam).to(device)
        rot = p.rot()
        scale = p.log_scale.exp()
        center = p.center.clone()
    zs = _zscale(verts)
    q0 = verts @ rot.T                                   # camera axes; q0[:, 2] stays fixed
    avg, _ = adjacency(faces, n)
    if cfg.direction == "normal":
        with torch.no_grad():
            nrm = vertex_normals(q0, faces)
            for _ in range(4):
                nrm = 0.5 * (nrm + avg @ nrm)
            basis = nrm[:, :2].clone()                   # length <= 1: 0 for front-facing vertices
        u = torch.zeros(n, 1, device=device, requires_grad=True)
    elif cfg.direction == "free":
        basis = None
        u = torch.zeros(n, 2, device=device, requires_grad=True)
    else:
        raise ValueError(f"unknown direction {cfg.direction!r}")
    opt = UniformAdam([u])
    t0 = time.perf_counter()

    def displacement():
        d = u
        for _ in range(cfg.smooth_iters):
            d = 0.5 * (d + avg @ d)
        return d * basis if basis is not None else d

    for long_side, ss, iters in cfg.deform_levels:
        hw = level_size((H, W), long_side)
        tgt = Target.make(alpha_full, sdf_full, hw, device)
        for g in opt.param_groups:
            g["lr"] = cfg.deform_lr_px / (float(scale) * hw[0])
        for it in range(iters):
            opt.zero_grad()
            d = displacement()                           # [V, 2], mesh units
            q = torch.cat([q0[:, :2] + d, q0[:, 2:]], dim=1)
            clip = to_clip(q, scale, center, W / H, zs)
            alpha = render_alpha(clip, faces, hw, ss)
            l_sdf, l_l2 = silhouette_losses(alpha, tgt)
            d_h = d * scale                              # displacement in units of the height H
            lap = d_h - avg @ d_h
            l_lap = (lap ** 2).sum(1).mean()
            l_anchor = (d_h ** 2).sum(1).mean()
            loss = cfg.w_sdf * l_sdf + cfg.w_l2 * l_l2 + cfg.w_lap * l_lap + cfg.w_anchor * l_anchor
            loss.backward()
            opt.step()
            if history is not None:
                history.append({"phase": "deform", "level": long_side, "iter": it,
                                "loss": float(loss), "sdf": float(l_sdf), "l2": float(l_l2),
                                "lap": float(l_lap), "anchor": float(l_anchor)})
            if it % cfg.log_every == 0 or it == iters - 1:
                log(f"  deform {long_side}x{ss} it {it:3d}: loss {float(loss):.5f} "
                    f"l2 {float(l_l2):.5f} sdf {float(l_sdf):.5f} lap {float(l_lap):.2e} "
                    f"max|d| {float(d_h.norm(dim=1).max()) * H:.1f} px "
                    f"({time.perf_counter() - t0:.0f} s)")
            if time_budget_s and time.perf_counter() - t0 > time_budget_s:
                log(f"  time budget of {time_budget_s:.0f} s reached, stopping")
                break
        else:
            continue
        break
    with torch.no_grad():
        d = displacement()
        q = torch.cat([q0[:, :2] + d, q0[:, 2:]], dim=1)
        return q @ rot                                   # back to mesh coordinates (rot orthonormal)


# --------------------------------------------------------------------------------------------------
# Measures and outputs
# --------------------------------------------------------------------------------------------------

def zone_masks(target: np.ndarray, head_fraction: float, band_px: float) -> Dict[str, np.ndarray]:
    ys, xs = np.nonzero(target)
    y0, y1 = ys.min(), ys.max() + 1
    head = np.zeros_like(target, dtype=bool)
    head[y0:y0 + max(1, int(round((y1 - y0) * head_fraction))), :] = True
    sdf = signed_distance(target)
    edge = np.abs(sdf) <= band_px
    return {"head": head, "edge": edge}


def measure(sil: np.ndarray, target: np.ndarray, zones: Dict[str, np.ndarray]) -> dict:
    t = target.astype(bool)
    return {
        "iou": iou(sil, t),
        "iou_head": iou(sil, t, zones["head"]),
        "iou_edge": iou(sil, t, zones["edge"]),
        "excess_px": int((sil & ~t).sum()),
        "missing_px": int((~sil & t).sum()),
    }


def error_map(sil: np.ndarray, target: np.ndarray) -> np.ndarray:
    """RGB: grey = both, red = rendered but not in the target, blue = target not covered."""
    t = target.astype(bool)
    out = np.full(sil.shape + (3,), 255, np.uint8)
    out[sil & t] = (170, 170, 170)
    out[sil & ~t] = (220, 40, 40)
    out[~sil & t] = (40, 90, 230)
    return out


def _panel(img: Image.Image, label: str, height: int) -> Image.Image:
    img = img.convert("RGB")
    w = max(1, int(round(img.width * height / img.height)))
    img = img.resize((w, height), Image.LANCZOS)
    out = Image.new("RGB", (w, height + 22), "white")
    out.paste(img, (0, 22))
    ImageDraw.Draw(out).text((4, 4), label, fill="black")
    return out


def loss_curve(history: List[dict], size=(640, 360)) -> Image.Image:
    """Loss curves (normalised per phase) drawn with PIL; one colour per series."""
    w, h = size
    img = Image.new("RGB", size, "white")
    dr = ImageDraw.Draw(img)
    margin = 30
    dr.rectangle([margin, 10, w - 10, h - margin], outline="black")
    n = len(history)
    if n < 2:
        return img
    series = {"total": "black", "l2": "red", "sdf": "blue"}
    for k, (name, color) in enumerate(series.items()):
        vals = np.array([r.get("loss" if name == "total" else name, np.nan) for r in history], float)
        ok = np.isfinite(vals)
        if ok.sum() < 2:
            continue
        lo, hi = np.nanmin(vals), np.nanmax(vals)
        span = hi - lo if hi > lo else 1.0
        pts = [(margin + i * (w - 10 - margin) / (n - 1),
                h - margin - (v - lo) / span * (h - margin - 10)) for i, v in enumerate(vals) if
               np.isfinite(v)]
        dr.line(pts, fill=color, width=1)
        dr.text((margin + 6 + 70 * k, h - margin + 8), name + " (min-max)", fill=color)
    # phase separators
    for i in range(1, n):
        if history[i]["phase"] != history[i - 1]["phase"] or history[i]["level"] != history[i - 1]["level"]:
            x = margin + i * (w - 10 - margin) / (n - 1)
            dr.line([(x, 10), (x, h - margin)], fill=(200, 200, 200))
    return img


def plate(source: Optional[Path], target: np.ndarray, before: np.ndarray, after: np.ndarray,
          history: List[dict], m_before: dict, m_after: dict, path: Path, height: int = 360):
    gray = lambda b: Image.fromarray((np.where(b, 0, 255)).astype(np.uint8))  # noqa: E731
    row1 = []
    if source is not None and source.exists():
        row1.append(_panel(Image.open(source), "source", height))
    row1 += [_panel(gray(target), "target mask", height),
             _panel(gray(before), f"before IoU {m_before['iou']:.3f}", height),
             _panel(gray(after), f"after IoU {m_after['iou']:.3f}", height)]
    row2 = [_panel(Image.fromarray(error_map(before, target)),
                   "errors before (red excess, blue missing)", height),
            _panel(Image.fromarray(error_map(after, target)), "errors after", height),
            _panel(loss_curve(history, (int(height * 1.6), height)), "loss", height)]
    width = max(sum(p.width for p in row1), sum(p.width for p in row2)) + 10 * 4
    out = Image.new("RGB", (width, row1[0].height + row2[0].height + 10), "white")
    for y, row in ((0, row1), (row1[0].height + 10, row2)):
        x = 0
        for p in row:
            out.paste(p, (x, y))
            x += p.width + 10
    out.save(path)


def render_sheet(render_dir: Path, stems: Sequence[str], path: Path):
    """Stack the Blender renders (one row per mesh, 4 views) into a single image."""
    views = ("front", "three_quarter", "side", "back")
    rows = []
    for s in stems:
        ims = [render_dir / f"{s}__{v}.png" for v in views]
        if not all(p.exists() for p in ims):
            return
        rows.append([Image.open(p).convert("RGB") for p in ims])
    tw, th = rows[0][0].size
    out = Image.new("RGB", (tw * 4, th * len(rows)), "white")
    for r, row in enumerate(rows):
        for c, im in enumerate(row):
            out.paste(im, (c * tw, r * th))
        ImageDraw.Draw(out).text((6, r * th + 6), stems[r], fill="white")
    out.save(path)


def export_glb(verts: np.ndarray, faces: np.ndarray, path: Path):
    import trimesh
    trimesh.Trimesh(vertices=verts, faces=faces, process=False).export(str(path))


# --------------------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------------------

def run(mesh_path: Path, mask_path: Path, out: Path, image_path: Optional[Path], cfg: FitConfig,
        blender: bool = False, force: bool = False, time_budget_s: Optional[float] = None) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "summary.json"
    log_file = open(out / "fit.log", "a", encoding="utf-8")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()

    if summary_path.exists() and not force:
        log(f"{summary_path} exists, skipping (use --force to redo)")
        return json.loads(summary_path.read_text(encoding="utf-8"))

    torch.set_grad_enabled(True)
    device = torch.device(cfg.device)
    t_start = time.perf_counter()
    v_np, f_np = load_mesh(mesh_path)
    alpha_full = load_mask(mask_path)
    H, W = alpha_full.shape
    target = alpha_full > 0.5
    sdf_full = signed_distance(target)
    verts = torch.from_numpy(v_np).to(device)
    faces = torch.from_numpy(f_np).to(device)
    log(f"mesh {mesh_path.name}: {len(v_np)} vertices, {len(f_np)} faces; mask {W}x{H}; "
        f"device {device}, {torch.get_num_threads()} threads")
    history: List[dict] = []
    timings = {}

    cam_path = out / "camera.json"
    t0 = time.perf_counter()
    if cam_path.exists() and not force:
        saved = json.loads(cam_path.read_text(encoding="utf-8"))
        cam, cam_info = OrthoCamera.from_dict(saved["camera"]), saved["info"]
        history += saved.get("history", [])
        log(f"camera loaded from {cam_path}")
    else:
        log("phase 1: camera")
        cam, cam_info = fit_camera(verts, faces, alpha_full, sdf_full, cfg, history, log=log)
        cam_path.write_text(json.dumps({"camera": cam.to_dict(), "info": cam_info,
                                        "history": history}, indent=1), encoding="utf-8")
    timings["camera_s"] = round(time.perf_counter() - t0, 1)
    log(f"camera: {cam.to_dict()}")

    zones = zone_masks(target, cfg.head_fraction, cfg.edge_band_px)
    sil_before = hard_silhouette(verts, faces, cam, (H, W), (H, W))
    m_before = measure(sil_before, target, zones)
    log(f"before: {m_before}")

    log("phase 2: deformation")
    t0 = time.perf_counter()
    new_verts = deform(verts, faces, alpha_full, sdf_full, cam, cfg, history, log, time_budget_s)
    timings["deform_s"] = round(time.perf_counter() - t0, 1)

    sil_after = hard_silhouette(new_verts, faces, cam, (H, W), (H, W))
    m_after = measure(sil_after, target, zones)
    disp = (new_verts - verts).norm(dim=1)
    n_old = face_normals(verts, faces)
    n_new = face_normals(new_verts, faces)
    flip_mask = (n_old * n_new).sum(1) < 0
    flipped = int(flip_mask.sum())
    area = face_normals(verts, faces, normalize=False).norm(dim=1)
    flipped_area = float(area[flip_mask].sum() / area.sum())
    scale_px = cam.scale * H
    m_after.update({
        "flipped_faces": flipped, "flipped_faces_fraction": flipped / len(f_np),
        "flipped_area_fraction": flipped_area,
        "displacement_mean": float(disp.mean()), "displacement_max": float(disp.max()),
        "displacement_mean_px": float(disp.mean()) * scale_px,
        "displacement_max_px": float(disp.max()) * scale_px,
        "moved_vertices_over_1px": int((disp * scale_px > 1.0).sum()),
    })
    log(f"after: {m_after}")

    before_glb, after_glb = out / "before.glb", out / "after.glb"
    if not before_glb.exists():
        shutil.copyfile(mesh_path, before_glb)
    export_glb(new_verts.cpu().numpy(), f_np, after_glb)
    plate(image_path, target, sil_before, sil_after, history, m_before, m_after, out / "plate.png")
    loss_curve(history).save(out / "loss.png")
    timings["total_s"] = round(time.perf_counter() - t_start, 1)

    summary = {
        "mesh": str(mesh_path), "mask": str(mask_path), "image": str(image_path) if image_path else None,
        "vertices": len(v_np), "faces": len(f_np), "mask_size_wh": [W, H],
        "config": asdict(cfg), "camera": cam.to_dict(), "camera_search": cam_info,
        "before": m_before, "after": m_after, "timings": timings,
        "torch": torch.__version__, "threads": torch.get_num_threads(),
    }
    summary_path.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    (out / "history.json").write_text(json.dumps(history), encoding="utf-8")
    log(f"written {summary_path} ({timings})")

    if blender:
        render_dir = out / "renders"
        if BLENDER.exists():
            cmd = [str(BLENDER), "-b", "-P", str(ROOT / "tools/debug/render_glb_views.py"), "--",
                   str(render_dir), str(before_glb), str(after_glb)]
            log("blender: " + " ".join(cmd))
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
            render_sheet(render_dir, ["before", "after"], out / "renders_sheet.png")
        else:
            log(f"Blender not found at {BLENDER}, renders skipped")
    log_file.close()
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True, help="RGBA PNG (alpha) or greyscale mask")
    parser.add_argument("--image", type=Path, default=None, help="source image, for the plate only")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--blender", action="store_true", help="render before/after with Blender")
    parser.add_argument("--force", action="store_true", help="redo even if outputs exist")
    parser.add_argument("--time-budget", type=float, default=None,
                        help="stop the deformation after this many seconds")
    parser.add_argument("--deform-levels", default=None,
                        help="e.g. '256:2:80,512:2:170' (long side:supersample:iterations)")
    parser.add_argument("--w-lap", type=float, default=None)
    parser.add_argument("--w-anchor", type=float, default=None)
    parser.add_argument("--lr", type=float, default=None, help="deformation step in pixels")
    parser.add_argument("--smooth-iters", type=int, default=None)
    parser.add_argument("--direction", choices=["free", "normal"], default=None)
    args = parser.parse_args()

    cfg = FitConfig(device=args.device)
    if args.deform_levels:
        cfg.deform_levels = tuple(tuple(int(x) for x in lvl.split(":"))
                                  for lvl in args.deform_levels.split(","))
    if args.w_lap is not None:
        cfg.w_lap = args.w_lap
    if args.w_anchor is not None:
        cfg.w_anchor = args.w_anchor
    if args.direction is not None:
        cfg.direction = args.direction
    if args.smooth_iters is not None:
        cfg.smooth_iters = args.smooth_iters
    if args.lr is not None:
        cfg.deform_lr_px = args.lr
    if cfg.device.startswith("cuda") and (ROOT / "data/.gpu-lock").exists():
        print("data/.gpu-lock exists: refusing to use the GPU (rule gpu-et-environnement)")
        sys.exit(2)
    run(args.mesh, args.mask, args.out, args.image, cfg, args.blender, args.force, args.time_budget)


if __name__ == "__main__":
    main()
