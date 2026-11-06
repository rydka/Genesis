"""Fit the shape of a mesh to its source image by the silhouette, with FlexiCubes (topology may change).

Follows tools/bench/fit_silhouette.py, whose vertex displacements can only do small corrections:
removing or adding material (an artist's signature turned into a lump, a missing piece of cloth)
needs a representation that changes topology (docs/conception/optimisation-par-zones.md, phase 1).

  1. camera: reused from fit_silhouette (camera.json), or fitted the same way if absent;
  2. initial SDF: signed distance of the input mesh at the vertices of a grid around it (open3d,
     sign by a parity vote over 15 ray directions, see fantasy_rag/diffrender/sdf_grid.py),
     truncated to +-trunc voxels. Extracted without optimisation, it must give back the input shape
     (non-regression measure: chamfer distance "flexicubes_init" in summary.json);
  3. zones of the grid vertices, by their projection into the target mask (one view only):
       - cut: nothing of the object can project outside the mask (visual hull of one view). With
         an orthographic camera the hull is the extrusion of the mask, and the SDF of the cut shape
         is max(SDF, 2D distance to the mask / scale), exact at every depth (CSG intersection);
       - active: projection within `band_px` of the mask boundary (inside, or outside by less than
         half a voxel) and |SDF| < trunc: the optimised parameters;
       - frozen: everything else (projection well inside the mask: the front and back surfaces seen
         through the silhouette, which one view cannot constrain). Their SDF does not change at all;
  4. optimisation of the active SDF values: silhouette rendered by the in-house rasterizer
     (`edge_gradients`, supersampled), long-range loss (rendered coverage x signed distance to the
     target boundary) + L2, FlexiCubes regularisers (L_dev, sign cross-entropy) and a fidelity term
     to the initial SDF; Adam with a uniform denominator (as in fit_silhouette). The change is
     smooth (S^k u over grid neighbours, else voxel terraces) and adding material is softly capped
     (else bubbles in the air next to the surface). Only the cubes near active vertices are
     re-extracted at each iteration;
  5. finishing pass: fit_silhouette's in-plane vertex displacements on the FlexiCubes mesh, capped
     (a few pixels), with the flipped-face penalty;
  6. measures for input / FlexiCubes / FlexiCubes + finishing: IoU (global, head, edge band),
     flipped faces (finishing), normals opposed to the input's, chamfer distance to the input split
     into zones (visible, hidden, silhouette band), times; plate, Blender renders and a sheet that
     stacks the three rows.

Usage (from the repository root; outputs go under data/, never in git):
    .venv\\Scripts\\python.exe tools/bench/fit_flexicubes.py --mesh M.glb --mask MASK.png
        [--image SOURCE.jpg] [--camera data/bench/<date>_fit-silhouette/<stem>/camera.json]
        --out data/bench/<date>_fit-flexicubes/<stem> [--device cpu|cuda|auto] [--res 128] [--blender]

--device cuda/auto take the GPU lock of the main repository (tools/bench/gpu_lock.py) and fall back
to the CPU when another job holds it. The script is resumable: the SDF grid, the FlexiCubes mesh and
the finished mesh are cached in the output folder, and a finished run (summary.json) is skipped
unless --force.
"""

import argparse
import json
import math
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fantasy_rag.diffrender import face_normals  # noqa: E402
from fantasy_rag.diffrender.sdf_grid import (  # noqa: E402
    OPEN3D_AVAILABLE, Grid, SparseFlexiCubes, SparseGrid, cube_corner_reduce, grid_edges,
    grid_points, make_grid, mesh_sdf, sdf_sign_regularizer, sign_change_cubes,
)
from tools.bench import fit_silhouette as fs  # noqa: E402
from tools.bench.gpu_lock import choose_device  # noqa: E402

if OPEN3D_AVAILABLE:
    import open3d as o3d


@dataclass
class FlexiConfig:
    device: str = "cpu"
    res: int = 128                      # cubes along the longest side of the mesh bounding box
    pad: int = 4                        # empty cubes around the box
    trunc_voxels: float = 6.0           # |SDF| clamped to trunc_voxels * h; active only below it
    n_dirs: int = 15                    # ray directions of the sign vote
    carve: bool = True
    carve_margin_px: float = 0.0        # the visual hull is the mask grown by this many pixels
    # The mask distance is smoothed (Gaussian, pixels) before the cut, so that the pixel staircase of
    # the mask contour is not extruded into the cut wall; the cut is a smooth max (round edge where
    # it meets the original surface) of radius carve_round_voxels.
    mask_smooth_px: float = 1.5
    carve_round_voxels: float = 1.5
    band_px: float = 6.0                # + 2 voxels (in pixels): active band inside the boundary
    # (long side, supersample, iterations) per level
    levels: Sequence[Tuple[int, int, int]] = ((256, 2, 60), (512, 2, 100))
    lr_voxels: float = 0.25             # largest SDF step per iteration, in voxels
    smooth_iters: int = 48              # change = S^k u, S = averaging with grid neighbours
    # Soft cap on adding material, in voxels: lowering the SDF far from the surface creates bubbles
    # in the air next to it (beads along the hair); within the cap they merge with the surface.
    # Larger gaps are left to the finishing pass. 0 = no cap.
    max_grow_voxels: float = 4.0
    w_sdf: float = 0.5
    w_l2: float = 1.0
    w_fid: float = 2e-4                 # mean over active vertices of ((s - s_ref) / h)^2
    w_dev: float = 0.02                 # FlexiCubes L_dev (in voxels)
    w_sign: float = 1e-3                # FlexiCubes sign cross-entropy (on s / h)
    log_every: int = 10


# --------------------------------------------------------------------------------------------------
# Projection of points into the target
# --------------------------------------------------------------------------------------------------

def project_px(points: torch.Tensor, cam: fs.OrthoCamera) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pixel coordinates (x, y) of the full target (origin top-left, H = 1 unit) and camera-axis
    depth q.z (larger = closer to the camera) of mesh-space points [N, 3]."""
    p = fs.CameraParams(cam).to(points.device)
    with torch.no_grad():
        q = points @ p.rot().T
        s = p.log_scale.exp()
        u = p.center[0] + s * q[:, 0]
        v = p.center[1] - s * q[:, 1]
    return torch.stack([u, v], 1), q[:, 2]


def sample_image(img: np.ndarray, uv: torch.Tensor, outside: float) -> np.ndarray:
    """Bilinear sample of an [H, W] image at uv (units of H); `outside` beyond the image."""
    H, W = img.shape
    x = uv[:, 0].cpu().numpy() * H - 0.5
    y = uv[:, 1].cpu().numpy() * H - 0.5
    from scipy import ndimage
    return ndimage.map_coordinates(img, [y, x], order=1, mode="constant", cval=outside)


def mask_distance_px(points: torch.Tensor, cam: fs.OrthoCamera, sdf_full: np.ndarray) -> np.ndarray:
    """Signed distance in pixels from the projection of each point to the target boundary
    (> 0 outside the mask). Beyond the image: far outside."""
    uv, _ = project_px(points, cam)
    return sample_image(sdf_full, uv, outside=1e4)


# --------------------------------------------------------------------------------------------------
# Initial SDF and zones
# --------------------------------------------------------------------------------------------------

def initial_sdf(v_np: np.ndarray, f_np: np.ndarray, cfg: FlexiConfig, cache: Optional[Path], log
                ) -> Tuple[Grid, np.ndarray, dict]:
    """(grid, SDF at every grid vertex [vx, vy, vz] truncated, info); cached in `cache` (.npz)."""
    grid = make_grid(v_np.min(0), v_np.max(0), cfg.res, cfg.pad)
    if cache is not None and cache.exists():
        z = np.load(cache)
        if (np.allclose(z["origin"], grid.origin) and np.isclose(float(z["h"]), grid.h)
                and tuple(z["dims"]) == grid.dims):
            log(f"SDF loaded from {cache}")
            return grid, z["s"], json.loads(str(z["info"]))
    t0 = time.perf_counter()
    pts = grid_points(grid)
    s, info = mesh_sdf(v_np, f_np, pts, cfg.n_dirs)
    info["seconds"] = round(time.perf_counter() - t0, 1)
    info["points"] = int(len(pts))
    log(f"SDF of {len(pts)} grid vertices ({grid.dims} cubes, h = {grid.h:.5f}) in "
        f"{info['seconds']} s; sign vote: {info['ambiguous']} ambiguous points, "
        f"{info['unanimous_fraction']:.4%} unanimous")
    s = s.reshape(grid.vertex_dims)
    if cache is not None:
        np.savez_compressed(cache, s=s, origin=grid.origin, h=grid.h, dims=np.array(grid.dims),
                            info=json.dumps(info))
    return grid, s, info


@dataclass
class Zones:
    carved: np.ndarray    # [vx, vy, vz] bool
    active: np.ndarray
    s_ref: np.ndarray     # reference SDF after carving (truncated), [vx, vy, vz]


def smooth_max(a: np.ndarray, b: np.ndarray, k: float) -> np.ndarray:
    """Polynomial smooth maximum of radius k (= max where |a - b| >= k)."""
    if k <= 0:
        return np.maximum(a, b)
    h = np.clip(k - np.abs(a - b), 0, None) / k
    return np.maximum(a, b) + h * h * k / 4


def make_zones(grid: Grid, s0: np.ndarray, cam: fs.OrthoCamera, sdf_full: np.ndarray,
               cfg: FlexiConfig) -> Tuple[Zones, dict]:
    from scipy import ndimage
    H = sdf_full.shape[0]
    voxel_px = grid.h * cam.scale * H
    pts = torch.from_numpy(grid_points(grid))
    sdf_s = (ndimage.gaussian_filter(sdf_full, cfg.mask_smooth_px) if cfg.mask_smooth_px > 0
             else sdf_full)
    md = mask_distance_px(pts, cam, sdf_s).reshape(grid.vertex_dims)
    trunc = cfg.trunc_voxels * grid.h
    s = s0
    info = {}
    if cfg.carve:
        # Orthographic camera: the visual hull of the mask is its extrusion along the view axis, and
        # the distance of a point to it is the 2D distance of its projection to the mask. The SDF of
        # the intersection (CSG) is max(s0, d_hull), here a smooth max: an exact cut at every depth.
        d_hull = (md - cfg.carve_margin_px) / (cam.scale * H)
        s = smooth_max(s0, d_hull, cfg.carve_round_voxels * grid.h)
    carved = (s0 < 0) & (s >= 0)                      # material removed by the cut
    s = np.clip(s, -trunc, trunc)
    outside = md > cfg.carve_margin_px + 0.5 * voxel_px
    near = (md >= -(cfg.band_px + 2 * voxel_px)) & ~outside
    active = near & (np.abs(s) < trunc)
    info.update({"voxel_px": float(voxel_px), "carved_inside_vertices": int(carved.sum()),
                 "active_vertices": int(active.sum()), "band_px_total": cfg.band_px + 2 * voxel_px,
                 "frozen_outside_threshold_px": cfg.carve_margin_px + 0.5 * voxel_px})
    return Zones(carved, active, s.astype(np.float32)), info


def build_sparse(grid: Grid, zones: Zones, device) -> SparseGrid:
    """Cubes that can hold the surface: sign change now, or a corner that may change sign."""
    cubes = sign_change_cubes(zones.s_ref) | cube_corner_reduce(zones.active, np.logical_or)
    return SparseGrid.from_cube_mask(grid, cubes, device)


# --------------------------------------------------------------------------------------------------
# Optimisation
# --------------------------------------------------------------------------------------------------

def split_dynamic(sparse: SparseGrid, active_idx: torch.Tensor
                  ) -> Tuple[SparseGrid, torch.Tensor, SparseGrid, torch.Tensor]:
    """Split the band into (dynamic sub-grid, its vertex indices in `sparse`, static sub-grid, its
    vertex indices). Dynamic: cubes within one cube of a cube that has an active corner, so that
    every surface quad (4 cubes around a grid edge) lies entirely in one of the two parts; static:
    cubes without an active corner (the overlap ring gives identical, constant faces in both)."""
    from scipy import ndimage
    dev = sparse.cubes.device
    is_active = torch.zeros(sparse.verts.shape[0], dtype=torch.bool, device=dev)
    is_active[active_idx] = True
    touching = is_active[sparse.cubes].any(1).cpu().numpy()
    ijk = sparse.cube_ijk.cpu().numpy()
    vol = np.zeros(sparse.grid.dims, bool)
    vol[tuple(ijk[touching].T)] = True
    vol = ndimage.binary_dilation(vol, np.ones((3, 3, 3), bool))
    dynamic = vol[tuple(ijk.T)]

    def sub(mask: np.ndarray) -> Tuple[SparseGrid, torch.Tensor]:
        m = torch.from_numpy(mask).to(dev)
        used, inv = torch.unique(sparse.cubes[m], return_inverse=True)
        return SparseGrid(sparse.grid, sparse.cube_ijk[m], inv.view(-1, 8), sparse.vert_flat[used],
                          sparse.verts[used]), used

    d, d_idx = sub(dynamic)
    st, st_idx = sub(~touching)
    return d, d_idx, st, st_idx


def optimise(sparse: SparseGrid, s_ref: torch.Tensor, active_idx: torch.Tensor, alpha_full, sdf_full,
             cam: fs.OrthoCamera, zscale: float, cfg: FlexiConfig, history: List[dict], log,
             time_budget_s: Optional[float] = None) -> torch.Tensor:
    """Optimise the SDF of the active vertices of `sparse`; returns the full SDF [N].

    Only the cubes near active vertices are extracted at every iteration; the rest of the band
    (constant) is rendered once per pyramid level as a fixed coverage image and composited
    (alpha = 1 - (1 - a_dynamic)(1 - a_static)). It lies well inside the silhouette, so it carries no
    silhouette gradient anyway."""
    device = s_ref.device
    H, W = alpha_full.shape
    h = sparse.grid.h
    fc = SparseFlexiCubes(device)
    dyn, dyn_idx, static, static_idx = split_dynamic(sparse, active_idx)
    with torch.no_grad():
        v_st, f_st, _ = fc.extract(static, s_ref[static_idx])
    # position of every active vertex inside the dynamic sub-grid
    pos = torch.full((sparse.verts.shape[0],), -1, dtype=torch.long, device=device)
    pos[dyn_idx] = torch.arange(dyn_idx.shape[0], device=device)
    act_in_dyn = pos[active_idx]
    assert bool((act_in_dyn >= 0).all())
    s_dyn_ref = s_ref[dyn_idx]
    edges = grid_edges(dyn.cubes)
    # Smooth parametrisation of the change (as fit_silhouette does for displacements): the free
    # parameters u live on the active vertices, the applied change is S^k u restricted to them, S the
    # averaging with grid neighbours. Without it each vertex moves alone and the surface gets voxel
    # terraces.
    n_dyn = dyn_idx.shape[0]
    act_mask = torch.zeros(n_dyn, 1, device=device)
    act_mask[act_in_dyn] = 1.0
    e2 = torch.cat([edges, edges.flip(1)])
    deg = torch.bincount(e2[:, 0], minlength=n_dyn).clamp(min=1).float()
    avg = torch.sparse_coo_tensor(e2.T, 1.0 / deg[e2[:, 0]], (n_dyn, n_dyn)).coalesce()

    def change():                                  # [n_dyn] in voxels, 0 off the active vertices
        d = torch.zeros(n_dyn, 1, device=device).index_add(0, act_in_dyn, x.unsqueeze(1))
        for _ in range(cfg.smooth_iters):
            d = 0.5 * (d + torch.sparse.mm(avg, d)) * act_mask
        d = d[:, 0]
        if cfg.max_grow_voxels > 0:                # soft cap on adding material (d < 0)
            g = cfg.max_grow_voxels
            d = torch.where(d < 0, -g * torch.tanh(-d / g), d)
        return d
    log(f"  dynamic part: {dyn.n_cubes} cubes, {dyn_idx.numel()} vertices; static part: "
        f"{static.n_cubes} cubes, {f_st.shape[0]} faces (rendered once per level)")
    with torch.no_grad():
        p = fs.CameraParams(cam).to(device)
        rot, scale, center = p.rot(), p.log_scale.exp(), p.center.clone()
    x = torch.zeros(active_idx.shape[0], device=device, requires_grad=True)   # change in voxels
    opt = fs.UniformAdam([x], lr=cfg.lr_voxels)
    t0 = time.perf_counter()

    def field():
        with torch.no_grad():
            s_full = s_ref.clone()
            s_full[dyn_idx] = s_dyn_ref + change() * h
            return s_full

    stop = False
    for long_side, ss, iters in cfg.levels:
        hw = fs.level_size((H, W), long_side)
        tgt = fs.Target.make(alpha_full, sdf_full, hw, device)
        with torch.no_grad():
            clip_st = fs.to_clip(v_st @ rot.T, scale, center, W / H, zscale)
            a_static = (fs.render_alpha(clip_st, f_st, hw, ss, grad=False) if f_st.shape[0]
                        else torch.zeros(hw, device=device))
        for it in range(iters):
            opt.zero_grad()
            dc = change()
            s = s_dyn_ref + dc * h
            v, f, l_dev = fc.extract(dyn, s)
            clip = fs.to_clip(v @ rot.T, scale, center, W / H, zscale)
            alpha = 1 - (1 - fs.render_alpha(clip, f, hw, ss)) * (1 - a_static)
            l_sdf, l_l2 = fs.silhouette_losses(alpha, tgt)
            l_fid = (dc[act_in_dyn] ** 2).mean()
            l_dev_m = l_dev.mean() / h if l_dev.numel() else torch.zeros((), device=device)
            l_sign = (sdf_sign_regularizer(s / h, edges) if cfg.w_sign > 0
                      else torch.zeros((), device=device))
            loss = (cfg.w_sdf * l_sdf + cfg.w_l2 * l_l2 + cfg.w_fid * l_fid + cfg.w_dev * l_dev_m
                    + cfg.w_sign * l_sign)
            loss.backward()
            opt.step()
            rec = {"phase": "flexicubes", "level": long_side, "iter": it, "loss": float(loss),
                   "sdf": float(l_sdf), "l2": float(l_l2), "fid": float(l_fid),
                   "dev": float(l_dev_m), "sign": float(l_sign), "faces": int(f.shape[0])}
            history.append(rec)
            if it % cfg.log_every == 0 or it == iters - 1:
                log(f"  flexicubes {long_side}x{ss} it {it:3d}: loss {rec['loss']:.5f} "
                    f"l2 {rec['l2']:.5f} sdf {rec['sdf']:.5f} fid {rec['fid']:.3e} "
                    f"dev {rec['dev']:.3e} max|ds| {float(dc.abs().max()):.2f} vox "
                    f"faces {rec['faces']} ({time.perf_counter() - t0:.0f} s)")
            if time_budget_s and time.perf_counter() - t0 > time_budget_s:
                log(f"  time budget of {time_budget_s:.0f} s reached, stopping")
                stop = True
                break
        if stop:
            break
    with torch.no_grad():
        return field()


# --------------------------------------------------------------------------------------------------
# Measures
# --------------------------------------------------------------------------------------------------

def sample_surface(verts: np.ndarray, faces: np.ndarray, n: int, seed: int = 0
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """n points uniformly on the surface (area-weighted) and their face index."""
    rng = np.random.default_rng(seed)
    tri = verts[faces].astype(np.float64)
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    fid = rng.choice(len(faces), size=n, p=area / area.sum())
    r1, r2 = rng.random(n), rng.random(n)
    s1 = np.sqrt(r1)
    w = np.stack([1 - s1, s1 * (1 - r2), s1 * r2], 1)
    pts = (tri[fid] * w[:, :, None]).sum(1)
    return pts.astype(np.float32), fid


class SurfaceQuery:
    """Closest-point queries on a triangle mesh (open3d)."""

    def __init__(self, verts: np.ndarray, faces: np.ndarray):
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(o3d.core.Tensor(np.ascontiguousarray(verts, np.float32)),
                                 o3d.core.Tensor(np.ascontiguousarray(faces, np.uint32)))

    def distance(self, pts: np.ndarray) -> np.ndarray:
        return self.scene.compute_distance(o3d.core.Tensor(np.ascontiguousarray(pts, np.float32))
                                           ).numpy()

    def closest_face(self, pts: np.ndarray) -> np.ndarray:
        r = self.scene.compute_closest_points(o3d.core.Tensor(np.ascontiguousarray(pts, np.float32)))
        return r["primitive_ids"].numpy().astype(np.int64)


class ZoneLabeller:
    """Zone of 3D points w.r.t. the input mesh and the view: 'band' (projects within `band_px` of
    the target boundary or outside the mask), else 'visible' (front-most surface of the input mesh at
    that pixel, within `tol_px`) or 'hidden'."""

    def __init__(self, verts: torch.Tensor, faces: torch.Tensor, cam: fs.OrthoCamera,
                 sdf_full: np.ndarray, band_px: float, tol_px: float = 2.0):
        H, W = sdf_full.shape
        self.cam, self.sdf_full, self.band_px, self.H = cam, sdf_full, band_px, H
        uv, qz = project_px(verts, cam)
        # front-most depth (largest q.z) per pixel by a z-buffer of the input mesh: rasterize at
        # full resolution and interpolate q.z
        with torch.no_grad():
            p = fs.CameraParams(cam)
            q = verts @ p.rot().T
            clip = fs.to_clip(q, p.log_scale.exp(), p.center, W / H, fs._zscale(verts))
            from fantasy_rag.diffrender import interpolate, rasterize
            rast = rasterize(clip, faces, (H, W))
            z = interpolate(q[:, 2:3].contiguous(), rast, faces)[0, ..., 0]
            self.zbuf = torch.where(rast.mask[0], z, torch.full_like(z, -1e9)).cpu().numpy()
        self.tol = tol_px / (cam.scale * H)                  # in mesh units

    def __call__(self, pts: np.ndarray) -> np.ndarray:
        t = torch.from_numpy(pts)
        uv, qz = project_px(t, self.cam)
        md = sample_image(self.sdf_full, uv, outside=1e4)
        H, W = self.zbuf.shape
        xi = np.clip((uv[:, 0].numpy() * H).astype(np.int64), 0, W - 1)
        yi = np.clip((uv[:, 1].numpy() * H).astype(np.int64), 0, H - 1)
        visible = qz.numpy() >= self.zbuf[yi, xi] - self.tol
        zone = np.where(visible, "visible", "hidden").astype(object)
        zone[md > -self.band_px] = "band"
        return zone


def chamfer_by_zone(ref_q: SurfaceQuery, ref_pts: np.ndarray, ref_zone: np.ndarray,
                    out_v: np.ndarray, out_f: np.ndarray, labeller: ZoneLabeller, px_per_unit: float,
                    n: int, seed: int = 1) -> Dict[str, dict]:
    """Symmetric chamfer (mean of the two directions, L1) per zone, in pixels of the target."""
    out_pts, _ = sample_surface(out_v, out_f, n, seed)
    out_zone = labeller(out_pts)
    d_ref_to_out = SurfaceQuery(out_v, out_f).distance(ref_pts) * px_per_unit
    d_out_to_ref = ref_q.distance(out_pts) * px_per_unit
    res = {}
    for z in ("visible", "hidden", "band", "all"):
        a = d_ref_to_out if z == "all" else d_ref_to_out[ref_zone == z]
        b = d_out_to_ref if z == "all" else d_out_to_ref[out_zone == z]
        if len(a) == 0 or len(b) == 0:
            continue
        res[z] = {"chamfer_px": float(0.5 * (a.mean() + b.mean())),
                  "input_to_out_mean_px": float(a.mean()), "out_to_input_mean_px": float(b.mean()),
                  "input_to_out_p95_px": float(np.percentile(a, 95)),
                  "out_to_input_p95_px": float(np.percentile(b, 95)),
                  "input_points_over_2px": float((a > 2).mean()),
                  "points": [int(len(a)), int(len(b))]}
    return res


def opposed_normals(ref_q: SurfaceQuery, ref_normals: np.ndarray, verts: np.ndarray,
                    faces: np.ndarray) -> float:
    """Area fraction of faces whose normal points against the normal of the closest input face
    (inside-out or crumpled pieces). Thin parts give a small floor: measure the input too."""
    tri = verts[faces]
    c = tri.mean(1)
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = np.linalg.norm(n, axis=1)
    nn = n / np.maximum(area, 1e-20)[:, None]
    ref = ref_normals[ref_q.closest_face(c)]
    return float(area[(nn * ref).sum(1) < 0].sum() / max(area.sum(), 1e-20))


def mesh_measures(verts: torch.Tensor, faces: torch.Tensor, cam, target, zones, H, W) -> dict:
    sil = fs.hard_silhouette(verts, faces, cam, (H, W), (H, W))
    m = fs.measure(sil, target, zones)
    m["vertices"], m["faces"] = int(verts.shape[0]), int(faces.shape[0])
    return m, sil


# --------------------------------------------------------------------------------------------------
# Outputs
# --------------------------------------------------------------------------------------------------

def plate3(source: Optional[Path], target, sils: Dict[str, np.ndarray], metrics: Dict[str, dict],
           path: Path, height: int = 340):
    gray = lambda b: Image.fromarray((np.where(b, 0, 255)).astype(np.uint8))  # noqa: E731
    row1 = []
    if source is not None and source.exists():
        row1.append(fs._panel(Image.open(source), "source", height))
    row1.append(fs._panel(gray(target), "target mask", height))
    row2 = []
    for name, sil in sils.items():
        row1.append(fs._panel(gray(sil), f"{name} IoU {metrics[name]['iou']:.3f}", height))
        row2.append(fs._panel(Image.fromarray(fs.error_map(sil, target)),
                              f"errors {name} (red excess, blue missing)", height))
    width = max(sum(p.width for p in row1), sum(p.width for p in row2)) + 10 * len(row1)
    out = Image.new("RGB", (width, row1[0].height + row2[0].height + 10), "white")
    for y, row in ((0, row1), (row1[0].height + 10, row2)):
        x = 0
        for p in row:
            out.paste(p, (x, y))
            x += p.width + 10
    out.save(path)


# --------------------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------------------

def load_camera(verts, faces, alpha_full, sdf_full, camera_json: Optional[Path], out: Path, log
                ) -> fs.OrthoCamera:
    for p in (camera_json, out / "camera.json"):
        if p is not None and p.exists():
            saved = json.loads(p.read_text(encoding="utf-8"))
            log(f"camera loaded from {p}")
            if p != out / "camera.json":
                shutil.copyfile(p, out / "camera.json")
            return fs.OrthoCamera.from_dict(saved["camera"])
    log("no camera.json: fitting the camera (fit_silhouette phase 1)")
    cam, info = fs.fit_camera(verts, faces, alpha_full, sdf_full, fs.FitConfig(device=str(verts.device)),
                              log=log)
    (out / "camera.json").write_text(json.dumps({"camera": cam.to_dict(), "info": info}, indent=1),
                                     encoding="utf-8")
    return cam


def run(mesh_path: Path, mask_path: Path, out: Path, image_path: Optional[Path], cfg: FlexiConfig,
        finish_cfg: Optional[fs.FitConfig], camera_json: Optional[Path] = None, blender: bool = False,
        force: bool = False, time_budget_s: Optional[float] = None,
        finish_budget_s: Optional[float] = None, n_samples: int = 300_000) -> dict:
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
    if not OPEN3D_AVAILABLE:
        raise RuntimeError("open3d is required (initial SDF and distances)")

    torch.set_grad_enabled(True)
    device = torch.device(cfg.device)
    t_start = time.perf_counter()
    timings = {}
    v_np, f_np = fs.load_mesh(mesh_path)
    alpha_full = fs.load_mask(mask_path)
    H, W = alpha_full.shape
    target = alpha_full > 0.5
    sdf_full = fs.signed_distance(target)
    verts = torch.from_numpy(v_np)
    faces = torch.from_numpy(f_np)
    log(f"mesh {mesh_path.name}: {len(v_np)} vertices, {len(f_np)} faces; mask {W}x{H}; "
        f"device {device}, {torch.get_num_threads()} threads; config {asdict(cfg)}")
    cam = load_camera(verts, faces, alpha_full, sdf_full, camera_json, out, log)
    zscale = fs._zscale(verts)
    px_per_unit = cam.scale * H

    # Initial SDF, zones, sparse grid
    t0 = time.perf_counter()
    grid, s0, sdf_info = initial_sdf(v_np, f_np, cfg, out / f"sdf_{cfg.res}.npz", log)
    zones, zone_info = make_zones(grid, s0, cam, sdf_full, cfg)
    sparse = build_sparse(grid, zones, device)
    flat = sparse.vert_flat.cpu().numpy()
    s_ref = torch.from_numpy(zones.s_ref.reshape(-1)[flat]).to(device)
    s_init = torch.from_numpy(np.clip(s0, -cfg.trunc_voxels * grid.h, cfg.trunc_voxels * grid.h
                                      ).reshape(-1)[flat]).to(device)
    active_idx = torch.from_numpy(np.nonzero(zones.active.reshape(-1)[flat])[0]).to(device)
    timings["sdf_s"] = round(time.perf_counter() - t0, 1)
    log(f"grid {grid.to_dict()}; voxel {zone_info['voxel_px']:.2f} px; {sparse.n_cubes} cubes in the "
        f"band, {sparse.verts.shape[0]} vertices, {active_idx.numel()} active; zones {zone_info}")

    fc = SparseFlexiCubes(device)
    stages: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {"input": (verts, faces)}
    # Non-regression: extraction of the initial SDF, no carving, no optimisation
    sparse_init = SparseGrid.from_cube_mask(grid, sign_change_cubes(s0), device)
    with torch.no_grad():
        v_i, f_i, _ = fc.extract(sparse_init, torch.from_numpy(
            s0.reshape(-1)[sparse_init.vert_flat.cpu().numpy()]).to(device))
    stages["flexicubes_init"] = (v_i.cpu(), f_i.cpu())
    del sparse_init

    # Optimisation (cached)
    fc_glb = out / "flexicubes.glb"
    history: List[dict] = []
    t0 = time.perf_counter()
    if fc_glb.exists() and not force:
        v_fc, f_fc = (torch.from_numpy(a) for a in fs.load_mesh(fc_glb))
        log(f"FlexiCubes mesh loaded from {fc_glb}")
        hist_path = out / "history_flexicubes.json"
        if hist_path.exists():
            history = json.loads(hist_path.read_text(encoding="utf-8"))
    else:
        with torch.no_grad():
            v_c, f_c, _ = fc.extract(sparse, s_ref)
        stages["carved"] = (v_c.cpu(), f_c.cpu())
        log("FlexiCubes optimisation")
        s_opt = optimise(sparse, s_ref, active_idx, alpha_full, sdf_full, cam, zscale, cfg, history,
                         log, time_budget_s)
        with torch.no_grad():
            v_fc, f_fc, _ = fc.extract(sparse, s_opt)
            ds = (s_opt - s_init).abs() / grid.h
        v_fc, f_fc = v_fc.cpu(), f_fc.cpu()
        fs.export_glb(v_fc.numpy(), f_fc.numpy(), fc_glb)
        (out / "history_flexicubes.json").write_text(json.dumps(history), encoding="utf-8")
        timings["flexicubes_optim_s"] = round(time.perf_counter() - t0, 1)
        zone_info["sdf_change_vox_max"] = float(ds.max())
        zone_info["vertices_changed_over_half_voxel"] = int((ds > 0.5).sum())
    stages["flexicubes"] = (v_fc, f_fc)

    # Finishing pass (cached)
    fin_glb = out / "flexicubes_finish.glb"
    if finish_cfg is not None:
        t0 = time.perf_counter()
        if fin_glb.exists() and not force:
            v_fin = torch.from_numpy(fs.load_mesh(fin_glb)[0])
            log(f"finished mesh loaded from {fin_glb}")
        else:
            log("finishing pass (fit_silhouette deformation, capped)")
            v_fin = fs.deform(v_fc.to(device), f_fc.to(device), alpha_full, sdf_full, cam, finish_cfg,
                              history, log, finish_budget_s).cpu()
            fs.export_glb(v_fin.numpy(), f_fc.numpy(), fin_glb)
            timings["finish_s"] = round(time.perf_counter() - t0, 1)
        stages["flexicubes_finish"] = (v_fin, f_fc)

    # Measures
    t0 = time.perf_counter()
    zone_masks = fs.zone_masks(target, 0.2, 4.0)
    metrics, sils = {}, {}
    for name, (v, f) in stages.items():
        metrics[name], sils[name] = mesh_measures(v, f, cam, target, zone_masks, H, W)
    if "flexicubes_finish" in stages:
        n, a = fs.flipped_faces(v_fc, stages["flexicubes_finish"][0], f_fc)
        metrics["flexicubes_finish"].update({"flipped_faces": n, "flipped_faces_fraction": n / len(f_fc),
                                             "flipped_area_fraction": a})
        disp = (stages["flexicubes_finish"][0] - v_fc).norm(dim=1) * px_per_unit
        metrics["flexicubes_finish"].update({"displacement_max_px": float(disp.max()),
                                             "displacement_mean_px": float(disp.mean())})
    ref_q = SurfaceQuery(v_np, f_np)
    ref_n = face_normals(verts, faces).numpy()
    band_meas = cfg.band_px + 2 * zone_info["voxel_px"]
    labeller = ZoneLabeller(verts, faces, cam, sdf_full, band_meas)
    ref_pts, _ = sample_surface(v_np, f_np, n_samples, 0)
    ref_zone = labeller(ref_pts)
    for name, (v, f) in stages.items():
        if name == "input":
            metrics[name]["opposed_normals_area"] = opposed_normals(ref_q, ref_n, v_np, f_np)
            continue
        metrics[name]["chamfer"] = chamfer_by_zone(ref_q, ref_pts, ref_zone, v.numpy(), f.numpy(),
                                                   labeller, px_per_unit, n_samples)
        metrics[name]["opposed_normals_area"] = opposed_normals(ref_q, ref_n, v.numpy(), f.numpy())
    timings["measures_s"] = round(time.perf_counter() - t0, 1)
    for name, m in metrics.items():
        ch = m.get("chamfer", {})
        log(f"{name}: IoU {m['iou']:.4f} head {m['iou_head']:.4f} edge {m['iou_edge']:.4f}; "
            f"{m['faces']} faces; opposed normals {m['opposed_normals_area']:.4%}; chamfer px "
            + ", ".join(f"{z} {c['chamfer_px']:.2f}" for z, c in ch.items()))

    in_glb = out / "input.glb"
    if not in_glb.exists():
        shutil.copyfile(mesh_path, in_glb)
    if "carved" in stages:
        fs.export_glb(stages["carved"][0].numpy(), stages["carved"][1].numpy(), out / "carved.glb")
    fs.export_glb(v_i.cpu().numpy(), f_i.cpu().numpy(), out / "flexicubes_init.glb")
    plate_sils = {k: sils[k] for k in ("input", "flexicubes", "flexicubes_finish") if k in sils}
    plate3(image_path, target, plate_sils, metrics, out / "plate.png")
    if history:
        fs.loss_curve(history).save(out / "loss.png")
    timings["total_s"] = round(time.perf_counter() - t_start, 1)

    summary = {
        "mesh": str(mesh_path), "mask": str(mask_path), "image": str(image_path) if image_path else None,
        "mask_size_wh": [W, H], "config": asdict(cfg),
        "finish_config": asdict(finish_cfg) if finish_cfg is not None else None,
        "camera": cam.to_dict(), "grid": grid.to_dict(), "sdf": sdf_info, "zones": zone_info,
        "sparse": {"cubes": sparse.n_cubes, "vertices": int(sparse.verts.shape[0]),
                   "active": int(active_idx.numel())},
        "metrics": metrics, "timings": timings, "device": str(device), "torch": torch.__version__,
        "threads": torch.get_num_threads(),
    }
    if device.type == "cuda":
        summary["cuda_peak_gib"] = torch.cuda.max_memory_allocated() / 2 ** 30
    summary_path.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    log(f"written {summary_path} ({timings})")

    if blender:
        render_dir = out / "renders"
        names = [n for n in ("input", "flexicubes", "flexicubes_finish") if (out / f"{n}.glb").exists()]
        if fs.BLENDER.exists():
            cmd = [str(fs.BLENDER), "-b", "-P", str(ROOT / "tools/debug/render_glb_views.py"), "--",
                   str(render_dir)] + [str(out / f"{n}.glb") for n in names]
            log("blender: " + " ".join(cmd))
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
            fs.render_sheet(render_dir, names, out / "renders_sheet.png")
        else:
            log(f"Blender not found at {fs.BLENDER}, renders skipped")
    log_file.close()
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True, help="RGBA PNG (alpha) or greyscale mask")
    parser.add_argument("--image", type=Path, default=None, help="source image, for the plate only")
    parser.add_argument("--camera", type=Path, default=None,
                        help="camera.json of a fit_silhouette run (else fitted here)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu", help="cpu, cuda or auto (GPU lock respected)")
    parser.add_argument("--gpu-lock", type=Path, default=None)
    parser.add_argument("--take-stale-lock", action="store_true")
    parser.add_argument("--res", default="auto",
                        help="grid resolution (longest side); auto = 256 on the GPU, 128 on the CPU")
    parser.add_argument("--levels", default=None, help="e.g. '256:2:60,512:2:100'")
    parser.add_argument("--no-carve", action="store_true", help="no visual-hull carving")
    parser.add_argument("--band-px", type=float, default=None)
    parser.add_argument("--w-fid", type=float, default=None)
    parser.add_argument("--lr", type=float, default=None, help="SDF step in voxels")
    parser.add_argument("--smooth-iters", type=int, default=None)
    parser.add_argument("--max-grow", type=float, default=None, help="cap on added material, voxels")
    parser.add_argument("--no-finish", action="store_true", help="skip the finishing pass")
    parser.add_argument("--finish-cap-px", type=float, default=3.0)
    parser.add_argument("--finish-levels", default="512:2:60")
    parser.add_argument("--finish-w-flip", type=float, default=None)
    parser.add_argument("--time-budget", type=float, default=None, help="FlexiCubes, seconds")
    parser.add_argument("--finish-budget", type=float, default=None, help="finishing, seconds")
    parser.add_argument("--samples", type=int, default=300_000, help="points per mesh (chamfer)")
    parser.add_argument("--blender", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    def levels(text):
        return tuple(tuple(int(x) for x in lvl.split(":")) for lvl in text.split(","))

    device, lock = choose_device(args.device, "fit_flexicubes",
                                 f"FlexiCubes fit of {args.mesh.name} -> {args.out}", args.gpu_lock,
                                 take_stale=args.take_stale_lock)
    try:
        cfg = FlexiConfig(device=device)
        cfg.res = (256 if device.startswith("cuda") else 128) if args.res == "auto" else int(args.res)
        if args.levels:
            cfg.levels = levels(args.levels)
        if args.no_carve:
            cfg.carve = False
        if args.band_px is not None:
            cfg.band_px = args.band_px
        if args.w_fid is not None:
            cfg.w_fid = args.w_fid
        if args.lr is not None:
            cfg.lr_voxels = args.lr
        if args.smooth_iters is not None:
            cfg.smooth_iters = args.smooth_iters
        if args.max_grow is not None:
            cfg.max_grow_voxels = args.max_grow
        finish = None
        if not args.no_finish:
            finish = fs.FitConfig(device=device, deform_levels=levels(args.finish_levels),
                                  max_disp_px=args.finish_cap_px, w_flip=FINISH_W_FLIP)
            if args.finish_w_flip is not None:
                finish.w_flip = args.finish_w_flip
        run(args.mesh, args.mask, args.out, args.image, cfg, finish, args.camera, args.blender,
            args.force, args.time_budget, args.finish_budget, args.samples)
    finally:
        if lock is not None:
            lock.release()


FINISH_W_FLIP = 1.0   # flipped-face penalty of the finishing pass (see fit_silhouette.FitConfig)


if __name__ == "__main__":
    main()
