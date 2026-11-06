"""FlexiCubes (vendored), sparse grid, mesh SDF and the FlexiCubes silhouette fit; CPU, no external files."""

import numpy as np
import pytest
import torch

from fantasy_rag.diffrender import face_normals, uv_sphere
from fantasy_rag.diffrender.flexicubes import FlexiCubes
from fantasy_rag.diffrender.sdf_grid import (
    OPEN3D_AVAILABLE, Grid, SparseFlexiCubes, SparseGrid, grid_points, make_grid, mesh_sdf,
    sign_change_cubes,
)

needs_open3d = pytest.mark.skipif(not OPEN3D_AVAILABLE, reason="open3d not installed")


def _two_spheres(points):
    a = np.linalg.norm(points - np.array([0.1, 0.0, 0.0]), axis=1) - 0.6
    b = np.linalg.norm(points - np.array([-0.5, 0.3, 0.2]), axis=1) - 0.3
    return np.minimum(a, b).astype(np.float32)


def test_sparse_flexicubes_matches_upstream_dense():
    # Same grid as upstream construct_voxel_grid(n): [-0.5, 0.5]^3, n cubes per side, pad 0.
    # A noisy field creates the ambiguous C16 / C19 configurations that _get_case_id resolves.
    n = 14
    grid = Grid(np.full(3, -0.5), 1.0 / n, (n, n, n))
    pts = grid_points(grid)
    rng = np.random.default_rng(0)
    s = (np.linalg.norm(pts, axis=1) - 0.33 + 0.05 * rng.standard_normal(len(pts))).astype(np.float32)

    up = FlexiCubes("cpu")
    x_up, cubes_up = up.construct_voxel_grid(n)
    assert torch.allclose(x_up, torch.from_numpy(pts), atol=1e-5)   # same vertex order
    v_up, f_up, _ = up(x_up, torch.from_numpy(s), cubes_up, n)

    fc = SparseFlexiCubes("cpu")
    for mask in (np.ones(grid.dims, bool), sign_change_cubes(s.reshape(grid.vertex_dims))):
        sp = SparseGrid.from_cube_mask(grid, mask)
        v, f, _ = fc.extract(sp, torch.from_numpy(s)[sp.vert_flat])
        assert f.shape == f_up.shape and v.shape == v_up.shape
        # Same triangles as sets of positions (vertex numbering may differ)
        key = lambda vv, ff: np.sort(np.round(vv[ff].reshape(-1, 9).numpy(), 5), axis=0)  # noqa: E731
        assert np.allclose(key(v, f), key(v_up, f_up), atol=1e-5)


def test_extraction_of_an_analytic_sdf_is_close_and_outward():
    grid = make_grid([-1, -1, -1], [1, 1, 1], 40)
    s = _two_spheres(grid_points(grid))
    sp = SparseGrid.from_cube_mask(grid, sign_change_cubes(s.reshape(grid.vertex_dims)))
    v, f, _ = SparseFlexiCubes("cpu").extract(sp, torch.from_numpy(s)[sp.vert_flat])
    err = np.abs(_two_spheres(v.numpy()))
    assert err.max() < 0.4 * grid.h, err.max() / grid.h      # worst at the crease
    # Normals point to positive SDF (outside): compare with the analytic gradient
    c = v[f].mean(1).numpy()
    eps = 1e-3
    grad = np.stack([(_two_spheres(c + eps * e) - _two_spheres(c - eps * e)) for e in np.eye(3)], 1)
    agree = (face_normals(v, f).numpy() * grad).sum(1) > 0
    assert agree.mean() > 0.99


@needs_open3d
def test_mesh_sdf_round_trip_on_a_sphere():
    verts, faces = uv_sphere(48, 96, radius=0.7)
    grid = make_grid([-0.7] * 3, [0.7] * 3, 32)
    pts = grid_points(grid)
    s, info = mesh_sdf(verts.numpy(), faces.numpy(), pts)
    exact = np.linalg.norm(pts, axis=1) - 0.7
    assert np.abs(s - exact).max() < 0.01                 # faceting of the UV sphere
    assert info["ambiguous"] <= 1e-3 * len(pts)           # grazing rays at points on the surface
    assert ((s < 0) != (exact < 0))[np.abs(exact) > 0.02].sum() == 0
    sp = SparseGrid.from_cube_mask(grid, sign_change_cubes(s.reshape(grid.vertex_dims)))
    v, f, _ = SparseFlexiCubes("cpu").extract(sp, torch.from_numpy(s)[sp.vert_flat])
    r = v.norm(dim=1)
    assert float((r - 0.7).abs().max()) < 0.3 * grid.h


@needs_open3d
def test_mesh_sdf_sign_next_to_a_hole():
    # Sphere with its cap above z = 0.9 removed: a single ray direction gives a column of wrong
    # signs through the hole (open3d compute_signed_distance); the vote over 15 directions does not
    verts, faces = uv_sphere(48, 96, radius=1.0)
    keep = verts[faces].mean(1)[:, 2] < 0.9
    faces = faces[keep]
    ax = np.linspace(-1.2, 1.2, 41, dtype=np.float32)
    pts = np.stack(np.meshgrid(ax, ax, ax, indexing="ij"), -1).reshape(-1, 3)
    r = np.linalg.norm(pts, axis=1)
    far = np.abs(r - 1) > 0.1
    s, _ = mesh_sdf(verts.numpy(), faces.numpy(), pts)
    assert ((s < 0) != (r < 1))[far].sum() == 0


@needs_open3d
def test_fit_removes_a_protrusion_outside_the_silhouette():
    fit = pytest.importorskip("tools.bench.fit_flexicubes")
    fs = fit.fs
    # Input: a sphere plus a lump behind it that sticks out sideways (like the signature on 1464).
    # Target: the silhouette of the sphere alone. The lump must go, the rest must barely move.
    sphere_v, sphere_f = uv_sphere(32, 64, radius=0.6)
    lump_v, lump_f = uv_sphere(12, 24, radius=0.18)
    lump_v = lump_v + torch.tensor([0.62, 0.0, -0.35])
    verts = torch.cat([sphere_v, lump_v])
    faces = torch.cat([sphere_f, lump_f + sphere_v.shape[0]])
    hw = (96, 128)
    cam = fs.OrthoCamera("+z", 0.6, 0.62, 0.5, (0.0, 0.0, 0.0))
    target = fs.hard_silhouette(sphere_v, sphere_f, cam, hw, hw)
    alpha = target.astype(np.float32)
    sdf_full = fs.signed_distance(target)

    cfg = fit.FlexiConfig(res=40, levels=((128, 2, 25),), log_every=1000)
    grid, s0, _ = fit.initial_sdf(verts.numpy(), faces.numpy(), cfg, None, log=lambda *_: None)
    zones, info = fit.make_zones(grid, s0, cam, sdf_full, cfg)
    assert info["carved_inside_vertices"] > 0
    sparse = fit.build_sparse(grid, zones, "cpu")
    flat = sparse.vert_flat.numpy()
    s_ref = torch.from_numpy(zones.s_ref.reshape(-1)[flat])
    active = torch.from_numpy(np.nonzero(zones.active.reshape(-1)[flat])[0])
    s = fit.optimise(sparse, s_ref, active, alpha, sdf_full, cam, fs._zscale(verts), cfg, [],
                     log=lambda *_: None)
    v, f, _ = SparseFlexiCubes("cpu").extract(sparse, s)

    iou_in = fs.iou(fs.hard_silhouette(verts, faces, cam, hw, hw), target)
    iou_out = fs.iou(fs.hard_silhouette(v, f, cam, hw, hw), target)
    assert iou_in < 0.96 and iou_out > 0.98, (iou_in, iou_out)
    # The lump is gone: nothing left near its centre
    assert float((v - torch.tensor([0.75, 0.0, -0.35])).norm(dim=1).min()) > 0.1
    # The sphere is preserved (radius within a fraction of a voxel, except where the lump joined;
    # the contour itself may move a little towards the rasterised silhouette of the UV sphere)
    r = v.norm(dim=1)
    far_from_lump = (v - torch.tensor([0.62, 0.0, -0.35])).norm(dim=1) > 0.3
    err = (r[far_from_lump] - 0.6).abs()
    assert float(err.max()) < 0.75 * grid.h and float(err.mean()) < 0.15 * grid.h
