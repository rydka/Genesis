"""Continuous remeshing (vendored, torch_scatter replaced) and the remeshing finish of fit_flexicubes."""

import numpy as np
import pytest
import torch

from fantasy_rag.diffrender import uv_sphere
from fantasy_rag.diffrender.remeshing import MeshOptimizer, remesh


def test_mesh_optimizer_grows_a_sphere_and_keeps_it_regular():
    v, f = uv_sphere(16, 32)
    opt = MeshOptimizer(v.clone(), f.clone(), lr=0.3, edge_len_lims=(0.05, 0.3))
    v = opt.vertices
    for _ in range(100):
        opt.zero_grad()
        ((v.norm(dim=1) - 1.5) ** 2).mean().backward()
        opt.step()
        v, f = opt.remesh()
    r = v.detach().norm(dim=1)
    assert abs(float(r.mean()) - 1.5) < 0.05 and float(r.std()) < 0.02
    # closed manifold: every edge has two faces
    e = torch.cat([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]).sort(dim=1).values
    _, counts = torch.unique(e, dim=0, return_counts=True)
    assert bool((counts == 2).all())


def test_scatter_reduce_matches_the_torch_scatter_semantics():
    # collapse_edges ranks edges with a "max with the initial content" scatter (formerly
    # torch_scatter.scatter_max(out=...)): check against a loop
    torch.manual_seed(0)
    idx = torch.randint(0, 10, (50,))
    src = torch.randint(0, 100, (50,))
    out = torch.randint(0, 50, (10,))
    ref = out.clone()
    for i, s in zip(idx.tolist(), src.tolist()):
        ref[i] = max(ref[i], s)
    out.scatter_reduce_(0, idx, src, reduce="amax", include_self=True)
    assert torch.equal(out, ref)


def test_remesh_leaves_frozen_vertices_alone():
    v, f = uv_sphere(24, 48)
    n = v.shape[0]
    free = v[:, 0] > 0.3                                       # remesh only a cap of the sphere
    etc = torch.cat([v, free.float()[:, None]], 1)
    l = 0.05
    mn = torch.where(free, torch.full((n,), l), torch.zeros(n))
    mx = torch.where(free, torch.full((n,), 2 * l), torch.full((n,), float("inf")))
    etc2, f2 = remesh(etc, f.clone(), mn, mx, flip=True, flip_mask_column=3)
    # the frozen part keeps its vertices and faces (up to renumbering)
    frozen_old = {tuple(np.round(p, 6)) for p in v[~free].numpy()}
    new = {tuple(np.round(p, 6)) for p in etc2[:, :3].numpy()}
    assert frozen_old <= new
    assert f2.shape[0] != f.shape[0]                            # the free cap was remeshed


@pytest.mark.parametrize("weighted", [False, True])
def test_remesh_finish_moves_the_contour_in_plane_only(weighted):
    fit = pytest.importorskip("tools.bench.fit_flexicubes")
    fs = fit.fs
    verts, faces = uv_sphere(32, 64, radius=0.6)
    big_v, big_f = uv_sphere(32, 64, radius=0.66)
    hw = (96, 128)
    cam = fs.OrthoCamera("+z", 0.6, 0.62, 0.5, (0.0, 0.0, 0.0))
    target = fs.hard_silhouette(big_v, big_f, cam, hw, hw)
    alpha = target.astype(np.float32)
    sdf_full = fs.signed_distance(target)
    weight = None
    if weighted:   # only the right half may move
        weight = np.zeros(hw, np.float32)
        weight[:, 70:] = 1.0
    cfg = fit.RemeshConfig(levels=((128, 2, 40),), max_disp_px=4.0, log_every=1000)
    v, f, info = fit.remesh_finish(verts, faces, alpha, sdf_full, cam, fs._zscale(verts), cfg, weight,
                                   [], log=lambda *_: None)
    sil_in = fs.hard_silhouette(verts, faces, cam, hw, hw)
    sil_out = fs.hard_silhouette(v, f, cam, hw, hw)
    assert fs.iou(sil_out, target) > fs.iou(sil_in, target) + 0.02
    assert info["displacement_max_px"] <= 4.0 + 1e-3
    # the sphere grows by at most the cap
    q = fit.SurfaceQuery(verts.numpy(), faces.numpy())
    r = v.norm(dim=1)
    assert float(r.max()) < 0.6 + 4.0 / (0.6 * 96) + 1e-3
    if weighted:   # the left half did not move
        left = v[:, 0] < -0.2
        assert float(torch.from_numpy(q.distance(v[left].numpy())).max()) < 1e-4


def test_collapse_mask_keeps_frozen_vertices_in_place():
    # a collapse moves both end vertices to the middle of the edge: an edge between a free and a frozen
    # vertex must not collapse when the frozen part has to stay exact (contour band of the finish).
    # One free vertex with a long reference length: all its edges are free/frozen and short.
    v, f = uv_sphere(24, 48)
    n = v.shape[0]
    free = torch.zeros(n, dtype=torch.bool)
    free[v[:, 0].argmax()] = True
    mn = torch.where(free, torch.full((n,), 0.4), torch.zeros(n))
    mx = torch.where(free, torch.full((n,), 0.8), torch.full((n,), float("inf")))
    frozen_old = {tuple(np.round(p, 6)) for p in v[~free].numpy()}
    etc = torch.cat([v, free.float()[:, None]], 1)
    etc_a, f_a = remesh(etc.clone(), f.clone(), mn, mx, flip=True, flip_mask_column=3)
    assert f_a.shape[0] < f.shape[0]                            # without the mask: a collapse...
    assert frozen_old - {tuple(np.round(p, 6)) for p in etc_a[:, :3].numpy()}   # ...moved a frozen vertex
    etc_b, f_b = remesh(etc.clone(), f.clone(), mn, mx, flip=True, flip_mask_column=3,
                        collapse_mask_column=3)
    assert frozen_old <= {tuple(np.round(p, 6)) for p in etc_b[:, :3].numpy()}
    assert torch.equal(f_b, f)


def test_contour_band_weight():
    fit = pytest.importorskip("tools.bench.fit_flexicubes")
    sil = np.zeros((80, 80), bool)
    sil[10:70, 10:70] = True
    target = np.zeros_like(sil)
    target[10:70, 10:74] = True
    w = fit.contour_band_weight(sil, target, fit.ContourBand(8.0, 4.0))
    assert w[40, 10] == 1.0 and w[40, 72] == 1.0               # on either boundary
    assert w[40, 40] == 0.0                                    # inside, 30 px from both
    assert 0.0 < w[40, 16] < 1.0 and w[40, 20] == 0.0           # smooth fall-off, 0 from 8 px
    assert fit.contour_band_weight(sil, target, fit.ContourBand(0.0, 4.0)) is None


@pytest.mark.parametrize("finish", ["remesh", "vertex"])
def test_finish_in_contour_band_leaves_the_inside_exact(finish):
    fit = pytest.importorskip("tools.bench.fit_flexicubes")
    fs = fit.fs
    verts, faces = uv_sphere(32, 64, radius=0.6)
    big_v, big_f = uv_sphere(32, 64, radius=0.66)
    hw = (96, 128)
    cam = fs.OrthoCamera("+z", 0.6, 0.62, 0.5, (0.0, 0.0, 0.0))
    target = fs.hard_silhouette(big_v, big_f, cam, hw, hw)
    alpha = target.astype(np.float32)
    sdf_full = fs.signed_distance(target)
    sil_in = fs.hard_silhouette(verts, faces, cam, hw, hw)
    band = fit.contour_band_weight(sil_in, target, fit.ContourBand(6.0, 3.0))
    w = fit.vertex_weight(verts, cam, band)
    inside = (w <= 0).numpy()
    assert inside.sum() > 100 and (~inside).sum() > 100
    if finish == "remesh":
        cfg = fit.RemeshConfig(levels=((128, 2, 40),), max_disp_px=4.0, log_every=1000)
        v, f, _ = fit.remesh_finish(verts, faces, alpha, sdf_full, cam, fs._zscale(verts), cfg, None,
                                    [], log=lambda *_: None, band_img=band)
        assert f.shape[0] != faces.shape[0]                    # the band was remeshed
    else:
        cfg = fs.FitConfig(deform_levels=((128, 2, 40),), max_disp_px=4.0, log_every=1000)
        v = fs.deform(verts, faces, alpha, sdf_full, cam, cfg, [], log=lambda *_: None,
                      vertex_weight=w)
        f = faces
        assert torch.equal(v[torch.from_numpy(inside)], verts[torch.from_numpy(inside)])
    # the inside (front and back caps of the sphere) is bit-exact, the contour grew towards the target
    new = {tuple(p) for p in v.numpy().tolist()}
    assert all(tuple(p) in new for p in verts[torch.from_numpy(inside)].numpy().tolist())
    sil_out = fs.hard_silhouette(v, f, cam, hw, hw)
    assert fs.iou(sil_out, target) > fs.iou(sil_in, target) + 0.02
