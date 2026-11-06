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
