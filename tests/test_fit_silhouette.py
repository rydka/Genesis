"""Silhouette fitting tool (tools/bench/fit_silhouette.py) on synthetic data, CPU, no external files."""

import math

import numpy as np
import pytest
import torch

from fantasy_rag.diffrender import uv_sphere

fit = pytest.importorskip("tools.bench.fit_silhouette")


def _target_from(verts, faces, cam, hw):
    return fit.hard_silhouette(verts, faces, cam, hw, hw).astype(np.float32)


@pytest.fixture(scope="module")
def scene():
    torch.manual_seed(0)
    verts, faces = uv_sphere(24, 48, radius=1.0)
    hw = (96, 128)
    # Ground truth: an ellipsoid, seen from +Z, scaled and shifted, slightly rotated in the image
    target_verts = verts * torch.tensor([1.25, 0.85, 1.0])
    cam_true = fit.OrthoCamera("+z", 0.3, 0.62, 0.48, (0.0, 0.0, 0.1))
    target = _target_from(target_verts, faces, cam_true, hw)
    return verts, faces, target, cam_true


def test_rodrigues_is_a_rotation():
    r = fit.rodrigues(torch.tensor([0.3, -0.2, 0.5]))
    assert torch.allclose(r @ r.T, torch.eye(3), atol=1e-6)
    assert math.isclose(float(torch.det(r)), 1.0, rel_tol=1e-6)


def test_signed_distance_sign():
    b = np.zeros((20, 20), bool)
    b[5:15, 5:15] = True
    d = fit.signed_distance(b)
    assert d[10, 10] < 0 and d[0, 0] > 0 and d[5, 5] < 0 and d[4, 10] > 0


def test_camera_recovers_pose_and_orientation():
    # Asymmetric ellipsoid: the front orientation (+z or its mirror -z) must win over the sides
    verts, faces = uv_sphere(24, 48, radius=1.0)
    verts = verts * torch.tensor([1.3, 0.8, 0.5])
    hw = (96, 128)
    cam_true = fit.OrthoCamera("+z", 0.3, 0.55, 0.5, (0.05, -0.05, 0.12))
    target = _target_from(verts, faces, cam_true, hw)
    sdf = fit.signed_distance(target > 0.5)
    cfg = fit.FitConfig(camera_levels=((64, 1, 40), (128, 1, 60)), camera_probe_iters=15)
    iou0 = fit.iou(fit.hard_silhouette(verts, faces, fit.init_camera(verts, "+z", target), hw, hw),
                   target > 0.5)
    cam, info = fit.fit_camera(verts, faces, target, sdf, cfg, log=lambda *_: None)
    assert info["orientation"] in ("+z", "-z")
    iou_cam = fit.iou(fit.hard_silhouette(verts, faces, cam, hw, hw), target > 0.5)
    assert iou_cam > max(iou0, 0.97), (iou0, iou_cam)


def test_deformation_fits_a_different_shape(scene):
    verts, faces, target, cam_true = scene
    hw = target.shape
    sdf = fit.signed_distance(target > 0.5)
    # Coarse sphere (1k vertices) with the default Laplacian weight: normalised by the mean edge
    # length, the same weight works here and on the 0.4 to 1.3 M face meshes
    cfg = fit.FitConfig(deform_levels=((64, 2, 40), (128, 2, 60)), log_every=1000, smooth_iters=8,
                        w_flip=1.0)
    # Camera fixed (true pose); the sphere has to become the ellipsoid's silhouette
    cam = cam_true
    iou_cam = fit.iou(fit.hard_silhouette(verts, faces, cam, hw, hw), target > 0.5)
    new_verts = fit.deform(verts, faces, target, sdf, cam, cfg, log=lambda *_: None)
    iou_after = fit.iou(fit.hard_silhouette(new_verts, faces, cam, hw, hw), target > 0.5)
    assert iou_cam < 0.9
    assert iou_after > 0.97, (iou_cam, iou_after)

    # Depth along the view axis is unchanged by construction
    rot = fit.CameraParams(cam).rot().detach()
    assert torch.allclose((new_verts @ rot.T)[:, 2], (verts @ rot.T)[:, 2], atol=1e-5)
    flipped, _ = fit.flipped_faces(verts, new_verts, faces)
    assert flipped < 0.02 * faces.shape[0]


def test_displacement_cap(scene):
    verts, faces, target, cam = scene
    hw = target.shape
    sdf = fit.signed_distance(target > 0.5)
    cfg = fit.FitConfig(deform_levels=((64, 2, 30), (128, 2, 30)), log_every=1000, smooth_iters=8,
                        max_disp_px=2.0)
    new_verts = fit.deform(verts, faces, target, sdf, cam, cfg, log=lambda *_: None)
    disp_px = (new_verts - verts).norm(dim=1) * cam.scale * hw[0]
    assert float(disp_px.max()) <= 2.0 + 1e-4
    iou_cam = fit.iou(fit.hard_silhouette(verts, faces, cam, hw, hw), target > 0.5)
    iou_after = fit.iou(fit.hard_silhouette(new_verts, faces, cam, hw, hw), target > 0.5)
    assert iou_after > iou_cam                    # still a correction, only a bounded one


def test_mean_edge_length_and_flipped_faces():
    v = torch.tensor([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]])
    f = torch.tensor([[0, 1, 2]])
    assert abs(float(fit.mean_edge_length(v, f)) - (2 + 2 ** 0.5) / 3) < 1e-6
    mirrored = v * torch.tensor([1.0, -1.0, 1.0])
    assert fit.flipped_faces(v, mirrored, f) == (1, 1.0)
    assert fit.flipped_faces(v, v * 2, f)[0] == 0
