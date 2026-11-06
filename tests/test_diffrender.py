"""Tests of the in-house differentiable rasterizer (CPU only, no GPU needed)."""

import math

import numpy as np
import pytest
import torch

from fantasy_rag.diffrender import (
    Camera,
    antialias,
    bake_face_attributes,
    bake_from_view,
    cube,
    edge_gradients,
    face_attributes,
    interpolate,
    look_at,
    perspective,
    perspective_from_intrinsics,
    rasterize,
    rasterize_uv,
    transform_points,
    uv_sphere,
    vertex_normals,
)
from fantasy_rag.diffrender import edgegrad, reference

F64 = torch.float64


def _centers(h, w):
    j, i = np.meshgrid(np.arange(w), np.arange(h))
    return 2 * (j + 0.5) / w - 1, 1 - 2 * (i + 0.5) / h


def _clip2d(xy, z=0.0, w=1.0, dtype=torch.float32):
    xy = torch.as_tensor(xy, dtype=dtype)
    n = xy.shape[0]
    return torch.cat([xy * w, torch.full((n, 1), z * w, dtype=dtype),
                      torch.full((n, 1), w, dtype=dtype)], dim=1)


# ---------------------------------------------------------------- coverage, barycentrics, z-buffer

class TestRasterize:
    def test_triangle_coverage_and_barycentrics_match_analytic(self):
        xy = np.array([[-0.71, -0.62], [0.83, -0.35], [-0.12, 0.77]])
        h, w = 37, 53
        rast = rasterize(_clip2d(xy, dtype=F64), torch.tensor([[0, 1, 2]]), (h, w))
        px, py = _centers(h, w)
        # Analytic barycentrics by solving the 2x2 system at every pixel centre
        a = np.array([[xy[1, 0] - xy[0, 0], xy[2, 0] - xy[0, 0]],
                      [xy[1, 1] - xy[0, 1], xy[2, 1] - xy[0, 1]]])
        q = np.stack([px - xy[0, 0], py - xy[0, 1]], -1)
        b12 = q @ np.linalg.inv(a).T
        bary = np.concatenate([1 - b12.sum(-1, keepdims=True), b12], -1)
        inside = (bary >= 0).all(-1)
        assert inside.sum() > 100
        np.testing.assert_array_equal(rast.mask[0].numpy(), inside)
        np.testing.assert_allclose(rast.bary[0].numpy()[inside], bary[inside], atol=1e-10)
        assert (rast.bary[0].numpy()[~inside] == 0).all()

    def test_perspective_correct_interpolation(self):
        # A tilted triangle in camera space; interpolated 3D points must lie on its plane and
        # project exactly onto the pixel centres.
        cam_pts = torch.tensor([[-1.0, -0.8, -2.0], [1.2, -0.5, -4.5], [0.1, 1.0, -3.0]], dtype=F64)
        proj = perspective(60.0, dtype=F64)
        clip = transform_points(cam_pts, proj)
        h = w = 24
        rast = rasterize(clip, torch.tensor([[0, 1, 2]]), (h, w))
        p = interpolate(cam_pts, rast, torch.tensor([[0, 1, 2]]))[0]
        mask = rast.mask[0]
        assert mask.sum() > 50
        normal = torch.linalg.cross(cam_pts[1] - cam_pts[0], cam_pts[2] - cam_pts[0])
        dist = ((p[mask] - cam_pts[0]) @ normal).abs().max()
        assert dist < 1e-10
        reproj = transform_points(p[mask], proj)
        ndc = reproj[:, :2] / reproj[:, 3:]
        px, py = _centers(h, w)
        np.testing.assert_allclose(ndc[:, 0].numpy(), px[mask.numpy()], atol=1e-10)
        np.testing.assert_allclose(ndc[:, 1].numpy(), py[mask.numpy()], atol=1e-10)
        # Depth output equals the NDC depth of the interpolated point
        z_ndc = reproj[:, 2] / reproj[:, 3]
        np.testing.assert_allclose(rast.depth[0][mask].numpy(), z_ndc.numpy(), atol=1e-10)

    @pytest.mark.parametrize("order", [[0, 1], [1, 0]])
    def test_zbuffer_keeps_nearest(self, order):
        near = _clip2d([[-0.9, -0.9], [0.5, -0.9], [-0.9, 0.5]], z=-0.2)
        far = _clip2d([[-0.5, -0.5], [0.9, -0.5], [0.9, 0.9]], z=0.4)
        verts = torch.cat([near, far])
        faces = torch.tensor([[0, 1, 2], [3, 4, 5]])[order]
        rast = rasterize(verts, faces, 32)
        near_id = order.index(0)
        far_id = order.index(1)
        px, py = _centers(32, 32)
        in_near = (px >= -0.9) & (py >= -0.9) & (px + py <= -0.4)
        in_far = (py >= -0.5) & (px <= 0.9) & (py <= px) & ~in_near
        tid = rast.triangle_id[0].numpy()
        assert (tid[in_near] == near_id).all()
        assert (tid[in_far] == far_id).all()
        np.testing.assert_allclose(rast.depth[0].numpy()[in_near], -0.2, atol=1e-6)

    def test_cube_front_view(self):
        v, f = cube(1.0)
        cam = Camera(look_at((0.0, 0.0, 3.0)), perspective(40.0), (64, 64))
        rast = rasterize(cam.to_clip(v), f, cam.resolution)
        # Only the +z face (faces 10, 11) is visible, as a square of half-size 0.5 / 2.5 * f
        half = 0.5 / 2.5 / math.tan(math.radians(20.0))
        px, py = _centers(64, 64)
        square = (np.abs(px) <= half) & (np.abs(py) <= half)
        tid = rast.triangle_id[0].numpy()
        np.testing.assert_array_equal(tid >= 0, square)
        assert set(np.unique(tid[square])) == {10, 11}
        # Depth of the front face: z_cam = -2.5
        z_ndc = (cam.projection[2, 2] * -2.5 + cam.projection[2, 3]) / 2.5
        np.testing.assert_allclose(rast.depth[0].numpy()[square], float(z_ndc), atol=1e-5)

    def test_backface_culling(self):
        v, f = cube(1.0)
        cam = Camera(look_at((0.0, 0.0, 3.0)), perspective(40.0), (32, 32))
        culled = rasterize(cam.to_clip(v), f, 32, cull_backfaces=True)
        reversed_faces = rasterize(cam.to_clip(v), f.flip(1), 32, cull_backfaces=True)
        full = rasterize(cam.to_clip(v), f, 32)
        assert torch.equal(culled.triangle_id, full.triangle_id)
        # Reversed winding: the front face is culled and the inside of the cube shows (back face and,
        # in perspective, the four sides)
        seen = set(reversed_faces.triangle_id.unique().tolist()) - {-1}
        assert {8, 9} <= seen and not seen & {10, 11}

    def test_chunking_does_not_change_the_result(self):
        v, f = uv_sphere(12, 24)
        cam = Camera(look_at((0.4, 0.3, 3.0)), perspective(45.0), (48, 40))
        clip = cam.to_clip(v)
        big = rasterize(clip, f, cam.resolution)
        small = rasterize(clip, f, cam.resolution, max_fragments=37)
        assert torch.equal(big.triangle_id, small.triangle_id)
        assert torch.allclose(big.bary, small.bary)

    def test_batch_and_behind_camera(self):
        tri = _clip2d([[-0.5, -0.5], [0.5, -0.5], [0.0, 0.5]])
        behind = tri.clone()
        behind[:, 3] = -1.0
        rast = rasterize(torch.stack([tri, behind]), torch.tensor([[0, 1, 2]]), 16)
        assert rast.triangle_id.shape == (2, 16, 16)
        assert rast.mask[0].any() and not rast.mask[1].any()

    def test_crop_matches_sub_window_of_full_render(self):
        v, f = uv_sphere(10, 20)
        cam = Camera(look_at((0.3, 0.5, 3.2)), perspective(50.0), (60, 80))
        full = rasterize(cam.to_clip(v), f, cam.resolution)
        box = (22, 14, 54, 38)                          # x0, y0, x1, y1 in pixels
        sub = cam.crop(box)
        assert sub.resolution == (24, 32)
        crop = rasterize(sub.to_clip(v), f, sub.resolution)
        x0, y0, x1, y1 = box
        assert torch.equal(crop.triangle_id[0], full.triangle_id[0, y0:y1, x0:x1])
        assert torch.allclose(crop.depth[0], full.depth[0, y0:y1, x0:x1], atol=1e-6)

    def test_intrinsics_match_symmetric_perspective(self):
        h, w = 30, 40
        fy = (h / 2) / math.tan(math.radians(25.0))
        a = perspective_from_intrinsics(fy, fy, w / 2, h / 2, w, h, 0.1, 100.0)
        b = perspective(50.0, aspect=w / h, near=0.1, far=100.0)
        assert torch.allclose(a, b, atol=1e-6)
        # An off-centre principal point shifts the image by whole pixels
        c = perspective_from_intrinsics(fy, fy, w / 2 + 3, h / 2 - 2, w, h, 0.1, 100.0)
        v, f = cube(0.8)
        view = look_at((0.0, 0.0, 3.0))
        ra = rasterize(transform_points(v, a @ view), f, (h, w)).triangle_id[0]
        rc = rasterize(transform_points(v, c @ view), f, (h, w)).triangle_id[0]
        # (coverage only: ids can swap on the diagonal shared by the two front triangles)
        assert torch.equal(rc[0:h - 2, 3:] >= 0, ra[2:, :w - 3] >= 0)
        assert (ra >= 0).sum() > 50


# ---------------------------------------------------------------- gradients

def _scene64():
    # Two overlapping triangles with generic positions (no pixel centre close to an edge)
    xy = torch.tensor([[-0.63, -0.71], [0.58, -0.43], [-0.07, 0.69],
                       [-0.21, -0.17], [0.81, 0.13], [0.33, 0.87]], dtype=F64)
    w = torch.tensor([1.3, 0.9, 1.1, 1.05, 1.2, 0.95], dtype=F64)
    z = torch.tensor([0.1, 0.2, -0.1, 0.4, 0.3, 0.5], dtype=F64)
    clip = torch.cat([xy * w[:, None], (z * w)[:, None], w[:, None]], 1)
    return clip, torch.tensor([[0, 1, 2], [3, 4, 5]])


class TestGradients:
    def test_gradcheck_barycentrics_and_depth(self):
        clip, faces = _scene64()

        def fn(c):
            r = rasterize(c, faces, 9)
            return r.bary, r.depth

        assert torch.autograd.gradcheck(fn, (clip.clone().requires_grad_(),), eps=1e-6, atol=1e-6)

    def test_gradcheck_interpolate(self):
        clip, faces = _scene64()
        attr = torch.randn(6, 2, dtype=F64, generator=torch.Generator().manual_seed(0))

        def fn(c, a):
            return interpolate(a, rasterize(c, faces, 9), faces)

        assert torch.autograd.gradcheck(
            fn, (clip.clone().requires_grad_(), attr.clone().requires_grad_()), eps=1e-6, atol=1e-6)

    def test_gradcheck_antialias(self):
        clip, faces = _scene64()
        colour = torch.tensor([[1.0, 0.2], [0.9, 0.1], [0.8, 0.3],
                               [0.1, 0.9], [0.2, 0.8], [0.3, 1.0]], dtype=F64)

        def fn(c):
            r = rasterize(c, faces, 11)
            return antialias(interpolate(colour, r, faces), r, c, faces)

        c = clip.clone().requires_grad_()
        out = fn(c)
        out.sum().backward()
        assert c.grad.abs().sum() > 0
        assert torch.autograd.gradcheck(fn, (clip.clone().requires_grad_(),), eps=1e-6, atol=1e-6)

    def test_antialias_gives_fractional_coverage(self):
        # A triangle much larger than the screen whose right edge is the vertical line x = e:
        # the antialiased mask of each row must sum to the exact covered width in pixels.
        w = 16
        e = 0.2 + 0.37 / w
        tri = _clip2d([[-3.0, -3.0], [e, -3.0], [e, 3.0]], dtype=F64)
        faces = torch.tensor([[0, 1, 2]])
        rast = rasterize(tri, faces, (4, w))
        rows = rast.mask.to(F64).unsqueeze(-1)
        aa = antialias(rows, rast, tri, faces)[0, ..., 0]
        # covered width of a row (centre band of the triangle, where the hypotenuse is off screen)
        # hypotenuse x = y, so the band x in [y, e] is only full for y <= -1; test the lowest row
        covered = (e + 1.0) * w / 2.0
        assert abs(float(aa[3].sum()) - covered) < 1e-9

    def test_silhouette_gradient_moves_edge_outwards(self):
        tri = _clip2d([[-3.0, -3.0], [0.11, -3.0], [0.11, 3.0]], dtype=F64).requires_grad_()
        faces = torch.tensor([[0, 1, 2]])
        rast = rasterize(tri, faces, 16)
        alpha = antialias(rast.mask.to(F64).unsqueeze(-1), rast, tri, faces)
        alpha.sum().backward()
        # Increasing coverage = moving the right edge (vertices 1 and 2) towards +x
        assert tri.grad[1, 0] > 0 and tri.grad[2, 0] > 0


# ---------------------------------------------------------------- image-based optimisation

def _silhouette(vertices, faces, cam, mode="analytic"):
    clip = cam.to_clip(vertices)
    rast = rasterize(clip, faces, cam.resolution)
    mask = rast.mask.to(clip.dtype).unsqueeze(-1)
    if mode == "analytic":
        alpha = antialias(mask, rast, clip, faces)
    else:
        alpha = edge_gradients(mask, rast, clip, faces)
    return alpha[..., 0]


@pytest.mark.parametrize("mode", ["analytic", "micro_edge"])
@pytest.mark.parametrize("shape", ["sphere", "cube"])
def test_recover_translation_from_silhouettes(shape, mode):
    torch.manual_seed(0)
    if shape == "sphere":
        v, f = uv_sphere(12, 24, radius=0.6)
    else:
        v, f = cube(1.0)
    cams = [Camera(look_at(eye), perspective(45.0), (48, 48))
            for eye in [(0.0, 0.0, 3.5), (3.5, 0.0, 0.0), (0.0, 3.5, 0.01)]]
    true_offset = torch.tensor([0.25, -0.15, 0.2])
    targets = [_silhouette(v + true_offset, f, c, mode).detach() for c in cams]

    offset = torch.zeros(3, requires_grad=True)
    opt = torch.optim.Adam([offset], lr=0.02)
    losses = []
    for _ in range(150):
        opt.zero_grad()
        loss = sum(((_silhouette(v + offset, f, c, mode) - t) ** 2).mean()
                   for c, t in zip(cams, targets))
        loss.backward()
        opt.step()
        losses.append(float(loss))
    assert losses[-1] < 0.05 * losses[0]
    assert (offset.detach() - true_offset).abs().max() < 0.03


@pytest.mark.parametrize("mode", ["analytic", "micro_edge"])
def test_vertex_level_optimisation_decreases_loss(mode):
    # Free vertices of a sphere pulled towards a larger sphere's silhouette
    v, f = uv_sphere(10, 20, radius=0.5)
    cams = [Camera(look_at(eye), perspective(45.0), (40, 40))
            for eye in [(0.0, 0.0, 3.0), (3.0, 0.0, 0.0), (0.0, 3.0, 0.01), (-2.1, 0.0, -2.1)]]
    targets = [_silhouette(v * 1.3, f, c, mode).detach() for c in cams]
    delta = torch.zeros_like(v, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=0.005)
    losses = []
    for _ in range(80):
        opt.zero_grad()
        loss = sum(((_silhouette(v + delta, f, c, mode) - t) ** 2).mean()
                   for c, t in zip(cams, targets))
        loss.backward()
        opt.step()
        losses.append(float(loss))
    assert losses[-1] < 0.3 * losses[0]
    radius = (v + delta).detach().norm(dim=1).mean()
    assert abs(float(radius) - 0.65) < 0.05


# ---------------------------------------------------------------- micro-edges

class TestMicroEdges:
    def test_forward_is_identity_and_image_gradient_passes_through(self):
        clip, faces = _scene64()
        clip = clip.clone().requires_grad_()
        rast = rasterize(clip, faces, 12)
        normals = interpolate(torch.randn(6, 3, dtype=F64), rast, faces).detach().requires_grad_()
        out = edge_gradients(normals, rast, clip, faces)
        assert torch.equal(out, normals)
        g = torch.randn_like(out)
        (out * g).sum().backward()
        assert torch.equal(normals.grad, g)
        assert clip.grad.abs().sum() > 0

    def test_batch_matches_single_views(self):
        v, f = uv_sphere(8, 16, radius=0.7)
        cams = [Camera(look_at(eye, dtype=F64), perspective(45.0, dtype=F64), (24, 24))
                for eye in [(0.0, 0.0, 3.0), (2.0, 0.5, 2.0)]]
        v = v.to(F64)
        grads = []
        for cam in cams:
            x = v.clone().requires_grad_()
            clip = cam.to_clip(x)
            rast = rasterize(clip, f, 24)
            edge_gradients(rast.mask.to(F64).unsqueeze(-1), rast, clip, f).sum().backward()
            grads.append(x.grad)
        x = v.clone().requires_grad_()
        clip = torch.stack([c.to_clip(x) for c in cams])
        rast = rasterize(clip, f, 24)
        edge_gradients(rast.mask.to(F64).unsqueeze(-1), rast, clip, f).sum().backward()
        assert grads[0].abs().sum() > 0
        assert torch.allclose(x.grad, grads[0] + grads[1])

    def test_sign_of_eq11_against_finite_differences(self, monkeypatch):
        # Settles the sign disagreement in Pidhorskyi et al.: eq. 11 has (I_A - I_B), the
        # supplementary eq. 26 has (I_B - I_A). Only eq. 11 matches the supersampled reference.
        scene = reference.scenes()["diagonal_edge"]
        ref = reference.reference_gradient(scene, supersample=64)
        assert reference.relative_error(reference.method_gradient(scene, "micro_edge"), ref) < 0.03
        monkeypatch.setattr(edgegrad, "_EQ11_SIGN", -1.0)
        flipped = reference.method_gradient(scene, "micro_edge")
        assert reference.relative_error(flipped, ref) > 1.9
        assert float((flipped * ref).sum()) < 0

    def test_intersection_matches_eq12(self):
        # Two screen-filling planes z = a x + c meeting along a vertical line; translating the
        # "varying" plane along its unit normal (x-z plane) moves the line by eq. 12 of the paper:
        # dp/dr = -n_f.z / (n_f^T [[0, 1], [-1, 0]] n_v), vectors written (x, z).
        a_f, c_f, a_v, c_v = 0.35, 0.05, -0.25, 0.1
        pts = [(-3.0, -3.0), (3.0, -3.0), (0.0, 3.0)]
        rows = ([[x, y, a_f * x + c_f, 1.0] for x, y in pts]
                + [[x, y, a_v * x + c_v, 1.0] for x, y in pts])
        clip = torch.tensor(rows, dtype=F64, requires_grad=True)
        faces = torch.tensor([[0, 1, 2], [3, 4, 5]])
        h = w = 16
        rast = rasterize(clip, faces, (h, w))
        values = torch.tensor([[1.0], [0.0]], dtype=F64)
        edge_gradients(face_attributes(values, rast), rast, clip, faces).sum().backward()
        n_f = torch.tensor([-a_f, 1.0], dtype=F64) / math.hypot(a_f, 1.0)
        n_v = torch.tensor([-a_v, 1.0], dtype=F64) / math.hypot(a_v, 1.0)
        rot = torch.tensor([[0.0, 1.0], [-1.0, 0.0]], dtype=F64)
        dp_dr = float(-n_f[1] / (n_f @ rot @ n_v))                # NDC per unit of translation
        # L = number of pixels showing face 0 (value 1). Face 0 is in front (smaller z) left of
        # the line, so L grows by H pixels per pixel of motion of the line towards +x.
        expected = h * dp_dr * (w / 2.0)
        direction = torch.zeros(6, 4, dtype=F64)
        direction[3:, 0], direction[3:, 2] = n_v[0], n_v[1]
        got = float((clip.grad * direction).sum())
        assert abs(expected) > 1.0
        assert got == pytest.approx(expected, rel=1e-6)

    def test_edge_scatter_is_independent_per_edge(self):
        # A blade thinner than a pixel: the micro-edge gradient of its width must not cancel
        # (it does with the paper's fragment scattering, by construction).
        x0 = torch.tensor(0.37, dtype=F64)
        width = torch.tensor(0.45, dtype=F64, requires_grad=True)
        grads = {}
        for scatter in ("edge", "fragment"):
            width.grad = None
            clip = reference.blade_vertices(x0, width, 32)
            rast = rasterize(clip, reference.BLADE_FACES, 32)
            mask = rast.mask.to(F64).unsqueeze(-1)
            edge_gradients(mask, rast, clip, reference.BLADE_FACES,
                           scatter=scatter).sum().backward()
            grads[scatter] = float(width.grad)
        assert grads["edge"] > 1.0                      # wider blade, more covered pixels
        assert abs(grads["fragment"]) < 1e-9


# ---------------------------------------------------------------- accuracy against a reference

@pytest.fixture(scope="module")
def references():
    return {name: reference.reference_gradient(scene, supersample=64)
            for name, scene in reference.scenes().items()}


# (case, mode, supersample, bound): relative error of the gradient below `bound`.
# Measured values are in the README table; bounds leave a margin.
_ACCURATE = [
    ("vertical_edge", "analytic", 1, 0.1), ("vertical_edge", "micro_edge", 1, 0.05),
    ("diagonal_edge", "micro_edge", 1, 0.03), ("occlusion", "micro_edge", 1, 0.25),
    ("intersection", "micro_edge", 1, 0.2),
    ("vertical_edge", "micro_edge", 4, 0.05), ("diagonal_edge", "micro_edge", 4, 0.03),
    ("thin_blade", "micro_edge", 4, 0.25), ("subpixel", "micro_edge", 4, 0.25),
    ("subpixel_no_centre", "micro_edge", 4, 0.25), ("occlusion", "micro_edge", 4, 0.12),
    ("intersection", "micro_edge", 4, 0.06),
]


@pytest.mark.parametrize("case,mode,supersample,bound", _ACCURATE)
def test_gradient_accuracy(references, case, mode, supersample, bound):
    scene = reference.scenes()[case]
    err = reference.relative_error(reference.method_gradient(scene, mode, supersample),
                                   references[case])
    assert err < bound


# Known failures, kept as tests so that they stay documented (README, "Justesse des gradients").
_FAILURES = [
    # analytic: the four-neighbour additive blend counts a diagonal edge on both axes (about 2x)
    ("diagonal_edge", "analytic", 1, 0.5),
    ("thin_blade", "analytic", 1, 1.0), ("subpixel", "analytic", 1, 1.0),
    ("occlusion", "analytic", 1, 1.0),
    # analytic: no depth gradient at all, intersection lines are not edges for it
    ("intersection", "analytic", 1, 1.0),
    # micro-edges at 1 sample per pixel: only rows whose centre is covered see the blade
    ("thin_blade", "micro_edge", 1, 0.25), ("subpixel", "micro_edge", 1, 0.5),
    # no pixel centre covered: the triangle is invisible, no gradient in either mode
    ("subpixel_no_centre", "analytic", 1, 0.999), ("subpixel_no_centre", "micro_edge", 1, 0.999),
]


@pytest.mark.parametrize("case,mode,supersample,above", _FAILURES)
def test_gradient_known_failures(references, case, mode, supersample, above):
    scene = reference.scenes()[case]
    err = reference.relative_error(reference.method_gradient(scene, mode, supersample),
                                   references[case])
    assert err > above


@pytest.mark.parametrize("mode,supersample,max_width_error", [
    ("micro_edge", 4, 0.02), ("analytic", 4, 0.1),
])
def test_recover_thin_blade(mode, supersample, max_width_error):
    # Blade 0.45 px wide, started 0.73 px off and 0.85 px too wide
    r = reference.blade_fit(mode, supersample)
    assert r["x0_error_px"] < 0.02
    assert r["width_error_px"] < max_width_error
    assert r["loss_last"] < 0.01 * r["loss_first"]


def test_thin_blade_collapses_without_supersampling():
    # Documented failure: the hard forward image of micro-edges against a soft target shrinks a
    # structure thinner than a pixel until it disappears; render at 4 samples per pixel instead.
    r = reference.blade_fit("micro_edge", 1)
    assert r["x0_error_px"] < 0.05
    assert r["width"] < 0.1


# ---------------------------------------------------------------- UV space

class TestUV:
    def test_face_constant_lands_on_the_right_texels(self):
        # Unit square split into 2 triangles along the diagonal, texture 8 x 8
        uv = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        uv_faces = torch.tensor([[0, 1, 2], [0, 2, 3]])
        values = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        tex, mask = bake_face_attributes(values, uv, uv_faces, 8)
        assert mask.all()
        # texel (i, j): u = (j + 0.5) / 8, v = 1 - (i + 0.5) / 8; face 0 is below the diagonal u > v
        j, i = np.meshgrid(np.arange(8), np.arange(8))
        u_c, v_c = (j + 0.5) / 8, 1 - (i + 0.5) / 8
        lower = u_c > v_c
        upper = u_c < v_c
        assert (tex[..., 0].numpy()[lower] == 1).all() and (tex[..., 1].numpy()[lower] == 0).all()
        assert (tex[..., 1].numpy()[upper] == 1).all() and (tex[..., 0].numpy()[upper] == 0).all()

    def test_uv_islands_and_empty_texels(self):
        uv = torch.tensor([[0.0, 0.0], [0.5, 0.0], [0.0, 0.5], [0.6, 0.6], [1.0, 0.6], [0.6, 1.0]])
        uv_faces = torch.tensor([[0, 1, 2], [3, 4, 5]])
        values = torch.tensor([[3.0], [7.0]])
        tex, mask = bake_face_attributes(values, uv, uv_faces, 10)
        # bottom-left triangle -> rows 5..9 (v < 0.5), top-right -> rows 0..3
        assert set(tex[mask][:, 0].tolist()) == {3.0, 7.0}
        assert (tex[5:, :5][mask[5:, :5]] == 3.0).all()
        assert (tex[:4, 6:][mask[:4, 6:]] == 7.0).all()
        assert not mask[5:, 6:].any() and not mask[:4, :5].any()

    def test_rasterize_uv_interpolates_world_positions(self):
        uv = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        uv_faces = torch.tensor([[0, 1, 2], [0, 2, 3]])
        pos = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 4.0, 0.0], [0.0, 4.0, 0.0]])
        rast = rasterize_uv(uv, uv_faces, 16)
        p = interpolate(pos, rast, uv_faces)[0]
        j, i = np.meshgrid(np.arange(16), np.arange(16))
        np.testing.assert_allclose(p[..., 0].numpy(), 2 * (j + 0.5) / 16, atol=1e-5)
        np.testing.assert_allclose(p[..., 1].numpy(), 4 * (1 - (i + 0.5) / 16), atol=1e-5)

    def test_bake_from_view_recovers_image(self):
        # A quad filling the view of an orthographic-like setup: texture == image
        h = w = 16
        img = torch.rand(h, w, 3, generator=torch.Generator().manual_seed(1))
        verts = torch.tensor([[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]])
        faces = torch.tensor([[0, 1, 2], [0, 2, 3]])
        uv = (verts[:, :2] + 1) / 2
        dist = 1.0 / math.tan(math.radians(20.0))
        mvp = perspective(40.0, near=0.5, far=10.0) @ look_at((0.0, 0.0, dist))
        normals = vertex_normals(verts, faces)
        tex, weight = bake_from_view(img, verts, faces, uv, faces, mvp, (h, w), normals=normals)
        assert (weight > 0.9).all()
        assert torch.allclose(tex, img, atol=1e-4)

    def test_bake_from_view_respects_occlusion(self):
        # Face-varying UVs: the back (-z, faces 8 and 9) and the front (+z, faces 10 and 11) each
        # fill one half of the texture, every other face is collapsed to a point (not rasterized).
        v, f = cube(1.0)
        corner_uv = {0: (0.0, 0.0), 2: (0.0, 1.0), 6: (0.5, 1.0), 4: (0.5, 0.0),
                     1: (0.5, 0.0), 5: (1.0, 0.0), 7: (1.0, 1.0), 3: (0.5, 1.0)}
        uv = torch.zeros(36, 2)
        for face in range(8, 12):
            for k in range(3):
                uv[face * 3 + k] = torch.tensor(corner_uv[int(f[face, k])])
        uv_faces = torch.arange(36).view(12, 3)
        mvp = perspective(40.0) @ look_at((0.0, 0.0, 3.0))
        img = torch.ones(32, 32, 3)
        tex, weight = bake_from_view(img, v, f, uv, uv_faces, mvp, 8)
        assert (weight[:, :4] == 0).all()               # back half: occluded
        assert (weight[:, 4:] == 1).all()               # front half: visible
        assert (tex[:, 4:] == 1).all()
