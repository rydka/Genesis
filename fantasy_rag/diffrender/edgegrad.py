"""Silhouette and occlusion gradients by micro-edges, with an unchanged forward image.

Re-implemented from the paper (no code copied): S. Pidhorskyi, T. Simon, G. Schwartz, H. Wen,
Y. Sheikh, J. Saragih, "Rasterized Edge Gradients: Handling Discontinuities Differentiably",
ECCV 2024, arXiv 2405.02508 (sections 4.1 to 4.4 and supplementary material).

Principle:
- the forward pass is the identity: the image is returned as rendered (hard edges), so normal or
  depth maps are never blended across two surfaces;
- every pair of horizontally or vertically adjacent pixels A, B whose triangle ids differ is seen
  as a "micro-edge" lying exactly between the two pixels, perpendicular to the pair;
- moving that micro-edge by one pixel towards B turns B into A, so with averaged one-sided
  derivatives (eq. 10) the gradient of the loss w.r.t. its position p (in pixels, from A to B) is

      dL/dp = 1/2 (dL/dI_A + dL/dI_B) (I_A - I_B)                                       (eq. 11)

  The supplementary material (eq. 26) writes (I_B - I_A). The sign of eq. 11 is the right one: it
  follows from dI_B/dp = I_A - I_B, and `tests/test_diffrender.py` checks it against
  finite differences of a supersampled coverage (the opposite sign gives the opposite gradient);
- the pair is classified from triangle ids and "centre inside the other triangle" tests
  (section 4.3): background (one id is -1) and overhang (exactly one centre inside the other
  triangle) send the gradient to the foreground triangle only; adjacent triangles (no centre
  inside the other one) get no gradient; self-intersection (both centres inside) moves the
  intersection line of the two triangles (section 4.4, eq. 12).

From edge position to vertices (our formulation; the paper goes through the pixel fragment):
the micro-edge follows the component, along the pair axis, of the motion of the actual edge at the
point where it crosses the segment between the two pixel centres. We compute that crossing t
(fraction from the owner pixel to the other one) differentiably, with the coordinate
perpendicular to the pair detached, and let autograd take dL/dt = dL/dp to the vertices:
- silhouette / overhang: t is where the segment leaves the foreground triangle, so only the two
  vertices of the crossed edge move it, with weights (1 - s, s) along the edge;
- self-intersection: t is where the depths of the two triangles are equal; for a translation of
  one triangle along its normal its derivative is eq. 12 (checked by a test). A depth change
  moves the intersection line along its screen normal n, and the crossing on a horizontal
  (vertical) segment then moves by delta / |n_x| (|n_y|): summed over both axes of pairs, that
  counts the motion twice for a slanted line (measured: 2x at 45 degrees, exact only for an
  axis-aligned line; eq. 12 applied per pair has the same property, as we read it). We weight
  the depth derivative of a horizontal (vertical) pair by n_x^2 (n_y^2), so that the two axes
  sum to the exact flux of the motion through the line.
The paper's own scattering (dp/dr = 1 for the foreground fragment at the pixel centre, then the
barycentric weights of that fragment) is available with scatter="fragment" for comparison. With
it, a triangle thinner than a pixel cannot change its width: its two silhouette edges share the
same fragment, so their gradients cancel (see README, error table).
"""

import torch

from .antialias import _pairs
from .rasterize import RasterOutput, pixel_centers_ndc, screen_barycentrics

# Sign of (I_A - I_B) in eq. 11; -1 would be the supplementary eq. 26. Kept as a module constant
# only so that the test settling the sign can flip it.
_EQ11_SIGN = 1.0
_MIN_W = 1e-8


def _gather_tri(ndc, batch_idx, faces, tri_ids):
    return ndc[batch_idx.unsqueeze(1), faces[tri_ids.clamp(min=0)]]       # [N, 3, 3]


def _axis_xy(tri, horizontal):
    """Screen xy of triangle vertices with the coordinate perpendicular to the pair detached."""
    x, y = tri[..., 0], tri[..., 1]
    h = horizontal.unsqueeze(1)
    return torch.stack([torch.where(h, x, x.detach()), torch.where(h, y.detach(), y)], dim=-1)


def _edge_vertex_grad(image, grad_out, ndc, tid_img, faces, scatter):
    batch, height, width, _ = image.shape
    hw = height * width
    tid = tid_img.reshape(-1)
    flat_img = image.reshape(-1, image.shape[-1])
    flat_g = grad_out.reshape(-1, grad_out.shape[-1])
    pa, pb, horizontal = _pairs(tid, batch, height, width)
    grad_ndc = torch.zeros_like(ndc)
    if pa.numel() == 0:
        return grad_ndc
    ta, tb = tid[pa], tid[pb]
    fg_a, fg_b = ta >= 0, tb >= 0
    batch_idx = torch.div(pa, hw, rounding_mode="floor")
    dtype = ndc.dtype
    xa, ya = pixel_centers_ndc(pa % hw, height, width, dtype)
    xb, yb = pixel_centers_ndc(pb % hw, height, width, dtype)

    # Classification (section 4.3), without gradients
    with torch.no_grad():
        tri_a = _gather_tri(ndc, batch_idx, faces, ta)
        tri_b = _gather_tri(ndc, batch_idx, faces, tb)
        a_in_b = (screen_barycentrics(tri_b[..., :2], xa, ya) >= 0).all(-1) & fg_b
        b_in_a = (screen_barycentrics(tri_a[..., :2], xb, yb) >= 0).all(-1) & fg_a
        background = fg_a ^ fg_b
        overhang = fg_a & fg_b & (a_in_b ^ b_in_a)
        intersect = fg_a & fg_b & a_in_b & b_in_a
        # Owner S of the edge: the only foreground triangle, or the one covering the other's centre
        s_is_a = torch.where(background, fg_a, a_in_b)
        silhouette = background | overhang

    # dL/dp per pair (eq. 11), p from the owner pixel S towards the other pixel O
    s_pix = torch.where(s_is_a, pa, pb)
    o_pix = torch.where(s_is_a, pb, pa)
    dl_dp = 0.5 * _EQ11_SIGN * ((flat_g[s_pix] + flat_g[o_pix])
                                * (flat_img[s_pix] - flat_img[o_pix])).sum(-1)
    dl_dp = dl_dp.to(dtype)

    leaf = ndc.detach().requires_grad_()
    outputs, weights = [], []
    with torch.enable_grad():
        sel = torch.nonzero(silhouette & (dl_dp != 0)).squeeze(1)
        if sel.numel():
            b_i, hz = batch_idx[sel], horizontal[sel]
            sa = s_is_a[sel]
            xs, ys = torch.where(sa, xa[sel], xb[sel]), torch.where(sa, ya[sel], yb[sel])
            xo, yo = torch.where(sa, xb[sel], xa[sel]), torch.where(sa, yb[sel], ya[sel])
            s_tri = torch.where(sa, ta[sel], tb[sel])
            xy = _axis_xy(_gather_tri(leaf, b_i, faces, s_tri), hz)
            bs = screen_barycentrics(xy, xs, ys)
            if scatter == "fragment":
                # Paper: the fragment at the centre of S moves the edge rigidly (dp/dr = 1)
                u_s = torch.where(hz, xs, ys)
                u_o = torch.where(hz, xo, yo)
                coord = torch.where(hz.unsqueeze(1), xy[..., 0], xy[..., 1])
                t = (bs.detach() * coord).sum(-1) / (u_o - u_s)
                ok = torch.ones_like(hz)
            else:
                bo = screen_barycentrics(xy, xo, yo)
                with torch.no_grad():
                    leaving = bo < 0
                    t_k = torch.where(leaving, bs / torch.where(leaving, bs - bo, 1.0),
                                      torch.full_like(bs, float("inf")))
                    k = t_k.argmin(-1, keepdim=True)
                    ok = torch.isfinite(t_k.gather(1, k).squeeze(1))
                bs_k, bo_k = bs.gather(1, k).squeeze(1), bo.gather(1, k).squeeze(1)
                t = bs_k / torch.where(ok, bs_k - bo_k, torch.ones_like(bs_k))
            outputs.append(t[ok])
            weights.append(dl_dp[sel][ok])

        sel = torch.nonzero(intersect & (dl_dp != 0)).squeeze(1)
        if sel.numel():
            # Crossing of the two depth planes along the segment A -> B (owner S = A)
            b_i, hz = batch_idx[sel], horizontal[sel]
            tri_a = _gather_tri(leaf, b_i, faces, ta[sel])
            tri_b = _gather_tri(leaf, b_i, faces, tb[sel])
            px_a, py_a, px_b, py_b = xa[sel], ya[sel], xb[sel], yb[sel]

            def gap(xy_a, z_a, xy_b, z_b, px, py):
                """Depth of triangle A minus depth of triangle B at screen points (affine)."""
                return ((screen_barycentrics(xy_a, px, py) * z_a).sum(-1)
                        - (screen_barycentrics(xy_b, px, py) * z_b).sum(-1))

            # A depth change moves the intersection line along its screen normal n. The crossing
            # on a horizontal (vertical) segment moves by delta / |n_x| (|n_y|): counting it in
            # full on both axes would count the motion twice for a slanted line. Weighting the
            # depth derivative by n_x^2 (n_y^2) makes the two axes sum to the exact flux.
            with torch.no_grad():
                args = (tri_a[..., :2], tri_a[..., 2], tri_b[..., :2], tri_b[..., 2])
                g0 = gap(*args, px_a, py_a)
                gx = (gap(*args, px_a + 2.0 / width, py_a) - g0) * (width / 2.0)
                gy = (gap(*args, px_a, py_a + 2.0 / height) - g0) * (height / 2.0)
                n2 = torch.where(hz, gx * gx, gy * gy) / (gx * gx + gy * gy).clamp(min=1e-30)

            def axis_z(tri):
                z = tri[..., 2]
                return z.detach() + n2.unsqueeze(1) * (z - z.detach())

            xy_a, xy_b = _axis_xy(tri_a, hz), _axis_xy(tri_b, hz)
            z_a, z_b = axis_z(tri_a), axis_z(tri_b)
            d_a = gap(xy_a, z_a, xy_b, z_b, px_a, py_a)
            d_b = gap(xy_a, z_a, xy_b, z_b, px_b, py_b)
            denom = d_a - d_b
            ok = (d_a.detach() <= 0) & (d_b.detach() >= 0) & (denom.detach() != 0)
            t = d_a / torch.where(ok, denom, torch.ones_like(denom))
            outputs.append(t[ok])
            weights.append(dl_dp[sel][ok])

        if outputs:
            t_all = torch.cat(outputs)
            if t_all.numel() and t_all.requires_grad:
                (grad_ndc,) = torch.autograd.grad(t_all, leaf, torch.cat(weights),
                                                  allow_unused=True)
                if grad_ndc is None:
                    grad_ndc = torch.zeros_like(ndc)
    return grad_ndc


class _EdgeGrad(torch.autograd.Function):
    """Identity on the image; backward adds the micro-edge gradients w.r.t. NDC positions."""

    @staticmethod
    def forward(ctx, image, ndc, triangle_id, faces, scatter):
        ctx.save_for_backward(image, ndc, triangle_id, faces)
        ctx.scatter = scatter
        return image.clone()

    @staticmethod
    def backward(ctx, grad_out):
        image, ndc, triangle_id, faces = ctx.saved_tensors
        grad_ndc = None
        if ctx.needs_input_grad[1]:
            grad_ndc = _edge_vertex_grad(image.detach(), grad_out.detach(), ndc.detach(),
                                         triangle_id, faces, ctx.scatter)
        grad_image = grad_out if ctx.needs_input_grad[0] else None
        return grad_image, grad_ndc, None, None, None


def edge_gradients(image: torch.Tensor, rast: RasterOutput, vertices_clip: torch.Tensor,
                   faces: torch.Tensor, scatter: str = "edge") -> torch.Tensor:
    """Pass a rendered image through unchanged, adding silhouette and occlusion gradients.

    image: [B, H, W, C] rendered image (mask, colours, normals, depth...), row 0 at the top.
    rast: the `RasterOutput` that produced it; vertices_clip [V, 4] or [B, V, 4] / faces: the
        rasterized geometry.
    scatter: "edge" (default) sends each micro-edge gradient to the vertices of the edge it lies
        on; "fragment" follows the paper literally (pixel fragment, barycentric weights).
    Returns an image equal to `image`. Gradients w.r.t. `image` pass through unchanged; gradients
    w.r.t. `vertices_clip` come from the micro-edges between pixels of different triangles.
    """
    if scatter not in ("edge", "fragment"):
        raise ValueError(f"unknown scatter {scatter!r}")
    batch = image.shape[0]
    if vertices_clip.dim() == 2:
        vertices_clip = vertices_clip.unsqueeze(0).expand(batch, -1, -1)
    w = vertices_clip[..., 3:4]
    w = torch.where(w > _MIN_W, w, torch.ones_like(w))
    ndc = vertices_clip[..., :3] / w
    faces = faces.to(device=image.device, dtype=torch.int64)
    return _EdgeGrad.apply(image, ndc, rast.triangle_id, faces, scatter)
