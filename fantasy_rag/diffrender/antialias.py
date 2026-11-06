"""Silhouette gradients by analytic edge antialiasing.

Re-implemented from the description of the antialiasing operation in Laine et al., "Modular
Primitives for High-Performance Differentiable Rendering" (SIGGRAPH Asia 2020), without reading
the nvdiffrast code. Principle:

- look at every pair of horizontally or vertically adjacent pixels whose triangle ids differ;
- decide which triangle's edge separates them (the occluder, or the only foreground triangle);
- intersect that edge with the segment joining the two pixel centres, at a fraction t in [0, 1]
  (0 = centre of the pixel showing that triangle, 1 = centre of the other pixel);
- blend: if t > 1/2 the triangle spills into the other pixel, which receives (t - 1/2) of its colour;
  if t < 1/2 the triangle covers less than its pixel, which receives (1/2 - t) of the other colour.

The blend is continuous in the vertex positions, so gradients reach the edge vertices through t.
Edges shared by the two triangles of the pair (interior mesh edges) are skipped.
Deltas from the four neighbours are summed (a diagonal edge is counted on both axes).
"""

import torch

from .rasterize import RasterOutput, pixel_centers_ndc, screen_barycentrics


def _pairs(tid_flat: torch.Tensor, batch: int, height: int, width: int):
    """Flat indices (with batch offset) of adjacent pixel pairs whose triangle ids differ.

    Returns (a, b, horizontal): b is the right (horizontal pair) or lower (vertical pair) pixel.
    """
    device = tid_flat.device
    grid = torch.arange(batch * height * width, device=device).view(batch, height, width)
    a_h, b_h = grid[:, :, :-1].reshape(-1), grid[:, :, 1:].reshape(-1)
    a_v, b_v = grid[:, :-1, :].reshape(-1), grid[:, 1:, :].reshape(-1)
    a = torch.cat([a_h, a_v])
    b = torch.cat([b_h, b_v])
    horizontal = torch.cat([torch.ones_like(a_h, dtype=torch.bool),
                            torch.zeros_like(a_v, dtype=torch.bool)])
    keep = tid_flat[a] != tid_flat[b]
    return a[keep], b[keep], horizontal[keep]


def antialias(color: torch.Tensor, rast: RasterOutput, vertices_clip: torch.Tensor,
              faces: torch.Tensor) -> torch.Tensor:
    """Antialias silhouette and occlusion edges of a rendered image.

    color: [B, H, W, C] shaded image (typically from `interpolate`), row 0 at the top.
    rast: the `RasterOutput` that produced it; vertices_clip / faces: the rasterized geometry.
    Returns a [B, H, W, C] image differentiable w.r.t. `color` and `vertices_clip`; this is where
    the silhouette gradients come from (e.g. render a mask of ones and compare it with a target).
    """
    batch, height, width, channels = color.shape
    if vertices_clip.dim() == 2:
        vertices_clip = vertices_clip.unsqueeze(0).expand(batch, -1, -1)
    faces = faces.to(device=color.device, dtype=torch.int64)
    hw = height * width
    tid = rast.triangle_id.reshape(-1)
    depth = rast.depth.detach().reshape(-1)
    flat_color = color.reshape(-1, channels)

    pa, pb, _ = _pairs(tid, batch, height, width)
    if pa.numel() == 0:
        return color
    ta, tb = tid[pa], tid[pb]
    fg_a, fg_b = ta >= 0, tb >= 0
    batch_idx = torch.div(pa, hw, rounding_mode="floor")

    dtype = vertices_clip.dtype
    xa, ya = pixel_centers_ndc(pa % hw, height, width, dtype)
    xb, yb = pixel_centers_ndc(pb % hw, height, width, dtype)

    def tri_xy(b_idx, t):
        tri = vertices_clip[b_idx.unsqueeze(1), faces[t.clamp(min=0)]]   # [N, 3, 4]
        return tri[..., :2] / tri[..., 3:4]

    # Which triangle owns the edge between the two centres (decided without gradients)
    with torch.no_grad():
        xy_a, xy_b = tri_xy(batch_idx, ta), tri_xy(batch_idx, tb)
        a_in_b = (screen_barycentrics(xy_b, xa, ya) >= 0).all(-1) & fg_b
        b_in_a = (screen_barycentrics(xy_a, xb, yb) >= 0).all(-1) & fg_a
        a_front = depth[pa] <= depth[pb]
        s_is_a = fg_a & (~fg_b | (a_in_b & ~b_in_a) | ((a_in_b == b_in_a) & a_front))

    s_pix = torch.where(s_is_a, pa, pb)
    o_pix = torch.where(s_is_a, pb, pa)
    s_tri = torch.where(s_is_a, ta, tb)
    o_tri = torch.where(s_is_a, tb, ta)
    xs, ys = torch.where(s_is_a, xa, xb), torch.where(s_is_a, ya, yb)
    xo, yo = torch.where(s_is_a, xb, xa), torch.where(s_is_a, yb, ya)

    # Crossing of the segment s -> o with the first edge of the S triangle (differentiable).
    # Screen barycentrics are linear along the segment, so b_k(t) = bs_k + t (bo_k - bs_k).
    xy_s = tri_xy(batch_idx, s_tri)
    bs = screen_barycentrics(xy_s, xs, ys)
    bo = screen_barycentrics(xy_s, xo, yo)
    leaving = bo < 0
    denom = torch.where(leaving, bs - bo, torch.ones_like(bs))
    t_k = torch.where(leaving, bs / denom, torch.full_like(bs, float("inf")))
    t, k = t_k.min(dim=-1)
    valid = torch.isfinite(t)

    # Skip interior edges: the other triangle shares both vertices of the crossed edge
    s_faces = faces[s_tri]
    e0 = s_faces.gather(1, ((k + 1) % 3).unsqueeze(1)).squeeze(1)
    e1 = s_faces.gather(1, ((k + 2) % 3).unsqueeze(1)).squeeze(1)
    o_faces = faces[o_tri.clamp(min=0)]
    shared = ((o_faces == e0.unsqueeze(1)).any(1) & (o_faces == e1.unsqueeze(1)).any(1)
              & (o_tri >= 0))
    valid &= ~shared
    if not bool(valid.any()):
        return color

    t = torch.where(valid, t, torch.full_like(t, 0.5)).clamp(0.0, 1.0)
    c_s, c_o = flat_color[s_pix], flat_color[o_pix]
    diff = (c_s - c_o) * valid.unsqueeze(1).to(flat_color.dtype)
    w_o = (t - 0.5).clamp(min=0.0).unsqueeze(1).to(flat_color.dtype)
    w_s = (0.5 - t).clamp(min=0.0).unsqueeze(1).to(flat_color.dtype)
    delta = torch.zeros_like(flat_color)
    delta = delta.index_add(0, o_pix, w_o * diff)
    delta = delta.index_add(0, s_pix, -w_s * diff)
    return (flat_color + delta).view(batch, height, width, channels)
