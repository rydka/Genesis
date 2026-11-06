"""Triangle rasterization and attribute interpolation in pure PyTorch.

Two passes (written from general rasterization principles, see README for the sources):

1. Visibility, without gradients: every triangle is expanded into the pixel centres of its screen
   bounding box ("fragments"), tested with edge functions (Pineda 1988) and depth-tested with a single
   `scatter_reduce(amin)` on a 64-bit key (depth bits << 32 | triangle id). Triangles are processed in
   chunks whose fragment count stays under a budget, so memory does not depend on the face count.
2. Re-evaluation, with gradients: for each covered pixel, the perspective-correct barycentrics and the
   depth are recomputed with ordinary torch operations from the clip-space vertices of the winning
   triangle. Autograd then gives d(bary)/d(vertices) inside triangles.

Gradients of the coverage itself (silhouettes) are not produced here: see `antialias`.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch

_BG_KEY = torch.iinfo(torch.int64).max
_ID_MASK = 0xFFFFFFFF
# Triangles with w below this (behind or on the camera plane) are culled: there is no near clipping
_MIN_W = 1e-8

Resolution = Union[int, Tuple[int, int]]


@dataclass
class RasterOutput:
    """Per-pixel rasterization result, batch first, row 0 at the top of the image.

    triangle_id: [B, H, W] int64, -1 for background.
    bary:        [B, H, W, 3] perspective-correct barycentrics (0 on background), differentiable.
    depth:       [B, H, W] NDC depth z / w (0 on background), differentiable.
    """

    triangle_id: torch.Tensor
    bary: torch.Tensor
    depth: torch.Tensor

    @property
    def mask(self) -> torch.Tensor:
        return self.triangle_id >= 0

    @property
    def resolution(self) -> Tuple[int, int]:
        return tuple(self.triangle_id.shape[1:3])


def as_resolution(resolution: Resolution) -> Tuple[int, int]:
    if isinstance(resolution, int):
        return resolution, resolution
    h, w = resolution
    return int(h), int(w)


def pixel_centers_ndc(flat_index: torch.Tensor, height: int, width: int,
                      dtype=torch.float32) -> Tuple[torch.Tensor, torch.Tensor]:
    """NDC coordinates of the centres of pixels given by flat indices (i * W + j)."""
    i = torch.div(flat_index, width, rounding_mode="floor")
    j = flat_index - i * width
    x = (2.0 * (j.to(dtype) + 0.5)) / width - 1.0
    y = 1.0 - (2.0 * (i.to(dtype) + 0.5)) / height
    return x, y


def screen_barycentrics(xy: torch.Tensor, px: torch.Tensor, py: torch.Tensor) -> torch.Tensor:
    """2D (screen-space, not perspective-corrected) barycentrics of points (px, py) in triangles.

    xy: [N, 3, 2] NDC vertex positions; px, py: [N]. Returns [N, 3]. Differentiable.
    Coordinates are taken relative to vertex 0 to limit float cancellation on tiny triangles.
    """
    x0, y0 = xy[:, 0, 0], xy[:, 0, 1]
    e1x, e1y = xy[:, 1, 0] - x0, xy[:, 1, 1] - y0
    e2x, e2y = xy[:, 2, 0] - x0, xy[:, 2, 1] - y0
    qx, qy = px - x0, py - y0
    area = e1x * e2y - e2x * e1y
    b1 = (qx * e2y - e2x * qy) / area
    b2 = (e1x * qy - qx * e1y) / area
    return torch.stack([1.0 - b1 - b2, b1, b2], dim=-1)


def _face_setup(clip: torch.Tensor, faces: torch.Tensor, height: int, width: int,
                cull_backfaces: bool):
    """Per-face screen data for the visibility pass: edge-function coefficients and pixel bbox."""
    tri = clip[faces]                                   # [F, 3, 4]
    w = tri[..., 3]
    valid = (w > _MIN_W).all(dim=1)
    w_safe = torch.where(w > _MIN_W, w, torch.ones_like(w))
    ndc = tri[..., :3] / w_safe.unsqueeze(-1)
    x, y, z = ndc.unbind(-1)                            # [F, 3] each

    e1x, e1y = x[:, 1] - x[:, 0], y[:, 1] - y[:, 0]
    e2x, e2y = x[:, 2] - x[:, 0], y[:, 2] - y[:, 0]
    area = e1x * e2y - e2x * e1y
    valid &= area.abs() > 1e-20
    if cull_backfaces:
        valid &= area > 0                               # counter-clockwise in NDC = front
    valid &= ~((z < -1).all(1) | (z > 1).all(1))
    inv_area = torch.where(valid, 1.0 / torch.where(area == 0, torch.ones_like(area), area),
                           torch.zeros_like(area))

    # b1 = (qx e2y - e2x qy) / area, b2 = (e1x qy - qx e1y) / area with q = p - v0
    # depth z(p) = z0 + b1 (z1 - z0) + b2 (z2 - z0), linear in screen space
    coeffs = torch.stack([
        x[:, 0], y[:, 0],
        e2y * inv_area, -e2x * inv_area,
        -e1y * inv_area, e1x * inv_area,
        z[:, 0], z[:, 1] - z[:, 0], z[:, 2] - z[:, 0],
    ], dim=1)

    # Pixel bbox: column j covered if its centre 2 (j + 0.5) / W - 1 is in [xmin, xmax]
    xmin, xmax = x.min(1).values, x.max(1).values
    ymin, ymax = y.min(1).values, y.max(1).values
    j0 = torch.ceil((xmin + 1.0) * (width / 2.0) - 0.5).clamp(0, width)
    j1 = torch.floor((xmax + 1.0) * (width / 2.0) - 0.5).clamp(-1, width - 1)
    i0 = torch.ceil((1.0 - ymax) * (height / 2.0) - 0.5).clamp(0, height)
    i1 = torch.floor((1.0 - ymin) * (height / 2.0) - 0.5).clamp(-1, height - 1)
    j0, j1, i0, i1 = j0.long(), j1.long(), i0.long(), i1.long()
    nw = (j1 - j0 + 1).clamp(min=0)
    nh = (i1 - i0 + 1).clamp(min=0)
    count = torch.where(valid, nw * nh, torch.zeros_like(nw))
    return coeffs, j0, i0, nw, count


def _visibility(clip: torch.Tensor, faces: torch.Tensor, height: int, width: int,
                max_fragments: int, cull_backfaces: bool) -> torch.Tensor:
    """Closest triangle id per pixel, [H * W] int64 (-1 for background). No gradients."""
    device = clip.device
    coeffs, j0, i0, nw, count = _face_setup(clip, faces, height, width, cull_backfaces)
    zkey = torch.full((height * width,), _BG_KEY, dtype=torch.int64, device=device)

    face_ids = torch.nonzero(count > 0).squeeze(1)
    if face_ids.numel() == 0:
        return torch.full_like(zkey, -1)
    counts = count[face_ids]
    # Chunk boundaries so that each chunk has at most max_fragments fragments (or a single face)
    ends = torch.cumsum(counts, 0).cpu()
    chunks, start, offset = [], 0, 0
    while start < face_ids.numel():
        stop = int(torch.searchsorted(ends, torch.tensor(offset + max_fragments), right=True))
        stop = max(stop, start + 1)
        chunks.append((start, stop))
        offset = int(ends[stop - 1])
        start = stop

    for start, stop in chunks:
        f = face_ids[start:stop]
        c = counts[start:stop]
        local_face = torch.repeat_interleave(torch.arange(f.numel(), device=device), c)
        first = torch.cumsum(c, 0) - c
        local = torch.arange(local_face.numel(), device=device) - first[local_face]
        ff = f[local_face]
        fw = nw[ff]
        row = torch.div(local, fw, rounding_mode="floor")
        px_i = j0[ff] + (local - row * fw)
        py_i = i0[ff] + row
        pix = py_i * width + px_i
        del local, row, first

        k = coeffs[ff]                                  # [N, 9]
        cx = (2.0 * (px_i.to(k.dtype) + 0.5)) / width - 1.0
        cy = 1.0 - (2.0 * (py_i.to(k.dtype) + 0.5)) / height
        qx, qy = cx - k[:, 0], cy - k[:, 1]
        b1 = qx * k[:, 2] + qy * k[:, 3]
        b2 = qx * k[:, 4] + qy * k[:, 5]
        inside = (b1 >= 0) & (b2 >= 0) & (b1 + b2 <= 1)
        z = k[:, 6] + b1 * k[:, 7] + b2 * k[:, 8]
        inside &= (z >= -1) & (z <= 1)
        if not bool(inside.any()):
            continue
        d = ((z[inside] + 1.0) * 0.5).float().clamp(0.0, 1.0).contiguous()
        # Non-negative float32 bit patterns sort like the floats: one amin does the depth test
        key = (d.view(torch.int32).to(torch.int64) << 32) | ff[inside]
        zkey.scatter_reduce_(0, pix[inside], key, reduce="amin", include_self=True)

    tid = zkey & _ID_MASK
    return torch.where(zkey == _BG_KEY, torch.full_like(tid, -1), tid)


def _reevaluate(clip: torch.Tensor, faces: torch.Tensor, tid: torch.Tensor,
                height: int, width: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Differentiable perspective-correct barycentrics and NDC depth for visible pixels."""
    pix = torch.nonzero(tid >= 0).squeeze(1)
    bary = clip.new_zeros(height * width, 3)
    depth = clip.new_zeros(height * width)
    if pix.numel() == 0:
        return bary, depth
    tri = clip[faces[tid[pix]]]                         # [N, 3, 4]
    w = tri[..., 3]
    xy = tri[..., :2] / w.unsqueeze(-1)
    z = tri[..., 2] / w
    px, py = pixel_centers_ndc(pix, height, width, clip.dtype)
    b = screen_barycentrics(xy, px, py)
    q = b / w                                           # perspective correction
    lam = q / q.sum(-1, keepdim=True)
    bary = bary.index_put((pix,), lam)
    depth = depth.index_put((pix,), (b * z).sum(-1))
    return bary, depth


def rasterize(vertices_clip: torch.Tensor, faces: torch.Tensor, resolution: Resolution,
              max_fragments: int = 1 << 22, cull_backfaces: bool = False) -> RasterOutput:
    """Rasterize triangles given in clip space.

    vertices_clip: [V, 4] or [B, V, 4] (float32 or float64), e.g. `Camera.to_clip(vertices)`.
    faces: [F, 3] integer vertex indices, shared by the batch.
    resolution: H or (H, W).
    max_fragments: candidate (pixel, triangle) pairs evaluated at once; the main memory knob
        (about 100 bytes per fragment).
    cull_backfaces: drop triangles that are clockwise in NDC (OpenGL front = counter-clockwise).

    Triangles with a vertex behind the camera (w <= 0) are dropped (no near-plane clipping).
    Only `bary` and `depth` carry gradients (w.r.t. vertices_clip); see `antialias` for silhouettes.
    """
    height, width = as_resolution(resolution)
    if vertices_clip.dim() == 2:
        vertices_clip = vertices_clip.unsqueeze(0)
    faces = faces.to(device=vertices_clip.device, dtype=torch.int64)
    tids, barys, depths = [], [], []
    for clip in vertices_clip:
        with torch.no_grad():
            tid = _visibility(clip.detach(), faces, height, width, max_fragments, cull_backfaces)
        bary, depth = _reevaluate(clip, faces, tid, height, width)
        tids.append(tid.view(height, width))
        barys.append(bary.view(height, width, 3))
        depths.append(depth.view(height, width))
    return RasterOutput(torch.stack(tids), torch.stack(barys), torch.stack(depths))


def interpolate(attributes: torch.Tensor, rast: RasterOutput, faces: torch.Tensor) -> torch.Tensor:
    """Interpolate per-vertex attributes [V, C] or [B, V, C] over the image -> [B, H, W, C].

    Background pixels get 0. Differentiable w.r.t. the attributes and, through `rast.bary`, the
    vertex positions. For face-varying attributes (e.g. UV seams), pass corners as vertices:
    attributes [F * 3, C] with faces = arange(F * 3).view(F, 3).
    """
    tid = rast.triangle_id
    batch = tid.shape[0]
    faces = faces.to(device=tid.device, dtype=torch.int64)
    if attributes.dim() == 2:
        attributes = attributes.unsqueeze(0).expand(batch, -1, -1)
    mask = tid >= 0
    corners = faces[tid.clamp(min=0)]                   # [B, H, W, 3]
    b_idx = torch.arange(batch, device=tid.device).view(batch, 1, 1, 1)
    values = attributes[b_idx, corners]                 # [B, H, W, 3, C]
    out = (values * rast.bary.unsqueeze(-1).to(values.dtype)).sum(-2)
    return out * mask.unsqueeze(-1).to(out.dtype)


def face_attributes(values: torch.Tensor, rast: RasterOutput) -> torch.Tensor:
    """Per-face constant values [F, C] gathered per pixel -> [B, H, W, C] (0 on background)."""
    tid = rast.triangle_id
    out = values[tid.clamp(min=0)]
    return out * (tid >= 0).unsqueeze(-1).to(out.dtype)
