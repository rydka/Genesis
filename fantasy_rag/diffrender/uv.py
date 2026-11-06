"""Rasterization in UV space, for texture baking.

Texture convention (OpenGL / OBJ): u goes left to right, v goes bottom to top, so texel (i, j) of a
texture image stored row 0 first has its centre at u = (j + 0.5) / W, v = 1 - (i + 0.5) / H.
For glTF (v pointing down), pass `1 - v`.
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from .camera import transform_points
from .rasterize import RasterOutput, Resolution, interpolate, rasterize


def uv_to_clip(uv: torch.Tensor) -> torch.Tensor:
    """UV coordinates [T, 2] in [0, 1] to clip space [T, 4] (orthographic, w = 1)."""
    xy = uv * 2.0 - 1.0
    zeros = torch.zeros_like(uv[..., :1])
    return torch.cat([xy, zeros, torch.ones_like(zeros)], dim=-1)


def rasterize_uv(uv: torch.Tensor, uv_faces: torch.Tensor, resolution: Resolution,
                 max_fragments: int = 1 << 22) -> RasterOutput:
    """Rasterize the UV layout: per texel, the face id and its barycentrics.

    Overlapping UV islands resolve to the smallest face id (all depths are equal). Use the result with
    `interpolate(attributes, rast, faces)` where `faces` are the geometric faces matching `uv_faces`.
    """
    return rasterize(uv_to_clip(uv), uv_faces, resolution, max_fragments=max_fragments)


def bake_vertex_attributes(attributes: torch.Tensor, faces: torch.Tensor, uv: torch.Tensor,
                           uv_faces: torch.Tensor, resolution: Resolution
                           ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Bake per-vertex attributes [V, C] into a texture -> (texture [H, W, C], mask [H, W])."""
    rast = rasterize_uv(uv, uv_faces, resolution)
    return interpolate(attributes, rast, faces)[0], rast.mask[0]


def bake_face_attributes(values: torch.Tensor, uv: torch.Tensor, uv_faces: torch.Tensor,
                         resolution: Resolution) -> Tuple[torch.Tensor, torch.Tensor]:
    """Bake per-face constants [F, C] into a texture -> (texture [H, W, C], mask [H, W])."""
    rast = rasterize_uv(uv, uv_faces, resolution)
    tid = rast.triangle_id[0]
    tex = values[tid.clamp(min=0)] * (tid >= 0).unsqueeze(-1).to(values.dtype)
    return tex, tid >= 0


def bake_from_view(image: torch.Tensor, vertices: torch.Tensor, faces: torch.Tensor,
                   uv: torch.Tensor, uv_faces: torch.Tensor, mvp: torch.Tensor,
                   texture_resolution: Resolution, depth_tolerance: float = 0.01,
                   normals: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Project an image seen by a camera onto the texture of a mesh.

    image: [H, W, C] (row 0 at the top); vertices: [V, 3] world space; mvp: 4x4 world-to-clip.
    A texel receives a colour if its surface point projects inside the image and is not occluded
    (its camera depth w is within `depth_tolerance` (relative) of the depth buffer).
    normals: optional per-vertex normals [V, 3]; the weight is then cos(view, normal) clamped at 0,
    so that several views can be blended by weight.
    Returns (texture [Ht, Wt, C], weight [Ht, Wt]); weight is 0 where nothing was projected.
    """
    img_h, img_w = image.shape[:2]
    clip = transform_points(vertices, mvp.to(vertices.dtype))
    with torch.no_grad():
        cam = rasterize(clip, faces, (img_h, img_w))
        w_buffer = interpolate(clip[:, 3:4], cam, faces)[0, ..., 0]    # camera depth per pixel

    tex_rast = rasterize_uv(uv, uv_faces, texture_resolution)
    texel_pos = interpolate(vertices, tex_rast, faces)[0]              # [Ht, Wt, 3]
    texel_mask = tex_rast.mask[0]
    texel_clip = transform_points(texel_pos, mvp.to(texel_pos.dtype))
    w = texel_clip[..., 3]
    w_safe = torch.where(w > 1e-8, w, torch.ones_like(w))
    ndc = texel_clip[..., :2] / w_safe.unsqueeze(-1)
    in_view = texel_mask & (w > 1e-8) & (ndc.abs() <= 1).all(-1)

    # grid_sample: grid y = -1 is the first row, i.e. the top of the image, where NDC y = +1
    grid = torch.stack([ndc[..., 0], -ndc[..., 1]], dim=-1).unsqueeze(0).to(image.dtype)
    src = image.permute(2, 0, 1).unsqueeze(0)
    colour = F.grid_sample(src, grid, mode="bilinear", align_corners=False)[0].permute(1, 2, 0)
    w_seen = F.grid_sample(w_buffer[None, None].to(image.dtype), grid, mode="nearest",
                           align_corners=False)[0, 0]
    visible = in_view & (w_seen > 0) & ((w.to(image.dtype) - w_seen).abs()
                                         <= depth_tolerance * w_seen)

    weight = visible.to(image.dtype)
    if normals is not None:
        texel_n = interpolate(normals, tex_rast, faces)[0]
        texel_n = F.normalize(texel_n, dim=-1)
        # Camera centre: the only point with clip x = y = w = 0
        m = mvp.to(torch.float64)
        a = torch.stack([m[0, :3], m[1, :3], m[3, :3]])
        rhs = -torch.stack([m[0, 3], m[1, 3], m[3, 3]])
        eye = torch.linalg.solve(a, rhs).to(texel_pos.dtype)
        view_dir = F.normalize(eye - texel_pos, dim=-1)
        weight = weight * (view_dir * texel_n).sum(-1).clamp(min=0.0).to(image.dtype)
    texture = colour * (weight > 0).unsqueeze(-1).to(colour.dtype)
    return texture, weight
