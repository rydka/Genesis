"""Small differentiable mesh helpers used with the rasterizer (normals, test primitives)."""

import math
from typing import Tuple

import torch
import torch.nn.functional as F


def face_normals(vertices: torch.Tensor, faces: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """Face normals [F, 3] (counter-clockwise winding). Unnormalised length = 2 x area."""
    v = vertices[faces.long()]
    n = torch.linalg.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    return F.normalize(n, dim=-1) if normalize else n


def vertex_normals(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Area-weighted vertex normals [V, 3]. Differentiable."""
    fn = face_normals(vertices, faces, normalize=False)
    faces = faces.long()
    n = torch.zeros_like(vertices)
    for k in range(3):
        n = n.index_add(0, faces[:, k], fn)
    return F.normalize(n, dim=-1)


def cube(size: float = 1.0, device=None, dtype=torch.float32) -> Tuple[torch.Tensor, torch.Tensor]:
    """Axis-aligned cube centred on the origin, 8 vertices, 12 outward counter-clockwise faces."""
    h = size / 2.0
    v = torch.tensor([[x, y, z] for x in (-h, h) for y in (-h, h) for z in (-h, h)],
                     device=device, dtype=dtype)
    # vertex index = 4 * (x > 0) + 2 * (y > 0) + (z > 0)
    f = torch.tensor([
        [0, 1, 3], [0, 3, 2],   # -x
        [4, 6, 7], [4, 7, 5],   # +x
        [0, 4, 5], [0, 5, 1],   # -y
        [2, 3, 7], [2, 7, 6],   # +y
        [0, 2, 6], [0, 6, 4],   # -z
        [1, 5, 7], [1, 7, 3],   # +z
    ], device=device, dtype=torch.int64)
    return v, f


def uv_sphere(rings: int = 16, segments: int = 32, radius: float = 1.0, device=None,
              dtype=torch.float32) -> Tuple[torch.Tensor, torch.Tensor]:
    """Closed UV sphere with outward counter-clockwise faces: 2 (rings - 1) segments faces."""
    verts = [[0.0, radius, 0.0]]
    for r in range(1, rings):
        theta = math.pi * r / rings
        for s in range(segments):
            phi = 2.0 * math.pi * s / segments
            verts.append([radius * math.sin(theta) * math.cos(phi), radius * math.cos(theta),
                          -radius * math.sin(theta) * math.sin(phi)])
    verts.append([0.0, -radius, 0.0])
    faces = []
    bottom = len(verts) - 1

    def idx(r, s):
        return 1 + (r - 1) * segments + (s % segments)

    for s in range(segments):
        faces.append([0, idx(1, s), idx(1, s + 1)])
        faces.append([bottom, idx(rings - 1, s + 1), idx(rings - 1, s)])
    for r in range(1, rings - 1):
        for s in range(segments):
            a, b = idx(r, s), idx(r, s + 1)
            c, d = idx(r + 1, s), idx(r + 1, s + 1)
            faces.append([a, c, d])
            faces.append([a, d, b])
    return (torch.tensor(verts, device=device, dtype=dtype),
            torch.tensor(faces, device=device, dtype=torch.int64))
