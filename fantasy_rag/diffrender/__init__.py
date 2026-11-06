"""Differentiable triangle rasterizer written for Genesis, in pure PyTorch.

Used for image-based mesh optimisation (silhouette, depth and normal losses) and for texture
baking in UV space. Written from papers and general principles; see README.md in this folder.
"""

from .antialias import antialias
from .camera import (
    Camera,
    crop_projection,
    look_at,
    perspective,
    perspective_from_intrinsics,
    transform_points,
)
from .edgegrad import edge_gradients
from .mesh_ops import cube, face_normals, uv_sphere, vertex_normals
from .rasterize import RasterOutput, face_attributes, interpolate, rasterize
from .uv import (
    bake_face_attributes,
    bake_from_view,
    bake_vertex_attributes,
    rasterize_uv,
    uv_to_clip,
)

__all__ = [
    "Camera", "RasterOutput", "antialias", "bake_face_attributes", "bake_from_view",
    "bake_vertex_attributes", "crop_projection", "cube", "edge_gradients", "face_attributes",
    "face_normals", "interpolate", "look_at", "perspective", "perspective_from_intrinsics",
    "rasterize", "rasterize_uv", "transform_points", "uv_sphere", "uv_to_clip", "vertex_normals",
]
