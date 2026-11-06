"""Cameras for the differentiable rasterizer.

Conventions (OpenGL-like, see README):
- camera space: right-handed, the camera looks down -Z, +Y is up;
- clip space: (x, y, z, w) with NDC = xyz / w in [-1, 1]^3, NDC y points up;
- images: row 0 is the TOP of the image, column 0 is the left. Pixel (i, j) has its centre at
  x_ndc = 2 (j + 0.5) / W - 1 and y_ndc = 1 - 2 (i + 0.5) / H.

A crop keeps the same pose and only changes the projection matrix, so that a sub-window of the full
image is mapped onto the whole NDC square (used by the per-zone optimisation).
"""

import math
from dataclasses import dataclass, replace
from typing import Optional, Sequence, Tuple

import torch


def _tensor(x, device=None, dtype=torch.float32) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device or x.device, dtype=dtype)
    return torch.tensor(x, device=device, dtype=dtype)


def perspective(fov_y_deg: float, aspect: float = 1.0, near: float = 0.1, far: float = 100.0,
                device=None, dtype=torch.float32) -> torch.Tensor:
    """OpenGL perspective matrix. `aspect` is width / height."""
    f = 1.0 / math.tan(math.radians(fov_y_deg) / 2.0)
    m = torch.zeros(4, 4, device=device, dtype=dtype)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = 2.0 * far * near / (near - far)
    m[3, 2] = -1.0
    return m


def perspective_from_intrinsics(fx: float, fy: float, cx: float, cy: float, width: int, height: int,
                                near: float = 0.1, far: float = 100.0,
                                device=None, dtype=torch.float32) -> torch.Tensor:
    """Projection matrix from pinhole intrinsics in pixels (origin at the top-left corner of the
    image, v pointing down, pixel (i, j) centred on (j + 0.5, i + 0.5))."""
    m = torch.zeros(4, 4, device=device, dtype=dtype)
    m[0, 0] = 2.0 * fx / width
    m[0, 2] = -(2.0 * cx / width - 1.0)
    m[1, 1] = 2.0 * fy / height
    m[1, 2] = -(1.0 - 2.0 * cy / height)
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = 2.0 * far * near / (near - far)
    m[3, 2] = -1.0
    return m


def look_at(eye: Sequence[float], target: Sequence[float] = (0.0, 0.0, 0.0),
            up: Sequence[float] = (0.0, 1.0, 0.0), device=None, dtype=torch.float32) -> torch.Tensor:
    """World-to-camera matrix for a camera at `eye` looking at `target`."""
    eye = _tensor(eye, device, dtype)
    target = _tensor(target, device, dtype)
    up = _tensor(up, device, dtype)
    forward = target - eye
    forward = forward / forward.norm()
    right = torch.linalg.cross(forward, up)
    right = right / right.norm()
    true_up = torch.linalg.cross(right, forward)
    m = torch.eye(4, device=eye.device, dtype=dtype)
    m[0, :3] = right
    m[1, :3] = true_up
    m[2, :3] = -forward
    m[:3, 3] = -m[:3, :3] @ eye
    return m


def crop_projection(projection: torch.Tensor, box: Sequence[float]) -> torch.Tensor:
    """Restrict a projection to a sub-window of the image, same pose.

    `box` = (left, top, right, bottom) in normalised image coordinates of the full image
    ([0, 1], top-left origin). The returned matrix maps that window onto the whole NDC square.
    """
    left, top, right, bottom = [float(b) for b in box]
    if not (right > left and bottom > top):
        raise ValueError(f"empty crop box {box}")
    # Window in NDC: x in [2 left - 1, 2 right - 1], y in [1 - 2 bottom, 1 - 2 top]
    sx, sy = right - left, bottom - top
    cx, cy = left + right - 1.0, 1.0 - (top + bottom)
    c = torch.eye(4, device=projection.device, dtype=projection.dtype)
    # x_clip' = (x_clip - cx w) / sx, same for y; z and w unchanged
    c[0, 0], c[0, 3] = 1.0 / sx, -cx / sx
    c[1, 1], c[1, 3] = 1.0 / sy, -cy / sy
    return c @ projection


def transform_points(points: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    """Apply a 4x4 matrix (or a batch [B, 4, 4]) to 3D points [V, 3] or [B, V, 3].

    Returns homogeneous coordinates [..., V, 4]. Differentiable w.r.t. both arguments.
    """
    ones = torch.ones_like(points[..., :1])
    homo = torch.cat([points, ones], dim=-1)
    return homo @ matrix.transpose(-1, -2)


@dataclass
class Camera:
    """Pose (world-to-camera), projection and output resolution (height, width)."""

    view: torch.Tensor
    projection: torch.Tensor
    resolution: Tuple[int, int]

    @property
    def mvp(self) -> torch.Tensor:
        return self.projection @ self.view

    def to_clip(self, vertices: torch.Tensor) -> torch.Tensor:
        """World-space vertices [V, 3] (or [B, V, 3]) to clip space [..., V, 4]."""
        return transform_points(vertices, self.mvp.to(vertices.dtype))

    def crop(self, box_px: Sequence[float], resolution: Optional[Tuple[int, int]] = None) -> "Camera":
        """Sub-window (x0, y0, x1, y1) in pixels of this camera's image, same pose.

        The output resolution defaults to the size of the window, so that one pixel of the crop is one
        pixel of the full image; pass `resolution` to render the crop at a higher resolution.
        """
        h, w = self.resolution
        x0, y0, x1, y1 = [float(b) for b in box_px]
        box = (x0 / w, y0 / h, x1 / w, y1 / h)
        if resolution is None:
            resolution = (int(round(y1 - y0)), int(round(x1 - x0)))
        return replace(self, projection=crop_projection(self.projection, box),
                       resolution=(int(resolution[0]), int(resolution[1])))
