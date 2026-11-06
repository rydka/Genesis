"""Reference gradients of pixel coverage, to measure the accuracy of the silhouette gradients.

The reference image is the box-filtered coverage of each pixel, computed by rasterizing at
`supersample` x `supersample` samples per pixel (with the same z-buffer, so occlusions and
self-intersections are handled) and averaging. The reference gradient of a loss w.r.t. the vertex
coordinates is obtained by central finite differences on that image. This is the quantity the
literature measures against (Pidhorskyi et al. 2024 use ray tracing for the same purpose).

Scenes are 2D-like: clip coordinates with w = 1, so clip x, y are NDC and a step of
2 / width in x is one pixel.
"""

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import torch

from .antialias import antialias
from .edgegrad import edge_gradients
from .rasterize import face_attributes, rasterize

F64 = torch.float64


@dataclass
class Scene:
    """Triangles in clip space (w = 1), per-face constant values, and a fixed loss weight image."""

    name: str
    clip: torch.Tensor            # [V, 4] float64
    faces: torch.Tensor           # [F, 3]
    values: torch.Tensor          # [F, C] per-face values (background = 0)
    resolution: Tuple[int, int]
    coords: List[Tuple[int, int]]  # (vertex, clip coordinate) pairs whose derivative is measured

    def weights(self) -> torch.Tensor:
        """Smooth loss weights g [H, W, C]: L = sum(g * image)."""
        h, w = self.resolution
        i = torch.arange(h, dtype=F64).view(h, 1)
        j = torch.arange(w, dtype=F64).view(1, w)
        y = 1 - 2 * (i + 0.5) / h
        x = 2 * (j + 0.5) / w - 1
        base = 1.0 + 0.5 * torch.sin(2.1 * x + 0.3) * torch.cos(1.7 * y - 0.2)
        chans = [base * (1.0 + 0.3 * c) for c in range(self.values.shape[1])]
        return torch.stack(chans, -1)


def coverage_image(clip: torch.Tensor, faces: torch.Tensor, values: torch.Tensor,
                   resolution: Tuple[int, int], supersample: int) -> torch.Tensor:
    """Box-filtered image [H, W, C] from `supersample`^2 regularly spaced samples per pixel."""
    h, w = resolution
    s = supersample
    rast = rasterize(clip, faces, (h * s, w * s), max_fragments=1 << 23)
    img = face_attributes(values, rast)[0]                       # [hS, wS, C]
    return img.view(h, s, w, s, -1).mean(dim=(1, 3))


def reference_gradient(scene: Scene, supersample: int = 64, step_px: float = 0.25) -> torch.Tensor:
    """d(sum(g * coverage)) / d(coordinate) for each (vertex, coord) of the scene, by central
    differences with a step of `step_px` pixels."""
    g = scene.weights()
    h, w = scene.resolution
    out = []
    for v, c in scene.coords:
        # Depth has no pixel scale: the same step as x moves an intersection line by a fraction
        # of a pixel for depth slopes of order 1 (a much smaller step would be below the
        # sample spacing and measure the sampling staircase instead of the derivative).
        step = step_px * 2.0 / (h if c == 1 else w)
        vals = []
        for sgn in (1.0, -1.0):
            clip = scene.clip.clone()
            clip[v, c] += sgn * step
            vals.append(float((g * coverage_image(clip, scene.faces, scene.values,
                                                  scene.resolution, supersample)).sum()))
        out.append((vals[0] - vals[1]) / (2 * step))
    return torch.tensor(out, dtype=F64)


MODES: Dict[str, Callable] = {
    "analytic": lambda img, rast, clip, faces: antialias(img, rast, clip, faces),
    "micro_edge": lambda img, rast, clip, faces: edge_gradients(img, rast, clip, faces),
    "micro_edge_fragment": lambda img, rast, clip, faces: edge_gradients(
        img, rast, clip, faces, scatter="fragment"),
}


def method_gradient(scene: Scene, mode: str, supersample: int = 1) -> torch.Tensor:
    """Same derivatives with a rendering mode of this package (hard rasterization + `mode`),
    optionally rendered at `supersample` times the resolution and average-pooled."""
    clip = scene.clip.clone().requires_grad_()
    h, w = scene.resolution
    s = supersample
    rast = rasterize(clip, scene.faces, (h * s, w * s))
    img = face_attributes(scene.values, rast)
    out = MODES[mode](img, rast, clip, scene.faces)[0]
    out = out.view(h, s, w, s, -1).mean(dim=(1, 3))
    loss = (out * scene.weights()).sum()
    if not loss.requires_grad:                                   # no edge found at all
        return torch.zeros(len(scene.coords), dtype=F64)
    loss.backward()
    return torch.stack([clip.grad[v, c] for v, c in scene.coords])


def relative_error(grad: torch.Tensor, ref: torch.Tensor) -> float:
    return float((grad - ref).norm() / ref.norm().clamp(min=1e-30))


def _clip(xyz: Sequence[Sequence[float]]) -> torch.Tensor:
    t = torch.tensor(xyz, dtype=F64)
    if t.shape[1] == 2:
        t = torch.cat([t, torch.zeros(t.shape[0], 1, dtype=F64)], 1)
    return torch.cat([t, torch.ones(t.shape[0], 1, dtype=F64)], 1)


def _xy(nv: int) -> List[Tuple[int, int]]:
    return [(v, c) for v in range(nv) for c in (0, 1)]


def scenes(resolution: int = 16) -> Dict[str, Scene]:
    """The test cases of the accuracy table (README). Pixel size in NDC: 2 / resolution."""
    r = (resolution, resolution)
    px = 2.0 / resolution
    one = torch.tensor([[1.0]], dtype=F64)
    cases = {}
    # Large triangle whose right edge is the vertical line x = e (not on a pixel boundary)
    e = 0.13 + 0.37 * px
    cases["vertical_edge"] = Scene(
        "vertical_edge", _clip([[-3.0, -3.0], [e, -3.0], [e, 3.0]]), torch.tensor([[0, 1, 2]]),
        one, r, [(1, 0), (2, 0)])
    # Large triangle with a generic diagonal edge
    cases["diagonal_edge"] = Scene(
        "diagonal_edge", _clip([[-0.83, -0.71], [0.62, -0.29], [-0.31, 0.77]]),
        torch.tensor([[0, 1, 2]]), one, r, _xy(3))
    # Blade: long triangle, at most 0.6 px wide, slightly tilted so that it covers some centres
    cases["thin_blade"] = Scene(
        "thin_blade", _clip([[0.031, -0.81], [0.031 + 0.6 * px, -0.79], [0.11, 0.83]]),
        torch.tensor([[0, 1, 2]]), one, r, _xy(3))
    # Triangle smaller than a pixel (about 0.7 px), covering one pixel centre
    c0 = (px * 0.5 + 0.013, px * 0.5 - 0.021)
    cases["subpixel"] = Scene(
        "subpixel", _clip([[c0[0] - 0.35 * px, c0[1] - 0.3 * px], [c0[0] + 0.4 * px, c0[1] - 0.2 * px],
                           [c0[0] - 0.05 * px, c0[1] + 0.4 * px]]),
        torch.tensor([[0, 1, 2]]), one, r, _xy(3))
    # Same size, covering no pixel centre: invisible to both modes (documented failure)
    cases["subpixel_no_centre"] = Scene(
        "subpixel_no_centre", _clip([[0.02 * px, 0.05 * px], [0.45 * px, 0.08 * px],
                                     [0.2 * px, 0.42 * px]]),
        torch.tensor([[0, 1, 2]]), one, r, _xy(3))
    # Occlusion: a front triangle over a larger back triangle, different values
    cases["occlusion"] = Scene(
        "occlusion", _clip([[-0.41, -0.53, -0.5], [0.57, -0.17, -0.5], [-0.12, 0.61, -0.5],
                            [-0.9, -0.87, 0.5], [0.83, -0.8, 0.5], [0.05, 0.92, 0.5]]),
        torch.tensor([[0, 1, 2], [3, 4, 5]]),
        torch.tensor([[1.0, 0.2], [0.3, 1.0]], dtype=F64), r, _xy(6))
    # Self-intersection: two triangles whose depth planes cross along a slanted line
    cases["intersection"] = Scene(
        "intersection", _clip([[-0.85, -0.8, -0.6], [0.8, -0.85, 0.6], [-0.05, 0.85, 0.0],
                               [-0.8, -0.75, 0.5], [0.85, -0.7, -0.4], [0.1, 0.9, 0.1]]),
        torch.tensor([[0, 1, 2], [3, 4, 5]]),
        torch.tensor([[1.0, 0.2], [0.3, 1.0]], dtype=F64), r,
        [(v, c) for v in range(6) for c in (0, 1, 2)])
    return cases


def accuracy_table(resolution: int = 16, supersample: int = 64,
                   modes: Sequence[str] = ("analytic", "micro_edge", "micro_edge_fragment"),
                   supersampled: Sequence[str] = ("analytic", "micro_edge")):
    """{case: {mode: relative error}} for all cases of `scenes`; `<mode>_x4` columns are rendered
    at 4 x 4 samples per pixel and average-pooled."""
    table = {}
    for name, scene in scenes(resolution).items():
        ref = reference_gradient(scene, supersample)
        row = {"reference_norm": float(ref.norm())}
        for mode in modes:
            row[mode] = relative_error(method_gradient(scene, mode), ref)
        for mode in supersampled:
            row[f"{mode}_x4"] = relative_error(method_gradient(scene, mode, 4), ref)
        table[name] = row
    return table


if __name__ == "__main__":
    for case, row in accuracy_table().items():
        print(f"{case:20s}", {k: (round(v, 4) if math.isfinite(v) else v) for k, v in row.items()})


# ---------------------------------------------------------------- thin structure fit

def blade_vertices(x0: torch.Tensor, width: torch.Tensor, resolution: int) -> torch.Tensor:
    """A thin, slightly tilted quad (two triangles) of `width` pixels whose centre line crosses
    y = 0 at x0 pixels from the image centre. Returns clip vertices [4, 4] (w = 1)."""
    px = 2.0 / resolution
    half = 0.5 * width * px
    tilt = 0.23                                          # x shift per unit of y (NDC)
    ys = torch.tensor([-0.8, -0.8, 0.8, 0.8], dtype=x0.dtype)
    side = torch.stack([-half, half, half, -half])
    x = x0 * px + tilt * ys + side
    zeros, ones = torch.zeros_like(ys), torch.ones_like(ys)
    return torch.stack([x, ys, zeros, ones], dim=1)


BLADE_FACES = torch.tensor([[0, 1, 2], [0, 2, 3]])


def blade_fit(mode: str, supersample: int = 1, resolution: int = 32, steps: int = 150,
              true_x0: float = 0.37, true_width: float = 0.45, init_x0: float = 1.1,
              init_width: float = 1.3, lr: float = 0.03) -> Dict[str, float]:
    """Recover the position and width (in pixels) of a thin blade from its antialiased
    silhouette (target: 64 x 64 supersampled coverage), with an L2 loss and Adam.
    `converged_at` is the step from which both errors stay below 0.05 px (-1: not at the end)."""
    one = torch.ones(2, 1, dtype=F64)
    target = coverage_image(blade_vertices(torch.tensor(true_x0, dtype=F64),
                                           torch.tensor(true_width, dtype=F64), resolution),
                            BLADE_FACES, one, (resolution, resolution), 64)
    x0 = torch.tensor(init_x0, dtype=F64, requires_grad=True)
    width = torch.tensor(init_width, dtype=F64, requires_grad=True)
    opt = torch.optim.Adam([x0, width], lr=lr)
    s = supersample
    losses = []
    close = []                                           # both errors < 0.05 px, per step
    for _ in range(steps):
        close.append(abs(float(x0) - true_x0) < 0.05 and abs(float(width) - true_width) < 0.05)
        opt.zero_grad()
        clip = blade_vertices(x0, width, resolution)
        rast = rasterize(clip, BLADE_FACES, (resolution * s, resolution * s))
        img = MODES[mode](face_attributes(one, rast), rast, clip, BLADE_FACES)[0]
        img = img.view(resolution, s, resolution, s, -1).mean(dim=(1, 3))
        loss = ((img - target) ** 2).sum()
        losses.append(float(loss))
        if loss.requires_grad:
            loss.backward()
            opt.step()
    close.append(abs(float(x0) - true_x0) < 0.05 and abs(float(width) - true_width) < 0.05)
    # first step from which both errors stay below 0.05 px until the end (-1: not at the end)
    converged_at = -1
    for step in range(len(close) - 1, -1, -1):
        if not close[step]:
            break
        converged_at = step
    return {"x0_error_px": abs(float(x0) - true_x0), "width_error_px": abs(float(width) - true_width),
            "width": float(width), "x0": float(x0), "loss_first": losses[0], "loss_last": losses[-1],
            "loss_min": min(losses), "converged_at": converged_at}
