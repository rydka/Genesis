"""Signed distance grids and FlexiCubes extraction on a sparse set of cubes.

Building blocks for shape optimisation with a change of topology (docs/conception/
optimisation-par-zones.md, phase 1):
  - `make_grid` / `grid_points`: a regular grid of cubic voxels around an object;
  - `mesh_sdf`: signed distance of points to a triangle mesh (open3d, MIT). The distance comes from
    `RaycastingScene.compute_distance`; the sign from a parity vote over several ray directions
    (`count_intersections`), which stays right next to holes of moderate size, unlike open3d's own
    `compute_signed_distance` (one ray direction: every point whose ray leaves through a hole is
    "outside", whatever `nsamples`; measured on a sphere with a cap removed);
  - `SparseGrid` + `SparseFlexiCubes`: FlexiCubes on a band of cubes only (those that can hold the
    surface), so that a 192^3 grid costs about as much as the band, not as the full volume.

The vendored FlexiCubes (`flexicubes/`) is unchanged; `SparseFlexiCubes` only replaces the one
method that assumes a dense grid (`_get_case_id`, ambiguity resolution between neighbouring cubes).
"""

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import torch

from .flexicubes import FlexiCubes

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:  # pragma: no cover - open3d is in config/requirements/generation.txt
    OPEN3D_AVAILABLE = False

# Corner order of FlexiCubes (bit 0 = x, bit 1 = y, bit 2 = z)
CORNERS = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
                    [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]], np.int64)


# --------------------------------------------------------------------------------------------------
# Grid
# --------------------------------------------------------------------------------------------------

@dataclass
class Grid:
    """Regular grid of `dims` = (nx, ny, nz) cubes of side `h`; vertex (i, j, k) is origin + h (i, j, k)."""

    origin: np.ndarray
    h: float
    dims: Tuple[int, int, int]

    @property
    def vertex_dims(self) -> Tuple[int, int, int]:
        return tuple(d + 1 for d in self.dims)

    def to_dict(self) -> dict:
        return {"origin": [float(x) for x in self.origin], "h": float(self.h), "dims": list(self.dims)}


def make_grid(lo: Sequence[float], hi: Sequence[float], res: int, pad: int = 3) -> Grid:
    """Grid whose longest side has `res` cubes over [lo, hi], plus `pad` cubes on every side."""
    lo, hi = np.asarray(lo, np.float64), np.asarray(hi, np.float64)
    h = float((hi - lo).max()) / res
    dims = tuple(int(math.ceil((hi[a] - lo[a]) / h - 1e-9)) + 2 * pad for a in range(3))
    center = (lo + hi) / 2
    origin = center - h * np.asarray(dims, np.float64) / 2
    return Grid(origin.astype(np.float64), h, dims)


def grid_points(grid: Grid) -> np.ndarray:
    """All vertex positions [(nx+1)(ny+1)(nz+1), 3], flat index (i * (ny+1) + j) * (nz+1) + k."""
    axes = [grid.origin[a] + grid.h * np.arange(grid.vertex_dims[a]) for a in range(3)]
    g = np.stack(np.meshgrid(*axes, indexing="ij"), -1)
    return g.reshape(-1, 3).astype(np.float32)


def cube_corner_reduce(values: np.ndarray, op) -> np.ndarray:
    """Reduce a per-vertex array [nx+1, ny+1, nz+1] over the 8 corners of every cube -> [nx, ny, nz]."""
    out = None
    for dx, dy, dz in CORNERS:
        sl = values[dx:values.shape[0] - 1 + dx, dy:values.shape[1] - 1 + dy,
                    dz:values.shape[2] - 1 + dz]
        out = sl.copy() if out is None else op(out, sl)
    return out


def sign_change_cubes(s: np.ndarray) -> np.ndarray:
    """Cubes [nx, ny, nz] whose corners do not all have the same sign (s < 0 is inside)."""
    inside = s < 0
    return cube_corner_reduce(inside, np.logical_or) & ~cube_corner_reduce(inside, np.logical_and)


# --------------------------------------------------------------------------------------------------
# Signed distance to a mesh
# --------------------------------------------------------------------------------------------------

def fibonacci_directions(k: int) -> np.ndarray:
    """k well spread unit directions (not aligned with the grid axes)."""
    i = np.arange(k) + 0.5
    phi = np.arccos(1 - 2 * i / k)
    theta = math.pi * (1 + 5 ** 0.5) * i + 0.3
    return np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)],
                    1).astype(np.float32)


def mesh_sdf(verts: np.ndarray, faces: np.ndarray, points: np.ndarray, n_dirs: int = 15,
             chunk: int = 2_000_000) -> Tuple[np.ndarray, dict]:
    """Signed distance [N] of `points` to the mesh: < 0 inside. Returns (sdf, info).

    The sign is a majority vote of the ray-parity test over `n_dirs` directions: a point is inside
    when most rays cross the surface an odd number of times. On a watertight mesh every direction
    agrees; next to a hole only the rays that leave through it disagree. info["ambiguous"] counts
    the points whose vote is close (between 1/3 and 2/3): they sit near a large hole or inside an
    open, thin part, and their sign is uncertain.
    """
    if not OPEN3D_AVAILABLE:
        raise RuntimeError("open3d is required for mesh_sdf (pip install open3d)")
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(np.ascontiguousarray(verts, np.float32)),
                        o3d.core.Tensor(np.ascontiguousarray(faces, np.uint32)))
    dirs = fibonacci_directions(n_dirs)
    points = np.ascontiguousarray(points, np.float32)
    dist = np.empty(len(points), np.float32)
    votes = np.zeros(len(points), np.int16)
    for a in range(0, len(points), chunk):
        p = points[a:a + chunk]
        dist[a:a + chunk] = scene.compute_distance(o3d.core.Tensor(p)).numpy()
        for d in dirs:
            rays = np.concatenate([p, np.broadcast_to(d, p.shape)], 1)
            votes[a:a + chunk] += (scene.count_intersections(o3d.core.Tensor(rays)).numpy() % 2
                                   ).astype(np.int16)
    inside = votes * 2 > n_dirs
    frac = votes / n_dirs
    info = {"n_dirs": n_dirs, "inside": int(inside.sum()),
            "ambiguous": int(((frac > 1 / 3) & (frac < 2 / 3)).sum()),
            "unanimous_fraction": float(((votes == 0) | (votes == n_dirs)).mean())}
    return np.where(inside, -dist, dist).astype(np.float32), info


# --------------------------------------------------------------------------------------------------
# Sparse FlexiCubes
# --------------------------------------------------------------------------------------------------

@dataclass
class SparseGrid:
    """A subset of the cubes of a Grid, with compact vertex indices.

    cube_ijk:  [C, 3] cube coordinates; cubes: [C, 8] indices into the compact vertex list
    vert_flat: [N] flat dense index of every compact vertex; verts: [N, 3] positions (float32).
    """

    grid: Grid
    cube_ijk: torch.Tensor
    cubes: torch.Tensor
    vert_flat: torch.Tensor
    verts: torch.Tensor

    @staticmethod
    def from_cube_mask(grid: Grid, cube_mask: np.ndarray, device="cpu") -> "SparseGrid":
        ijk = np.argwhere(cube_mask).astype(np.int64)                    # [C, 3]
        vx, vy, vz = grid.vertex_dims
        corners = ijk[:, None, :] + CORNERS[None]                         # [C, 8, 3]
        flat = (corners[..., 0] * vy + corners[..., 1]) * vz + corners[..., 2]
        vert_flat, inv = np.unique(flat.reshape(-1), return_inverse=True)
        cubes = inv.reshape(-1, 8)
        k = vert_flat % vz
        j = (vert_flat // vz) % vy
        i = vert_flat // (vz * vy)
        pos = grid.origin[None] + grid.h * np.stack([i, j, k], 1)
        t = lambda a, dt=torch.long: torch.as_tensor(a, dtype=dt, device=device)  # noqa: E731
        return SparseGrid(grid, t(ijk), t(cubes), t(vert_flat), t(pos.astype(np.float32), torch.float32))

    @property
    def n_cubes(self) -> int:
        return self.cubes.shape[0]


class SparseFlexiCubes(FlexiCubes):
    """FlexiCubes on an arbitrary subset of the cubes of a regular grid.

    Upstream `_get_case_id` resolves the C16/C19 ambiguity by looking up the neighbouring cube in a
    dense [res^3, 5] array indexed by the position of the cube in a dense grid. Here the neighbour
    is found by its coordinates (sorted keys + searchsorted); a neighbour that is not in the subset
    is not a surface cube, exactly as in the dense version.
    """

    def __init__(self, device="cpu", **kw):
        super().__init__(device=device, **kw)
        self._cube_ijk = None
        self._dims = None

    def extract(self, sparse: SparseGrid, s: torch.Tensor, x: Optional[torch.Tensor] = None, **kw):
        """(vertices, faces, L_dev) of the zero level set of `s` [N] on the sparse grid."""
        self._cube_ijk = sparse.cube_ijk
        self._dims = sparse.grid.dims
        x = sparse.verts if x is None else x
        return self(x, s, sparse.cubes, list(sparse.grid.dims), **kw)

    @torch.no_grad()
    def _get_case_id(self, occ_fx8, surf_cubes, res):
        case_ids = (occ_fx8[surf_cubes] * self.cube_corners_idx.to(self.device).unsqueeze(0)).sum(-1)
        problem_config = self.check_table.to(self.device)[case_ids]
        to_check = problem_config[..., 0] == 1
        problem_config = problem_config[to_check]
        if problem_config.shape[0] == 0:
            return case_ids
        nx, ny, nz = self._dims
        ijk = self._cube_ijk[surf_cubes][to_check]
        adj = ijk + problem_config[..., 1:4]
        within = ((adj >= 0) & (adj < torch.tensor([nx, ny, nz], device=adj.device))).all(-1)

        def key(c):
            return (c[..., 0] * ny + c[..., 1]) * nz + c[..., 2]

        keys = key(ijk)
        order = torch.argsort(keys)
        sorted_keys = keys[order]
        adj_keys = key(adj[within])
        pos = torch.searchsorted(sorted_keys, adj_keys).clamp(max=sorted_keys.shape[0] - 1)
        # The neighbour is itself an ambiguous surface cube: both cases are inverted
        to_invert = sorted_keys[pos] == adj_keys
        idx = torch.arange(case_ids.shape[0], device=self.device)[to_check][within][to_invert]
        case_ids.index_put_((idx,), problem_config[within][to_invert][..., -1])
        return case_ids


def grid_edges(cubes: torch.Tensor) -> torch.Tensor:
    """Unique grid edges [E, 2] (compact vertex indices) of a set of cubes [C, 8]."""
    e = cubes[:, [0, 1, 0, 2, 0, 4, 1, 3, 1, 5, 2, 3, 2, 6, 3, 7, 4, 5, 4, 6, 5, 7, 6, 7]].reshape(-1, 2)
    return torch.unique(e, dim=0)


def sdf_sign_regularizer(s: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    """Regulariser of the FlexiCubes paper (from nvdiffrec): binary cross-entropy between the signs
    of the two ends of every grid edge that crosses the surface. Discourages spurious sign changes.
    `s` should be in voxel units (s / h) so that the logits have a sensible scale."""
    a, b = s[edges[:, 0]], s[edges[:, 1]]
    cross = (a > 0) != (b > 0)
    if not bool(cross.any()):
        return s.sum() * 0
    a, b = a[cross], b[cross]
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    return bce(a, (b > 0).float()) + bce(b, (a > 0).float())
