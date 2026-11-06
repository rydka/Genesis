"""Continuous remeshing (Palfinger 2022, MIT), vendored from Profactor/continuous-remeshing; see
README.md in this folder.

Only `core/remesh.py` and `core/opt.py` are copied; `torch_scatter` is replaced by native PyTorch
scatter operations, so the only dependency is `torch`. Nothing of the upstream example (nvdiffrast)
is used or imported.
"""

from .opt import MeshOptimizer, remesh
from .remesh import calc_edges, calc_face_normals, calc_vertex_normals

__all__ = ["MeshOptimizer", "calc_edges", "calc_face_normals", "calc_vertex_normals", "remesh"]
