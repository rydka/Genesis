"""FlexiCubes (NVIDIA, Apache-2.0), vendored from nv-tlabs/FlexiCubes; see README.md in this folder.

Only `flexicubes.py` and `tables.py` are copied, at a commit after the switch to Apache-2.0. Nothing
from the upstream `examples/` (nvdiffrast, Kaolin) is used or imported.
"""

from .flexicubes import FlexiCubes

__all__ = ["FlexiCubes"]
