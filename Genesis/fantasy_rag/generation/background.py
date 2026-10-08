import gc
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import onnxruntime as ort
    # onnxruntime-gpu needs the CUDA/cuDNN DLLs shipped with torch on Windows
    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls()
    from rembg import new_session, remove as rembg_remove
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False
    logger.warning("rembg not installed — background removal unavailable")

try:
    import cv2
    from scipy import ndimage
    from pymatting import estimate_foreground_ml
    REFINE_AVAILABLE = True
except ImportError:
    REFINE_AVAILABLE = False
    logger.warning("opencv/scipy/pymatting missing — mask refinement unavailable")

from PIL import Image

# u2net is kept for comparison only: its soft halo makes Hunyuan3D reconstruct the background
BG_MODELS = ["birefnet-general", "isnet-anime", "u2net"]


@dataclass
class RefineConfig:
    alpha_threshold: int = 30
    close_ratio: float = 1 / 80
    thicken_ratio: float = 1 / 250
    min_island_ratio: float = 0.0005
    max_hole_ratio: float = 0.001


class BackgroundRemover:

    def __init__(self, model: str = "birefnet-general", refine: bool = True,
                 refine_config: Optional[RefineConfig] = None):
        self.model = model
        self.refine = refine
        self.refine_config = refine_config or RefineConfig()
        self._session = None
        self._session_model = None

    @property
    def is_available(self) -> bool:
        return REMBG_AVAILABLE

    def _get_session(self):
        if self._session is None or self._session_model != self.model:
            self._session = new_session(self.model)
            self._session_model = self.model
            logger.info("rembg session %s on %s", self.model,
                        self._session.inner_session.get_providers()[0])
        return self._session

    def __call__(self, image: Image.Image) -> Image.Image:
        source = image.convert("RGB")
        rgba = np.asarray(rembg_remove(source, session=self._get_session()).convert("RGBA"))
        if self.refine:
            if REFINE_AVAILABLE:
                rgba = refine_mask(rgba, np.asarray(source), self.refine_config)
            else:
                logger.warning("Mask refinement skipped (missing dependencies)")
        return Image.fromarray(rgba)

    def unload(self):
        if self._session is None:
            return
        self._session = None
        self._session_model = None
        gc.collect()


def refine_mask(rgba: np.ndarray, source_rgb: np.ndarray,
                config: Optional[RefineConfig] = None) -> np.ndarray:
    """Turn a soft rembg alpha into a clean binary silhouette with thickened thin parts.

    Hunyuan3D crops on every alpha > 0 pixel and works at 512px, so soft halos skew the
    framing and 1-2px hair strands become floating fragments.
    """
    cfg = config or RefineConfig()
    h, w = rgba.shape[:2]
    alpha = rgba[..., 3]

    # True foreground colors: un-mix semi-transparent pixels from the source background
    rgb = estimate_foreground_ml(source_rgb / 255.0, alpha / 255.0)
    rgb = (rgb.clip(0, 1) * 255).astype(np.uint8)

    # Binarize with a low threshold to keep faint strands
    mask = (alpha > cfg.alpha_threshold).astype(np.uint8)
    initial = mask.copy()

    def ellipse(ratio):
        k = max(3, int(round(max(h, w) * ratio)) | 1)
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    # Close gaps between neighbouring strands so hair becomes a solid mass
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ellipse(cfg.close_ratio))

    # Thicken only thin structures (what an opening would erase)
    k_thin = ellipse(cfg.thicken_ratio)
    thin = cv2.subtract(mask, cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_thin))
    mask = cv2.max(mask, cv2.dilate(thin, k_thin))

    # Drop small floating islands
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        keep = np.zeros(n, dtype=bool)
        keep[1:] = areas >= areas.max() * cfg.min_island_ratio
        mask = keep[labels].astype(np.uint8)

    # Fill small interior holes only (keep real gaps like arm/torso)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(1 - mask, connectivity=4)
    border_labels = set(np.unique(np.concatenate(
        [labels[0], labels[-1], labels[:, 0], labels[:, -1]])))
    for lbl in range(1, n):
        if lbl not in border_labels and stats[lbl, cv2.CC_STAT_AREA] < h * w * cfg.max_hole_ratio:
            mask[labels == lbl] = 1

    # Pixels added by closing/thickening have no color of their own:
    # copy the nearest originally-kept pixel, otherwise they stay background-colored
    if initial.any():
        _, (iy, ix) = ndimage.distance_transform_edt(initial == 0, return_indices=True)
        fix = (mask == 1) & (initial == 0)
        rgb[fix] = rgb[iy[fix], ix[fix]]

    rgb[mask == 0] = 255
    return np.dstack([rgb, mask * 255]).astype(np.uint8)
