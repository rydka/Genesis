import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    logger.warning("sentence-transformers not installed — CLIP embeddings unavailable")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("Pillow not installed — image loading unavailable")


@dataclass
class CLIPConfig:
    model_name: str = "clip-ViT-L-14"
    batch_size: int = 32
    normalize_embeddings: bool = True


@dataclass
class CLIPEmbeddingResult:
    embedding: np.ndarray
    source_type: str
    processing_time: float
    model_name: str
    embedding_dim: int


class CLIPEmbedder:
    def __init__(self, config: Optional[CLIPConfig] = None):
        self.config = config or CLIPConfig()
        self.model: Optional[SentenceTransformer] = None
        self._model_loaded = False

    def _load_model(self) -> bool:
        if self._model_loaded:
            return True
        if not CLIP_AVAILABLE:
            logger.error("sentence-transformers not available")
            return False

        start = time.time()
        try:
            self.model = SentenceTransformer(self.config.model_name)
            self._model_loaded = True
            elapsed = time.time() - start
            logger.info("CLIP model '%s' loaded in %.2fs (dim=%d)",
                        self.config.model_name,
                        elapsed,
                        self.model.get_sentence_embedding_dimension())
            return True
        except Exception:
            logger.exception("Failed to load CLIP model '%s'", self.config.model_name)
            return False

    def embed_images(self, image_paths: List[Path]) -> np.ndarray:
        if not self._load_model():
            raise RuntimeError("CLIP model not available")
        if not PIL_AVAILABLE:
            raise RuntimeError("Pillow not available")

        images = []
        valid_indices = []
        for i, path in enumerate(image_paths):
            try:
                img = Image.open(path).convert("RGB")
                images.append(img)
                valid_indices.append(i)
            except Exception:
                logger.warning("Failed to load image: %s", path)

        if not images:
            raise ValueError("No valid images to embed")

        embeddings = self.model.encode(
            images,
            batch_size=self.config.batch_size,
            normalize_embeddings=self.config.normalize_embeddings,
        )
        return np.asarray(embeddings)

    def embed_text(self, query: str) -> np.ndarray:
        if not self._load_model():
            raise RuntimeError("CLIP model not available")

        embedding = self.model.encode(
            [query],
            normalize_embeddings=self.config.normalize_embeddings,
        )
        return np.asarray(embedding[0])

    def embed_texts(self, queries: List[str]) -> np.ndarray:
        if not self._load_model():
            raise RuntimeError("CLIP model not available")

        embeddings = self.model.encode(
            queries,
            batch_size=self.config.batch_size,
            normalize_embeddings=self.config.normalize_embeddings,
        )
        return np.asarray(embeddings)

    def embed_asset_views(self, view_paths: Dict[str, Path]) -> np.ndarray:
        paths = [p for p in view_paths.values() if Path(p).exists()]
        if not paths:
            raise ValueError("No valid view paths provided")

        embeddings = self.embed_images(paths)
        composite = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(composite)
        if norm > 0:
            composite = composite / norm
        return composite

    def get_embedding_dim(self) -> int:
        if not self._load_model():
            return 768
        return self.model.get_sentence_embedding_dimension() or 768

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_name": self.config.model_name,
            "loaded": self._model_loaded,
            "embedding_dim": self.get_embedding_dim(),
            "normalize": self.config.normalize_embeddings,
        }
