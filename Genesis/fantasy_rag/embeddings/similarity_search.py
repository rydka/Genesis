import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

from .clip_embedder import CLIPEmbedder, CLIPConfig

logger = logging.getLogger(__name__)


@dataclass
class SearchQuery:
    text: str
    category_filter: Optional[str] = None
    max_results: int = 10
    min_similarity: float = 0.0
    metadata_filters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    asset_id: str
    description: str
    similarity_score: float
    metadata: Dict[str, Any]
    distance: float


@dataclass
class SearchResponse:
    query: str
    results: List[SearchResult]
    total_found: int
    processing_time: float
    search_metadata: Dict[str, Any] = field(default_factory=dict)


class SimilaritySearchEngine:

    def __init__(self,
                 db_path: str = "ressources/data/chroma_db",
                 collection_name: str = "clip_asset_embeddings",
                 clip_config: Optional[CLIPConfig] = None):
        self.db_path = Path(db_path)
        self.collection_name = collection_name
        self.clip_config = clip_config or CLIPConfig()

        self.client = None
        self.collection = None
        self.embedder: Optional[CLIPEmbedder] = None
        self.initialized = False

    def initialize(self) -> bool:
        if not CHROMADB_AVAILABLE:
            logger.error("ChromaDB not available")
            return False

        try:
            self.db_path.mkdir(parents=True, exist_ok=True)

            self.client = chromadb.PersistentClient(
                path=str(self.db_path),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                ),
            )

            self.embedder = CLIPEmbedder(self.clip_config)

            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={
                    "hnsw:space": "cosine",
                    "description": "CLIP embeddings for fantasy 3D assets",
                },
            )
            logger.info("Search engine initialized — collection '%s' (%d documents)",
                        self.collection_name, self.collection.count())

            self.initialized = True
            return True

        except Exception:
            logger.exception("Failed to initialize search engine")
            return False

    def index_asset(self, asset_id: str, view_paths: Dict[str, Path],
                    metadata: Optional[Dict[str, Any]] = None) -> bool:
        if not self.initialized and not self.initialize():
            return False

        start = time.time()
        try:
            composite = self.embedder.embed_asset_views(view_paths)

            doc_metadata = {
                "asset_id": asset_id,
                "views_count": len(view_paths),
                "views_used": ",".join(sorted(view_paths.keys())),
                "embedding_model": self.clip_config.model_name,
                "indexed_at": time.time(),
            }
            if metadata:
                for k, v in metadata.items():
                    if isinstance(v, (str, int, float, bool)):
                        doc_metadata[k] = v

            self.collection.upsert(
                ids=[asset_id],
                embeddings=[composite.tolist()],
                documents=[asset_id],
                metadatas=[doc_metadata],
            )

            elapsed = time.time() - start
            logger.info("Indexed asset '%s' in %.2fs", asset_id, elapsed)
            return True

        except Exception:
            logger.exception("Failed to index asset '%s'", asset_id)
            return False

    def search(self, query: SearchQuery) -> SearchResponse:
        if not self.initialized and not self.initialize():
            raise RuntimeError("Search engine not initialized")

        start = time.time()
        try:
            query_emb = self.embedder.embed_text(query.text)

            where_clause = {}
            if query.category_filter:
                where_clause["category"] = query.category_filter
            where_clause.update(query.metadata_filters)

            chroma_results = self.collection.query(
                query_embeddings=[query_emb.tolist()],
                n_results=query.max_results,
                where=where_clause if where_clause else None,
                include=["documents", "metadatas", "distances"],
            )

            results = []
            ids = chroma_results.get("ids", [[]])[0]
            distances = chroma_results.get("distances", [[]])[0]
            metadatas = chroma_results.get("metadatas", [[]])[0]
            documents = chroma_results.get("documents", [[]])[0]

            for i, doc_id in enumerate(ids):
                distance = distances[i]
                similarity = 1.0 - distance

                if similarity < query.min_similarity:
                    continue

                meta = metadatas[i] or {}
                results.append(SearchResult(
                    asset_id=meta.get("asset_id", doc_id),
                    description=documents[i] if documents[i] else doc_id,
                    similarity_score=similarity,
                    metadata=meta,
                    distance=distance,
                ))

            elapsed = time.time() - start
            logger.info("Search '%s' — %d results in %.3fs",
                        query.text[:40], len(results), elapsed)

            return SearchResponse(
                query=query.text,
                results=results,
                total_found=len(results),
                processing_time=elapsed,
                search_metadata={
                    "collection_size": self.collection.count(),
                    "embedding_model": self.clip_config.model_name,
                },
            )

        except Exception:
            logger.exception("Search failed for query '%s'", query.text[:40])
            raise

    def get_collection_stats(self) -> Dict[str, Any]:
        if not self.initialized:
            return {"initialized": False}

        try:
            count = self.collection.count()
            stats = {
                "initialized": True,
                "total_documents": count,
                "collection_name": self.collection_name,
                "database_path": str(self.db_path),
            }

            if count > 0:
                sample = self.collection.get(
                    limit=min(100, count), include=["metadatas"]
                )
                categories = {}
                for meta in (sample.get("metadatas") or []):
                    if meta and "category" in meta:
                        cat = meta["category"]
                        categories[cat] = categories.get(cat, 0) + 1
                stats["categories"] = categories

            return stats

        except Exception:
            logger.exception("Error getting collection stats")
            return {"initialized": True, "error": "failed"}

    def clear_collection(self) -> bool:
        if not self.initialized:
            return False

        try:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={
                    "hnsw:space": "cosine",
                    "description": "CLIP embeddings for fantasy 3D assets",
                },
            )
            logger.info("Cleared collection '%s'", self.collection_name)
            return True
        except Exception:
            logger.exception("Error clearing collection")
            return False
