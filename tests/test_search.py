import pytest
from pathlib import Path
from PIL import Image

from fantasy_rag.embeddings.similarity_search import (
    SimilaritySearchEngine,
    SearchQuery,
)
from fantasy_rag.embeddings.clip_embedder import CLIPConfig


@pytest.fixture
def engine(chroma_tmp_dir):
    e = SimilaritySearchEngine(
        db_path=str(chroma_tmp_dir),
        clip_config=CLIPConfig(),
    )
    assert e.initialize()
    return e


def _make_colored_views(tmp_path, color, prefix="asset"):
    views = {}
    for view in ["front", "back", "left", "right", "top", "iso_3d"]:
        img = Image.new("RGB", (224, 224), color=color)
        path = tmp_path / f"{prefix}_{view}.png"
        img.save(path)
        views[view] = path
    return views


class TestInitialization:
    def test_engine_initializes(self, chroma_tmp_dir):
        e = SimilaritySearchEngine(db_path=str(chroma_tmp_dir))
        assert e.initialize()
        assert e.initialized

    def test_collection_starts_empty(self, engine):
        stats = engine.get_collection_stats()
        assert stats["total_documents"] == 0


class TestIndexing:
    def test_index_single_asset(self, engine, sample_view_paths):
        assert engine.index_asset("asset_001", sample_view_paths)
        stats = engine.get_collection_stats()
        assert stats["total_documents"] == 1

    def test_index_with_metadata(self, engine, sample_view_paths):
        meta = {"original_filename": "sword.stl", "file_size": 1234}
        assert engine.index_asset("asset_002", sample_view_paths, metadata=meta)

    def test_upsert_no_duplicates(self, engine, sample_view_paths):
        engine.index_asset("asset_dup", sample_view_paths)
        engine.index_asset("asset_dup", sample_view_paths)
        stats = engine.get_collection_stats()
        assert stats["total_documents"] == 1

    def test_index_multiple_assets(self, engine, tmp_path):
        for i, color in enumerate(["red", "green", "blue"]):
            views = _make_colored_views(tmp_path, color, prefix=f"asset_{i}")
            engine.index_asset(f"asset_{i}", views)
        stats = engine.get_collection_stats()
        assert stats["total_documents"] == 3


class TestSearch:
    @pytest.fixture(autouse=True)
    def _index_assets(self, engine, tmp_path):
        for i, color in enumerate(["red", "green", "blue"]):
            views = _make_colored_views(tmp_path, color, prefix=f"s_{i}")
            meta = {"original_filename": f"asset_{color}.stl"}
            engine.index_asset(f"s_{i}", views, metadata=meta)

    def test_search_returns_results(self, engine):
        response = engine.search(SearchQuery(text="a red object"))
        assert response.total_found > 0
        assert len(response.results) > 0

    def test_search_has_similarity_scores(self, engine):
        response = engine.search(SearchQuery(text="a colorful item"))
        for r in response.results:
            assert -1.0 <= r.similarity_score <= 1.0

    def test_search_respects_max_results(self, engine):
        response = engine.search(SearchQuery(text="object", max_results=2))
        assert len(response.results) <= 2

    def test_search_empty_collection(self, chroma_tmp_dir):
        e = SimilaritySearchEngine(db_path=str(chroma_tmp_dir / "empty"))
        e.initialize()
        response = e.search(SearchQuery(text="anything"))
        assert response.total_found == 0

    def test_search_metadata_present(self, engine):
        response = engine.search(SearchQuery(text="object"))
        for r in response.results:
            assert "asset_id" in r.metadata


class TestClearCollection:
    def test_clear_removes_all(self, engine, sample_view_paths):
        engine.index_asset("to_clear", sample_view_paths)
        assert engine.get_collection_stats()["total_documents"] == 1
        engine.clear_collection()
        assert engine.get_collection_stats()["total_documents"] == 0
