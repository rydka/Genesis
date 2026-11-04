import numpy as np
import pytest
from pathlib import Path

from fantasy_rag.embeddings.clip_embedder import CLIPEmbedder, CLIPConfig


@pytest.fixture(scope="module")
def embedder():
    return CLIPEmbedder(CLIPConfig())


class TestCLIPEmbedderInit:
    def test_init_does_not_load_model(self):
        e = CLIPEmbedder()
        assert not e._model_loaded

    def test_lazy_load_on_first_use(self, embedder, sample_image):
        embedder.embed_images([sample_image])
        assert embedder._model_loaded


class TestEmbedImages:
    def test_single_image_shape(self, embedder, sample_image):
        result = embedder.embed_images([sample_image])
        assert result.shape == (1, 768)

    def test_multiple_images(self, embedder, sample_view_paths):
        paths = list(sample_view_paths.values())
        result = embedder.embed_images(paths)
        assert result.shape == (6, 768)

    def test_embeddings_are_normalized(self, embedder, sample_image):
        result = embedder.embed_images([sample_image])
        norm = np.linalg.norm(result[0])
        assert abs(norm - 1.0) < 1e-5

    def test_nonexistent_path_skipped(self, embedder, sample_image):
        result = embedder.embed_images([sample_image, Path("/nonexistent.png")])
        assert result.shape == (1, 768)

    def test_all_invalid_raises(self, embedder):
        with pytest.raises(ValueError, match="No valid images"):
            embedder.embed_images([Path("/nonexistent.png")])


class TestEmbedText:
    def test_text_shape(self, embedder):
        result = embedder.embed_text("a magical sword")
        assert result.shape == (768,)

    def test_text_is_normalized(self, embedder):
        result = embedder.embed_text("dragon shield")
        norm = np.linalg.norm(result)
        assert abs(norm - 1.0) < 1e-5

    def test_different_texts_differ(self, embedder):
        a = embedder.embed_text("a sword")
        b = embedder.embed_text("a flower")
        assert not np.allclose(a, b)


class TestEmbedTexts:
    def test_batch_shape(self, embedder):
        result = embedder.embed_texts(["sword", "shield", "armor"])
        assert result.shape == (3, 768)


class TestEmbedAssetViews:
    def test_composite_shape(self, embedder, sample_view_paths):
        result = embedder.embed_asset_views(sample_view_paths)
        assert result.shape == (768,)

    def test_composite_is_normalized(self, embedder, sample_view_paths):
        result = embedder.embed_asset_views(sample_view_paths)
        norm = np.linalg.norm(result)
        assert abs(norm - 1.0) < 1e-5

    def test_partial_views(self, embedder, sample_view_paths):
        partial = {k: v for k, v in list(sample_view_paths.items())[:3]}
        result = embedder.embed_asset_views(partial)
        assert result.shape == (768,)

    def test_no_valid_views_raises(self, embedder):
        with pytest.raises(ValueError):
            embedder.embed_asset_views({"front": Path("/nope.png")})


class TestCrossModal:
    def test_image_text_similarity_is_float(self, embedder, sample_image):
        img_emb = embedder.embed_images([sample_image])[0]
        txt_emb = embedder.embed_text("a red square")
        similarity = float(np.dot(img_emb, txt_emb))
        assert -1.0 <= similarity <= 1.0
