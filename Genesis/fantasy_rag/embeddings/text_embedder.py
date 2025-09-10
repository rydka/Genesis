#!/usr/bin/env python3
"""
Text Embedder for Fantasy Asset Descriptions

This module handles text embedding generation using SentenceTransformers
with multilingual support for French and English descriptions.
"""

import logging
import numpy as np
from typing import List, Dict, Optional, Union, Tuple
from dataclasses import dataclass
from pathlib import Path
import time

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class EmbeddingConfig:
    """Configuration for text embedding generation"""
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    max_seq_length: int = 512
    batch_size: int = 32
    normalize_embeddings: bool = True
    cache_dir: Optional[str] = None

@dataclass
class EmbeddingResult:
    """Result of text embedding generation"""
    embeddings: np.ndarray
    texts: List[str]
    processing_time: float
    model_name: str
    embedding_dim: int

class TextEmbedder:
    """
    Text embedder using SentenceTransformers for multilingual support
    """
    
    def __init__(self, config: EmbeddingConfig = None):
        """
        Initialize text embedder
        
        Args:
            config: Embedding configuration
        """
        self.config = config or EmbeddingConfig()
        self.model = None
        self.model_loaded = False
        
        logger.info(f"TextEmbedder initialized with model: {self.config.model_name}")
    
    def _load_model(self) -> bool:
        """Load the SentenceTransformer model"""
        
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.error("sentence-transformers not available. Install with: pip install sentence-transformers")
            return False
        
        if self.model_loaded:
            return True
        
        try:
            logger.info(f"Loading embedding model: {self.config.model_name}")
            start_time = time.time()
            
            self.model = SentenceTransformer(
                self.config.model_name,
                cache_folder=self.config.cache_dir
            )
            
            # Set max sequence length
            if hasattr(self.model, 'max_seq_length'):
                self.model.max_seq_length = self.config.max_seq_length
            
            load_time = time.time() - start_time
            logger.info(f"✅ Model loaded successfully in {load_time:.1f}s")
            logger.info(f"   Model dimension: {self.model.get_sentence_embedding_dimension()}")
            
            self.model_loaded = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            return False
    
    def embed_texts(self, texts: Union[str, List[str]]) -> EmbeddingResult:
        """
        Generate embeddings for text(s)
        
        Args:
            texts: Single text or list of texts to embed
            
        Returns:
            EmbeddingResult with embeddings and metadata
        """
        if not self._load_model():
            raise RuntimeError("Failed to load embedding model")
        
        # Ensure texts is a list
        if isinstance(texts, str):
            texts = [texts]
        
        if not texts:
            raise ValueError("No texts provided for embedding")
        
        logger.info(f"Generating embeddings for {len(texts)} texts")
        start_time = time.time()
        
        try:
            # Generate embeddings
            embeddings = self.model.encode(
                texts,
                batch_size=self.config.batch_size,
                normalize_embeddings=self.config.normalize_embeddings,
                show_progress_bar=len(texts) > 10
            )
            
            processing_time = time.time() - start_time
            
            logger.info(f"✅ Generated embeddings in {processing_time:.1f}s")
            logger.info(f"   Shape: {embeddings.shape}")
            logger.info(f"   Avg processing: {processing_time/len(texts)*1000:.1f}ms per text")
            
            return EmbeddingResult(
                embeddings=embeddings,
                texts=texts,
                processing_time=processing_time,
                model_name=self.config.model_name,
                embedding_dim=embeddings.shape[1]
            )
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise
    
    def embed_descriptions(self, descriptions: Dict[str, str]) -> Dict[str, EmbeddingResult]:
        """
        Generate embeddings for multiple descriptions
        
        Args:
            descriptions: Dictionary mapping asset IDs to descriptions
            
        Returns:
            Dictionary mapping asset IDs to EmbeddingResults
        """
        logger.info(f"Embedding {len(descriptions)} descriptions")
        
        results = {}
        
        # Batch process for efficiency
        asset_ids = list(descriptions.keys())
        texts = list(descriptions.values())
        
        # Generate embeddings for all texts at once
        embedding_result = self.embed_texts(texts)
        
        # Split results back to individual assets
        for i, asset_id in enumerate(asset_ids):
            results[asset_id] = EmbeddingResult(
                embeddings=embedding_result.embeddings[i:i+1],
                texts=[embedding_result.texts[i]],
                processing_time=embedding_result.processing_time / len(texts),
                model_name=embedding_result.model_name,
                embedding_dim=embedding_result.embedding_dim
            )
        
        return results
    
    def compute_similarity(self, 
                          embeddings1: np.ndarray, 
                          embeddings2: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between embeddings
        
        Args:
            embeddings1: First set of embeddings
            embeddings2: Second set of embeddings
            
        Returns:
            Similarity matrix
        """
        # Ensure embeddings are normalized
        if not self.config.normalize_embeddings:
            embeddings1 = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
            embeddings2 = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)
        
        # Compute cosine similarity
        similarity = np.dot(embeddings1, embeddings2.T)
        return similarity
    
    def get_model_info(self) -> Dict[str, any]:
        """Get information about the loaded model"""
        
        if not self.model_loaded:
            return {
                "model_name": self.config.model_name,
                "loaded": False
            }
        
        return {
            "model_name": self.config.model_name,
            "loaded": True,
            "embedding_dimension": self.model.get_sentence_embedding_dimension(),
            "max_seq_length": self.config.max_seq_length,
            "normalize_embeddings": self.config.normalize_embeddings
        }

def test_text_embedder():
    """Test the text embedder functionality"""
    
    print("🧠 Testing Text Embedder")
    print("-" * 40)
    
    # Test texts in French and English
    test_texts = [
        "Cette épée magique brille d'une lueur mystique avec des runes anciennes gravées sur la lame.",
        "This magical sword glows with mystical light and has ancient runes carved on the blade.",
        "Un bouclier robuste orné de motifs celtiques, forgé dans un métal résistant.",
        "A sturdy shield adorned with Celtic patterns, forged from resistant metal."
    ]
    
    try:
        # Initialize embedder
        config = EmbeddingConfig(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            batch_size=4
        )
        
        embedder = TextEmbedder(config)
        
        # Generate embeddings
        result = embedder.embed_texts(test_texts)
        
        print(f"✅ Embeddings generated:")
        print(f"   Texts: {len(result.texts)}")
        print(f"   Shape: {result.embeddings.shape}")
        print(f"   Processing time: {result.processing_time:.2f}s")
        print(f"   Model: {result.model_name}")
        
        # Test similarity
        similarities = embedder.compute_similarity(
            result.embeddings[:2],  # French and English sword descriptions
            result.embeddings[2:]   # French and English shield descriptions
        )
        
        print(f"\n📊 Similarity Matrix:")
        print(f"   Sword (FR) vs Shield (FR): {similarities[0,0]:.3f}")
        print(f"   Sword (FR) vs Shield (EN): {similarities[0,1]:.3f}")
        print(f"   Sword (EN) vs Shield (FR): {similarities[1,0]:.3f}")
        print(f"   Sword (EN) vs Shield (EN): {similarities[1,1]:.3f}")
        
        # Cross-language similarity (should be high for same objects)
        cross_lang_sword = embedder.compute_similarity(
            result.embeddings[0:1],  # French sword
            result.embeddings[1:2]   # English sword
        )[0,0]
        
        print(f"\n🌍 Cross-language similarity:")
        print(f"   Sword (FR) vs Sword (EN): {cross_lang_sword:.3f}")
        
        if cross_lang_sword > 0.7:
            print("   ✅ Good multilingual alignment")
        else:
            print("   ⚠️ Multilingual alignment could be better")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

if __name__ == "__main__":
    test_text_embedder()
