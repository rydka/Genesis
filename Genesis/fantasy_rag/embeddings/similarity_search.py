#!/usr/bin/env python3
"""
Similarity Search Engine for Fantasy Assets

This module implements semantic search using ChromaDB for vector storage
and similarity matching of fantasy asset descriptions.
"""

import logging
import numpy as np
from typing import List, Dict, Optional, Union, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
import time
import uuid

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

from .text_embedder import TextEmbedder, EmbeddingConfig

logger = logging.getLogger(__name__)

@dataclass
class SearchQuery:
    """Search query configuration"""
    text: str
    language: str = "auto"
    category_filter: Optional[str] = None
    max_results: int = 10
    min_similarity: float = 0.0
    metadata_filters: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SearchResult:
    """Single search result"""
    asset_id: str
    description: str
    similarity_score: float
    metadata: Dict[str, Any]
    distance: float

@dataclass
class SearchResponse:
    """Complete search response"""
    query: str
    results: List[SearchResult]
    total_found: int
    processing_time: float
    search_metadata: Dict[str, Any] = field(default_factory=dict)

class SimilaritySearchEngine:
    """
    Semantic search engine using ChromaDB and SentenceTransformers
    """
    
    def __init__(self, 
                 db_path: str = "ressources/data/chroma_db",
                 collection_name: str = "fantasy_assets",
                 embedding_config: EmbeddingConfig = None):
        """
        Initialize search engine
        
        Args:
            db_path: Path to ChromaDB database
            collection_name: Name of the collection
            embedding_config: Configuration for text embedder
        """
        self.db_path = Path(db_path)
        self.collection_name = collection_name
        self.embedding_config = embedding_config or EmbeddingConfig()
        
        self.client = None
        self.collection = None
        self.embedder = None
        self.initialized = False
        
        logger.info(f"SimilaritySearchEngine initialized:")
        logger.info(f"  Database: {self.db_path}")
        logger.info(f"  Collection: {self.collection_name}")
    
    def initialize(self) -> bool:
        """Initialize ChromaDB and embedder"""
        
        if not CHROMADB_AVAILABLE:
            logger.error("ChromaDB not available. Install with: pip install chromadb")
            return False
        
        try:
            # Create database directory
            self.db_path.mkdir(parents=True, exist_ok=True)
            
            # Initialize ChromaDB client
            logger.info("Initializing ChromaDB client...")
            self.client = chromadb.PersistentClient(
                path=str(self.db_path),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Initialize text embedder
            logger.info("Initializing text embedder...")
            self.embedder = TextEmbedder(self.embedding_config)
            
            # Get or create collection
            try:
                self.collection = self.client.get_collection(
                    name=self.collection_name
                )
                logger.info(f"✅ Loaded existing collection: {self.collection_name}")
                logger.info(f"   Documents: {self.collection.count()}")
            except:
                # Create new collection
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Fantasy asset descriptions with multilingual support"}
                )
                logger.info(f"✅ Created new collection: {self.collection_name}")
            
            self.initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize search engine: {str(e)}")
            return False
    
    def add_descriptions(self, 
                        descriptions: Dict[str, str],
                        metadata: Dict[str, Dict[str, Any]] = None) -> bool:
        """
        Add descriptions to the search index
        
        Args:
            descriptions: Dictionary mapping asset IDs to descriptions
            metadata: Optional metadata for each asset
            
        Returns:
            True if successful
        """
        if not self.initialized and not self.initialize():
            return False
        
        logger.info(f"Adding {len(descriptions)} descriptions to search index")
        start_time = time.time()
        
        try:
            # Generate embeddings
            embedding_results = self.embedder.embed_descriptions(descriptions)
            
            # Prepare data for ChromaDB
            ids = []
            embeddings = []
            documents = []
            metadatas = []
            
            for asset_id, description in descriptions.items():
                # Generate unique ID for ChromaDB
                doc_id = f"{asset_id}_{uuid.uuid4().hex[:8]}"
                ids.append(doc_id)
                
                # Get embedding
                embedding_result = embedding_results[asset_id]
                embeddings.append(embedding_result.embeddings[0].tolist())
                
                # Store description
                documents.append(description)
                
                # Prepare metadata
                doc_metadata = {
                    "asset_id": asset_id,
                    "description_length": len(description),
                    "word_count": len(description.split()),
                    "embedding_model": embedding_result.model_name,
                    "indexed_at": time.time()
                }
                
                # Add custom metadata if provided
                if metadata and asset_id in metadata:
                    doc_metadata.update(metadata[asset_id])
                
                metadatas.append(doc_metadata)
            
            # Add to ChromaDB
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            
            processing_time = time.time() - start_time
            
            logger.info(f"✅ Added {len(descriptions)} descriptions in {processing_time:.1f}s")
            logger.info(f"   Total documents in collection: {self.collection.count()}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding descriptions: {str(e)}")
            return False
    
    def search(self, query: SearchQuery) -> SearchResponse:
        """
        Search for similar descriptions
        
        Args:
            query: Search query configuration
            
        Returns:
            SearchResponse with results
        """
        if not self.initialized and not self.initialize():
            raise RuntimeError("Search engine not initialized")
        
        logger.info(f"Searching for: '{query.text[:50]}...'")
        start_time = time.time()
        
        try:
            # Generate query embedding
            query_embedding = self.embedder.embed_texts([query.text])

            # Check if embedding generation was successful
            if query_embedding is None or query_embedding.embeddings is None:
                logger.error("Failed to generate query embedding")
                raise RuntimeError("Query embedding generation failed")

            if len(query_embedding.embeddings) == 0:
                logger.error("Empty embedding result")
                raise RuntimeError("Empty embedding result")

            # Prepare ChromaDB query
            where_clause = {}

            # Add category filter
            if query.category_filter:
                where_clause["category"] = query.category_filter

            # Add custom metadata filters
            where_clause.update(query.metadata_filters)

            # Search in ChromaDB
            chroma_results = self.collection.query(
                query_embeddings=[query_embedding.embeddings[0].tolist()],
                n_results=query.max_results,
                where=where_clause if where_clause else None,
                include=["documents", "metadatas", "distances"]
            )
            
            # Process results
            results = []

            # Check if ChromaDB returned valid results
            if (chroma_results is None or
                'ids' not in chroma_results or
                chroma_results['ids'] is None or
                len(chroma_results['ids']) == 0 or
                chroma_results['ids'][0] is None):
                logger.warning("No results returned from ChromaDB")
            else:
                # Validate all required fields are present
                required_fields = ['distances', 'metadatas', 'documents']
                for field in required_fields:
                    if (field not in chroma_results or
                        chroma_results[field] is None or
                        len(chroma_results[field]) == 0 or
                        chroma_results[field][0] is None):
                        logger.error(f"Missing or invalid field in ChromaDB results: {field}")
                        break
                else:
                    # Process valid results
                    for i in range(len(chroma_results['ids'][0])):
                        try:
                            # Calculate similarity from distance
                            distance = chroma_results['distances'][0][i]
                            similarity = 1.0 - distance  # Convert distance to similarity

                            # Apply similarity threshold
                            if similarity >= query.min_similarity:
                                # Validate metadata has required asset_id
                                metadata = chroma_results['metadatas'][0][i]
                                if metadata is None or 'asset_id' not in metadata:
                                    logger.warning(f"Missing asset_id in metadata for result {i}")
                                    continue

                                result = SearchResult(
                                    asset_id=metadata['asset_id'],
                                    description=chroma_results['documents'][0][i],
                                    similarity_score=similarity,
                                    metadata=metadata,
                                    distance=distance
                                )
                                results.append(result)
                        except (IndexError, KeyError, TypeError) as e:
                            logger.warning(f"Error processing result {i}: {str(e)}")
                            continue
            
            processing_time = time.time() - start_time
            
            logger.info(f"✅ Found {len(results)} results in {processing_time:.2f}s")
            
            return SearchResponse(
                query=query.text,
                results=results,
                total_found=len(results),
                processing_time=processing_time,
                search_metadata={
                    "collection_size": self.collection.count(),
                    "embedding_model": self.embedding_config.model_name,
                    "filters_applied": bool(where_clause)
                }
            )
            
        except Exception as e:
            logger.error(f"Error during search: {str(e)}")
            raise
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        
        if not self.initialized:
            return {"initialized": False}
        
        try:
            count = self.collection.count()
            
            # Get sample metadata to analyze
            sample = self.collection.get(limit=min(100, count), include=["metadatas"])
            
            stats = {
                "initialized": True,
                "total_documents": count,
                "collection_name": self.collection_name,
                "database_path": str(self.db_path)
            }
            
            if sample['metadatas']:
                # Analyze categories
                categories = {}
                languages = {}
                
                for metadata in sample['metadatas']:
                    if 'category' in metadata:
                        cat = metadata['category']
                        categories[cat] = categories.get(cat, 0) + 1
                    
                    if 'language' in metadata:
                        lang = metadata['language']
                        languages[lang] = languages.get(lang, 0) + 1
                
                stats.update({
                    "categories": categories,
                    "languages": languages,
                    "sample_size": len(sample['metadatas'])
                })
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {"initialized": True, "error": str(e)}
    
    def clear_collection(self) -> bool:
        """Clear all documents from the collection"""
        
        if not self.initialized:
            return False
        
        try:
            # Delete and recreate collection
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Fantasy asset descriptions with multilingual support"}
            )
            
            logger.info(f"✅ Cleared collection: {self.collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing collection: {str(e)}")
            return False
