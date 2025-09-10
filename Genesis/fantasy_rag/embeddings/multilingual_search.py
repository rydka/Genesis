#!/usr/bin/env python3
"""
Multilingual Search Engine for Fantasy Assets

This module extends the similarity search with advanced multilingual capabilities,
cross-language search, and language-aware ranking.
"""

import logging
from typing import List, Dict, Optional, Union, Tuple, Any
from dataclasses import dataclass
from pathlib import Path

from .similarity_search import SimilaritySearchEngine, SearchQuery, SearchResponse, SearchResult
from .text_embedder import EmbeddingConfig
from ..description.multilingual_handler import MultilingualHandler
from ..description.prompt_templates import DescriptionLanguage

logger = logging.getLogger(__name__)

@dataclass
class MultilingualSearchQuery(SearchQuery):
    """Extended search query with multilingual support"""
    source_language: Optional[DescriptionLanguage] = None
    target_languages: List[DescriptionLanguage] = None
    cross_language_boost: float = 1.0
    language_preference: Optional[DescriptionLanguage] = None

class MultilingualSearchEngine(SimilaritySearchEngine):
    """
    Advanced multilingual search engine with cross-language capabilities
    """
    
    def __init__(self, 
                 db_path: str = "ressources/data/chroma_db",
                 collection_name: str = "fantasy_assets_multilingual",
                 embedding_config: EmbeddingConfig = None):
        """
        Initialize multilingual search engine
        
        Args:
            db_path: Path to ChromaDB database
            collection_name: Name of the collection
            embedding_config: Configuration for text embedder
        """
        super().__init__(db_path, collection_name, embedding_config)
        self.multilingual_handler = MultilingualHandler()
        
        logger.info("MultilingualSearchEngine initialized with cross-language support")
    
    def add_multilingual_descriptions(self, 
                                    descriptions: Dict[str, Dict[str, str]],
                                    metadata: Dict[str, Dict[str, Any]] = None) -> bool:
        """
        Add multilingual descriptions to the search index
        
        Args:
            descriptions: Dictionary mapping asset IDs to language->description mappings
            metadata: Optional metadata for each asset
            
        Returns:
            True if successful
        """
        logger.info(f"Adding multilingual descriptions for {len(descriptions)} assets")
        
        # Flatten descriptions with language information
        flat_descriptions = {}
        flat_metadata = {}
        
        for asset_id, lang_descriptions in descriptions.items():
            for language, description in lang_descriptions.items():
                # Create unique key for each language version
                doc_key = f"{asset_id}_{language}"
                flat_descriptions[doc_key] = description
                
                # Prepare metadata with language information
                doc_metadata = {
                    "language": language,
                    "original_asset_id": asset_id,
                    "is_multilingual": True
                }
                
                # Add base metadata if provided
                if metadata and asset_id in metadata:
                    doc_metadata.update(metadata[asset_id])
                
                flat_metadata[doc_key] = doc_metadata
        
        # Use parent class method to add flattened descriptions
        return self.add_descriptions(flat_descriptions, flat_metadata)
    
    def search_multilingual(self, query: MultilingualSearchQuery) -> SearchResponse:
        """
        Perform multilingual search with language-aware ranking
        
        Args:
            query: Multilingual search query
            
        Returns:
            SearchResponse with language-aware results
        """
        logger.info(f"Multilingual search: '{query.text[:50]}...'")
        
        # Detect query language if not specified
        if not query.source_language:
            detection = self.multilingual_handler.detect_language(query.text)
            query.source_language = detection.detected_language
            logger.info(f"Detected query language: {query.source_language.value}")
        
        # Perform base search
        base_query = SearchQuery(
            text=query.text,
            language=query.source_language.value,
            category_filter=query.category_filter,
            max_results=query.max_results * 2,  # Get more results for reranking
            min_similarity=query.min_similarity,
            metadata_filters=query.metadata_filters
        )
        
        response = self.search(base_query)
        
        # Apply multilingual reranking
        reranked_results = self._rerank_multilingual_results(
            response.results, 
            query
        )
        
        # Limit to requested number of results
        final_results = reranked_results[:query.max_results]
        
        # Update response
        response.results = final_results
        response.total_found = len(final_results)
        response.search_metadata.update({
            "query_language": query.source_language.value,
            "multilingual_reranking": True,
            "cross_language_boost": query.cross_language_boost
        })
        
        return response
    
    def _rerank_multilingual_results(self, 
                                   results: List[SearchResult],
                                   query: MultilingualSearchQuery) -> List[SearchResult]:
        """
        Rerank results based on multilingual preferences
        
        Args:
            results: Original search results
            query: Multilingual query configuration
            
        Returns:
            Reranked results
        """
        if not results:
            return results
        
        # Group results by original asset ID
        asset_groups = {}
        for result in results:
            original_id = result.metadata.get('original_asset_id', result.asset_id)
            if original_id not in asset_groups:
                asset_groups[original_id] = []
            asset_groups[original_id].append(result)
        
        # Select best result for each asset
        final_results = []
        
        for asset_id, asset_results in asset_groups.items():
            best_result = self._select_best_language_result(asset_results, query)
            if best_result:
                final_results.append(best_result)
        
        # Sort by adjusted similarity score
        final_results.sort(key=lambda x: x.similarity_score, reverse=True)
        
        return final_results
    
    def _select_best_language_result(self, 
                                   asset_results: List[SearchResult],
                                   query: MultilingualSearchQuery) -> Optional[SearchResult]:
        """
        Select the best language version for an asset
        
        Args:
            asset_results: Results for the same asset in different languages
            query: Multilingual query configuration
            
        Returns:
            Best result or None
        """
        if not asset_results:
            return None
        
        if len(asset_results) == 1:
            return asset_results[0]
        
        # Score each result based on language preferences
        scored_results = []
        
        for result in asset_results:
            score = result.similarity_score
            result_language = result.metadata.get('language', 'unknown')
            
            # Apply language preference boost
            if query.language_preference:
                if result_language == query.language_preference.value:
                    score *= 1.2  # Boost preferred language
                elif result_language == query.source_language.value:
                    score *= 1.1  # Slight boost for same language as query
            
            # Apply cross-language boost
            if result_language != query.source_language.value:
                score *= query.cross_language_boost
            
            scored_results.append((score, result))
        
        # Return result with highest adjusted score
        scored_results.sort(key=lambda x: x[0], reverse=True)
        return scored_results[0][1]
    
    def search_cross_language(self, 
                            query_text: str,
                            source_language: DescriptionLanguage,
                            target_language: DescriptionLanguage,
                            max_results: int = 10) -> SearchResponse:
        """
        Search for assets in one language using a query in another language
        
        Args:
            query_text: Search query text
            source_language: Language of the query
            target_language: Language of desired results
            max_results: Maximum number of results
            
        Returns:
            SearchResponse with cross-language results
        """
        logger.info(f"Cross-language search: {source_language.value} -> {target_language.value}")
        
        # Create multilingual query
        query = MultilingualSearchQuery(
            text=query_text,
            source_language=source_language,
            language_preference=target_language,
            max_results=max_results,
            cross_language_boost=1.0,  # No penalty for cross-language
            metadata_filters={"language": target_language.value}
        )
        
        return self.search_multilingual(query)
    
    def get_language_statistics(self) -> Dict[str, Any]:
        """Get statistics about languages in the collection"""
        
        if not self.initialized:
            return {"initialized": False}
        
        try:
            # Get all documents with metadata
            all_docs = self.collection.get(include=["metadatas"])
            
            if not all_docs['metadatas']:
                return {"initialized": True, "total_documents": 0}
            
            # Analyze language distribution
            language_stats = {}
            asset_language_coverage = {}
            
            for metadata in all_docs['metadatas']:
                language = metadata.get('language', 'unknown')
                original_id = metadata.get('original_asset_id', metadata.get('asset_id', 'unknown'))
                
                # Count by language
                language_stats[language] = language_stats.get(language, 0) + 1
                
                # Track language coverage per asset
                if original_id not in asset_language_coverage:
                    asset_language_coverage[original_id] = set()
                asset_language_coverage[original_id].add(language)
            
            # Calculate coverage statistics
            coverage_stats = {}
            for asset_id, languages in asset_language_coverage.items():
                lang_count = len(languages)
                coverage_stats[lang_count] = coverage_stats.get(lang_count, 0) + 1
            
            return {
                "initialized": True,
                "total_documents": len(all_docs['metadatas']),
                "language_distribution": language_stats,
                "unique_assets": len(asset_language_coverage),
                "language_coverage": coverage_stats,
                "multilingual_assets": sum(1 for langs in asset_language_coverage.values() if len(langs) > 1)
            }
            
        except Exception as e:
            logger.error(f"Error getting language statistics: {str(e)}")
            return {"initialized": True, "error": str(e)}
