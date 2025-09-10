#!/usr/bin/env python3
"""
Debug Search Issue - Investigate why search returns no results
"""

import logging
from pathlib import Path

def debug_search_issue():
    """Debug the search functionality"""
    
    print("🔍 Debug Search Issue")
    print("=" * 50)
    
    try:
        from fantasy_rag.database.chromadb_manager import ChromaDBManager, DatabaseConfig
        from fantasy_rag.embeddings.multilingual_search import MultilingualSearchEngine, MultilingualSearchQuery
        from fantasy_rag.description.prompt_templates import DescriptionLanguage
        
        # Initialize database manager
        config = DatabaseConfig(
            db_path="ressources/data/fantasy_rag_db"
        )
        
        db_manager = ChromaDBManager(config)
        if not db_manager.initialize():
            print("❌ Failed to initialize database manager")
            return False
        
        print("✅ Database manager initialized")
        
        # Check database contents
        stats = db_manager.get_database_stats()
        print(f"\n📊 Database Statistics:")
        for collection_name, collection_stats in stats['collections'].items():
            print(f"   {collection_name}: {collection_stats['document_count']} documents")
        
        # Check descriptions specifically
        print(f"\n📝 Checking descriptions...")
        
        # Get all assets
        from fantasy_rag.database.schema_definitions import AssetStatus
        indexed_assets = db_manager.get_assets_by_status(AssetStatus.INDEXED)
        
        print(f"   Found {len(indexed_assets)} indexed assets")
        
        for asset in indexed_assets:
            print(f"\n📦 Asset: {asset.asset_id} - {asset.original_filename}")
            
            # Get descriptions
            descriptions = db_manager.get_asset_descriptions(asset.asset_id)
            print(f"   Descriptions: {len(descriptions)}")
            
            for desc in descriptions:
                print(f"   - Language: {desc.language}")
                print(f"   - Text preview: {desc.description_text[:100]}...")
                print(f"   - Word count: {desc.word_count}")
        
        # Test search engine initialization
        print(f"\n🔍 Testing search engine...")
        
        search_engine = MultilingualSearchEngine(
            db_path="ressources/data/fantasy_rag_db",
            collection_name="asset_descriptions"
        )
        
        if not search_engine.initialize():
            print("❌ Failed to initialize search engine")
            return False
        
        print("✅ Search engine initialized")
        
        # Get search engine stats
        search_stats = search_engine.get_language_statistics()
        print(f"   Total documents: {search_stats.get('total_documents', 0)}")
        print(f"   Languages: {search_stats.get('language_distribution', {})}")
        
        # Test direct ChromaDB query
        print(f"\n🧪 Testing direct ChromaDB queries...")
        
        collection = db_manager.collections.get("asset_descriptions")
        if collection:
            # Get all documents
            all_docs = collection.get(include=["documents", "metadatas"])
            print(f"   Direct query found {len(all_docs['documents']) if all_docs['documents'] else 0} documents")
            
            if all_docs['documents']:
                for i, doc in enumerate(all_docs['documents']):
                    metadata = all_docs['metadatas'][i] if all_docs['metadatas'] else {}
                    print(f"   Doc {i+1}: {doc[:50]}... (lang: {metadata.get('language', 'unknown')})")
        
        # Test search queries
        print(f"\n🔎 Testing search queries...")
        
        test_queries = [
            ("épée", DescriptionLanguage.FRENCH),
            ("prince", DescriptionLanguage.FRENCH),
            ("sword", DescriptionLanguage.ENGLISH),
            ("royal", DescriptionLanguage.ENGLISH)
        ]
        
        for query_text, language in test_queries:
            print(f"\n   Testing: '{query_text}' ({language.value})")
            
            query = MultilingualSearchQuery(
                text=query_text,
                source_language=language,
                max_results=5,
                min_similarity=0.0  # Accept any similarity
            )
            
            try:
                result = search_engine.search_multilingual(query)
                print(f"   Results: {result.total_found}")
                print(f"   Processing time: {result.processing_time:.3f}s")
                
                for i, res in enumerate(result.results[:2]):
                    print(f"     {i+1}. {res.asset_id} (sim: {res.similarity_score:.3f})")
                    print(f"        {res.description[:80]}...")
                    
            except Exception as e:
                print(f"   ❌ Search error: {str(e)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def fix_search_issue():
    """Try to fix the search issue by re-indexing"""
    
    print("\n🔧 Attempting to fix search issue...")
    
    try:
        from fantasy_rag.database.chromadb_manager import ChromaDBManager, DatabaseConfig
        from fantasy_rag.embeddings.text_embedder import TextEmbedder, EmbeddingConfig
        
        # Initialize components
        config = DatabaseConfig(db_path="ressources/data/fantasy_rag_db")
        db_manager = ChromaDBManager(config)
        
        if not db_manager.initialize():
            print("❌ Failed to initialize database manager")
            return False
        
        # Get all descriptions
        print("📝 Getting all descriptions...")
        
        # We need to manually query the descriptions collection
        collection = db_manager.collections.get("asset_descriptions")
        if not collection:
            print("❌ No descriptions collection found")
            return False
        
        all_docs = collection.get(include=["documents", "metadatas", "embeddings"])
        
        if not all_docs['documents']:
            print("❌ No descriptions found in collection")
            return False
        
        print(f"✅ Found {len(all_docs['documents'])} descriptions")
        
        # Check if embeddings exist
        embeddings = all_docs.get('embeddings')
        has_embeddings = embeddings is not None and len(embeddings) > 0 and embeddings[0] is not None
        print(f"   Embeddings present: {has_embeddings}")
        print(f"   Embeddings data: {type(embeddings)} - {len(embeddings) if embeddings else 0} items")
        
        if not has_embeddings:
            print("🔄 Generating missing embeddings...")
            
            # Initialize text embedder
            embedder = TextEmbedder(EmbeddingConfig())
            
            # Generate embeddings for all descriptions
            texts = all_docs['documents']
            embedding_result = embedder.embed_texts(texts)
            
            if embedding_result.embeddings is not None:
                print(f"✅ Generated embeddings for {len(texts)} descriptions")
                
                # Update the collection with embeddings
                ids = [f"desc_{i}" for i in range(len(texts))]
                
                collection.upsert(
                    ids=ids,
                    documents=texts,
                    metadatas=all_docs['metadatas'],
                    embeddings=embedding_result.embeddings.tolist()
                )
                
                print("✅ Updated collection with embeddings")
                return True
            else:
                print("❌ Failed to generate embeddings")
                return False
        else:
            print("✅ Embeddings already present")
            return True
        
    except Exception as e:
        print(f"❌ Fix error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    
    logging.basicConfig(level=logging.WARNING)
    
    print("🔍 Fantasy RAG Search Debug")
    print("=" * 60)
    
    if debug_search_issue():
        print("\n🔧 Debug completed. Attempting fix...")
        if fix_search_issue():
            print("\n✅ Search issue potentially fixed!")
            print("   Try searching again in the web interface")
        else:
            print("\n❌ Could not fix search issue")
    else:
        print("\n❌ Debug failed")

if __name__ == "__main__":
    main()
