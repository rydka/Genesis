#!/usr/bin/env python3
"""
Fix Embeddings - Regenerate embeddings for existing descriptions
"""

import logging

def fix_embeddings():
    """Fix the embeddings issue"""
    
    print("🔧 Fixing Embeddings Issue")
    print("=" * 50)
    
    try:
        from fantasy_rag.database.chromadb_manager import ChromaDBManager, DatabaseConfig
        from fantasy_rag.embeddings.text_embedder import TextEmbedder, EmbeddingConfig
        
        # Initialize database manager
        config = DatabaseConfig(db_path="ressources/data/fantasy_rag_db")
        db_manager = ChromaDBManager(config)
        
        if not db_manager.initialize():
            print("❌ Failed to initialize database manager")
            return False
        
        print("✅ Database manager initialized")
        
        # Get descriptions collection
        descriptions_collection = db_manager.collections.get("asset_descriptions")
        if not descriptions_collection:
            print("❌ No descriptions collection found")
            return False
        
        # Get all descriptions
        all_docs = descriptions_collection.get(include=["documents", "metadatas"])
        
        if not all_docs['documents']:
            print("❌ No descriptions found")
            return False
        
        print(f"📝 Found {len(all_docs['documents'])} descriptions")
        
        # Show current description
        for i, doc in enumerate(all_docs['documents']):
            metadata = all_docs['metadatas'][i] if all_docs['metadatas'] else {}
            print(f"   {i+1}. Language: {metadata.get('language', 'unknown')}")
            print(f"      Text: {doc[:100]}...")
        
        # Initialize text embedder
        print("\n🧠 Initializing text embedder...")
        embedder = TextEmbedder(EmbeddingConfig())
        
        # Generate embeddings
        print("🔄 Generating embeddings...")
        texts = all_docs['documents']
        embedding_result = embedder.embed_texts(texts)
        
        if embedding_result.embeddings is None:
            print("❌ Failed to generate embeddings")
            return False
        
        print(f"✅ Generated embeddings: {embedding_result.embeddings.shape}")
        
        # Update collection with embeddings
        print("💾 Updating collection with embeddings...")

        # Generate IDs for the documents
        ids = [f"desc_{i}" for i in range(len(all_docs['documents']))]

        descriptions_collection.upsert(
            ids=ids,
            documents=all_docs['documents'],
            metadatas=all_docs['metadatas'],
            embeddings=embedding_result.embeddings.tolist()
        )
        
        print("✅ Collection updated with embeddings")
        
        # Test search
        print("\n🔍 Testing search after fix...")
        
        from fantasy_rag.embeddings.multilingual_search import MultilingualSearchEngine, MultilingualSearchQuery
        from fantasy_rag.description.prompt_templates import DescriptionLanguage
        
        search_engine = MultilingualSearchEngine(
            db_path="ressources/data/fantasy_rag_db",
            collection_name="asset_descriptions"
        )
        
        if not search_engine.initialize():
            print("❌ Failed to initialize search engine")
            return False
        
        # Test queries
        test_queries = [
            ("épée", DescriptionLanguage.FRENCH),
            ("sword", DescriptionLanguage.ENGLISH),
            ("analyze", DescriptionLanguage.ENGLISH),  # Word from the description
            ("fantastical", DescriptionLanguage.ENGLISH)  # Word from the description
        ]
        
        for query_text, language in test_queries:
            print(f"\n   Testing: '{query_text}' ({language.value})")
            
            query = MultilingualSearchQuery(
                text=query_text,
                source_language=language,
                max_results=5,
                min_similarity=0.0
            )
            
            try:
                result = search_engine.search_multilingual(query)
                print(f"   ✅ Results: {result.total_found}")
                
                if result.results:
                    best = result.results[0]
                    print(f"      Best match: {best.asset_id} (similarity: {best.similarity_score:.3f})")
                    print(f"      Text: {best.description[:80]}...")
                else:
                    print("      No results found")
                    
            except Exception as e:
                print(f"   ❌ Search error: {str(e)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    
    logging.basicConfig(level=logging.WARNING)
    
    print("🔧 Fantasy RAG Embeddings Fix")
    print("=" * 60)
    
    if fix_embeddings():
        print("\n🎉 Embeddings fixed successfully!")
        print("   Try searching again in the web interface:")
        print("   • 'sword' or 'épée'")
        print("   • 'analyze' or 'fantastical'")
        print("   • 'prince' or 'royal'")
        print("\n🌐 Web interface: http://localhost:5000")
        return 0
    else:
        print("\n❌ Failed to fix embeddings")
        return 1

if __name__ == "__main__":
    exit(main())
