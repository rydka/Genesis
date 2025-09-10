#!/usr/bin/env python3
"""
Test Direct Search - Test ChromaDB search directly
"""

def test_direct_search():
    """Test ChromaDB search functionality directly"""
    
    print("🔍 Testing Direct ChromaDB Search")
    print("=" * 50)
    
    try:
        import chromadb
        from fantasy_rag.embeddings.text_embedder import TextEmbedder, EmbeddingConfig
        
        # Connect to ChromaDB
        client = chromadb.PersistentClient(path="ressources/data/fantasy_rag_db")
        collection = client.get_collection("asset_descriptions")
        
        print(f"✅ Connected to collection: {collection.name}")
        print(f"   Document count: {collection.count()}")
        
        # Get all documents
        all_docs = collection.get(include=["documents", "metadatas", "embeddings"])
        
        print(f"\n📝 Collection contents:")
        print(f"   Documents: {len(all_docs['documents']) if all_docs['documents'] else 0}")
        print(f"   Metadatas: {len(all_docs['metadatas']) if all_docs['metadatas'] else 0}")

        embeddings = all_docs.get('embeddings')
        embeddings_count = len(embeddings) if embeddings is not None else 0
        print(f"   Embeddings: {embeddings_count}")
        
        if all_docs['documents']:
            for i, doc in enumerate(all_docs['documents']):
                metadata = all_docs['metadatas'][i] if all_docs['metadatas'] else {}
                embedding = embeddings[i] if embeddings is not None and i < len(embeddings) else None
                
                print(f"\n   Document {i+1}:")
                print(f"     Text: {doc[:100]}...")
                print(f"     Language: {metadata.get('language', 'unknown')}")
                print(f"     Asset ID: {metadata.get('asset_id', 'unknown')}")

                embedding_status = 'Present' if embedding is not None else 'Missing'
                embedding_dims = len(embedding) if embedding is not None else 0
                print(f"     Embedding: {embedding_status} ({embedding_dims} dims)")
        
        # Test direct ChromaDB query
        print(f"\n🔍 Testing direct ChromaDB query...")
        
        # Initialize embedder
        embedder = TextEmbedder(EmbeddingConfig())
        
        # Test queries
        test_queries = ["épée", "sword", "analyze", "fantastical", "prince", "royal"]
        
        for query_text in test_queries:
            print(f"\n   Query: '{query_text}'")
            
            # Generate embedding for query
            query_result = embedder.embed_texts([query_text])
            
            if query_result.embeddings is not None:
                query_embedding = query_result.embeddings[0].tolist()
                
                # Search with ChromaDB
                search_results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=3,
                    include=["documents", "metadatas", "distances"]
                )
                
                print(f"     Results: {len(search_results['documents'][0]) if search_results['documents'] else 0}")
                
                if search_results['documents'] and search_results['documents'][0]:
                    for j, doc in enumerate(search_results['documents'][0]):
                        distance = search_results['distances'][0][j] if search_results['distances'] else 0
                        similarity = 1 - distance  # Convert distance to similarity
                        metadata = search_results['metadatas'][0][j] if search_results['metadatas'] else {}
                        
                        print(f"       {j+1}. Similarity: {similarity:.3f}")
                        print(f"          Asset: {metadata.get('asset_id', 'unknown')}")
                        print(f"          Text: {doc[:60]}...")
                else:
                    print(f"     No results found")
            else:
                print(f"     Failed to generate query embedding")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    
    print("🧪 Direct ChromaDB Search Test")
    print("=" * 60)
    
    if test_direct_search():
        print("\n✅ Direct search test completed")
    else:
        print("\n❌ Direct search test failed")

if __name__ == "__main__":
    main()
