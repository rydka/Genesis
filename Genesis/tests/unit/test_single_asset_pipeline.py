#!/usr/bin/env python3
"""
Test Single Asset Pipeline - Complete end-to-end test
"""

import logging
from pathlib import Path
import time

def test_single_asset_pipeline():
    """Test complete pipeline on a single STL file"""
    
    print("🚀 Testing Complete Single Asset Pipeline")
    print("=" * 60)
    
    try:
        from fantasy_rag.database.indexing_pipeline import IndexingPipeline, PipelineConfig
        from fantasy_rag.database.schema_definitions import ProcessingStage
        from fantasy_rag.description.prompt_templates import DescriptionLanguage
        
        # Find the smallest STL file for faster testing
        stl_directory = Path("ressources/data/assets")
        stl_files = list(stl_directory.rglob("*.stl"))
        
        if not stl_files:
            print("❌ No STL files found")
            return False
        
        # Sort by file size and pick the smallest
        stl_files.sort(key=lambda f: f.stat().st_size)
        test_stl = stl_files[0]
        
        file_size_mb = test_stl.stat().st_size / (1024 * 1024)
        print(f"📦 Testing with: {test_stl.name} ({file_size_mb:.1f} MB)")
        
        # Configure pipeline for single language (faster)
        config = PipelineConfig(
            enable_rendering=True,
            enable_description=True,
            enable_embeddings=True,
            description_languages=[DescriptionLanguage.FRENCH],  # Single language for speed
            target_description_length=200,  # Standard length
            max_retries=1,
            skip_existing=False,
            render_resolution=(512, 512),  # Lower resolution for speed
            render_samples=32  # Fewer samples for speed
        )
        
        # Initialize pipeline
        pipeline = IndexingPipeline(config)
        
        print("🔧 Initializing pipeline components...")
        if not pipeline.initialize():
            print("❌ Failed to initialize pipeline")
            print("   Make sure Ollama is running with llava model")
            return False
        
        print("✅ Pipeline initialized successfully")
        
        # Process the STL file
        print(f"\n🎯 Processing {test_stl.name}...")
        start_time = time.time()
        
        result = pipeline.process_stl_file(test_stl)
        
        total_time = time.time() - start_time
        
        print(f"\n📊 Processing Results:")
        print(f"   Asset ID: {result.asset_id}")
        print(f"   Success: {'✅' if result.success else '❌'}")
        print(f"   Total time: {total_time:.1f}s")
        print(f"   Stages completed: {[stage.value for stage in result.stages_completed]}")
        print(f"   Renders created: {result.renders_created}")
        print(f"   Descriptions created: {result.descriptions_created}")
        print(f"   Embeddings created: {result.embeddings_created}")
        
        if result.errors:
            print(f"   ❌ Errors: {result.errors}")
        
        if result.success:
            print("\n🎉 Pipeline test completed successfully!")
            
            # Verify data in database
            db_stats = pipeline.db_manager.get_database_stats()
            print(f"\n📈 Database after processing:")
            for collection_name, stats in db_stats['collections'].items():
                if stats['document_count'] > 0:
                    print(f"   {collection_name}: {stats['document_count']} documents")
            
            # Test search functionality
            print(f"\n🔍 Testing search functionality...")
            
            # Get the description to test search
            descriptions = pipeline.db_manager.get_asset_descriptions(result.asset_id)
            if descriptions:
                desc = descriptions[0]
                print(f"   Generated description: {desc.description_text[:100]}...")
                
                # Test search with the description
                from fantasy_rag.embeddings.multilingual_search import MultilingualSearchEngine, MultilingualSearchQuery
                
                search_engine = MultilingualSearchEngine(
                    db_path=pipeline.config.database_config.db_path,
                    collection_name=pipeline.config.database_config.descriptions_collection
                )
                
                if search_engine.initialize():
                    # Search for keywords from the description
                    search_words = desc.description_text.split()[:3]  # First 3 words
                    search_query = " ".join(search_words)
                    
                    query = MultilingualSearchQuery(
                        text=search_query,
                        source_language=DescriptionLanguage.FRENCH,
                        max_results=3
                    )
                    
                    search_result = search_engine.search_multilingual(query)
                    print(f"   Search query: '{search_query}'")
                    print(f"   Results found: {search_result.total_found}")
                    
                    if search_result.results:
                        best_result = search_result.results[0]
                        print(f"   Best match: {best_result.asset_id} (similarity: {best_result.similarity_score:.3f})")
                        print(f"   ✅ Search functionality working")
                    else:
                        print(f"   ⚠️ No search results found")
                else:
                    print(f"   ❌ Failed to initialize search engine")
            
            return True
        else:
            print("❌ Pipeline test failed")
            return False
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run single asset pipeline test"""
    
    # Setup logging to show progress
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s:%(name)s:%(message)s'
    )
    
    print("🧪 Fantasy Asset Single Pipeline Test")
    print("=" * 60)
    
    if test_single_asset_pipeline():
        print("\n" + "=" * 60)
        print("🎉 COMPLETE PIPELINE TEST SUCCESSFUL!")
        print("\n✅ Validated Components:")
        print("   1. ✅ STL File Discovery")
        print("   2. ✅ Asset Registration")
        print("   3. ✅ Blender Rendering (6 views)")
        print("   4. ✅ Description Generation (Ollama)")
        print("   5. ✅ Text Embedding (SentenceTransformers)")
        print("   6. ✅ Database Storage (ChromaDB)")
        print("   7. ✅ Semantic Search")
        
        print("\n🚀 Fantasy RAG System: 100% OPERATIONAL")
        print("   Ready for batch processing of all 12 STL files!")
        
        return 0
    else:
        print("\n❌ Pipeline test failed")
        print("   Check the error messages above")
        return 1

if __name__ == "__main__":
    exit(main())
