#!/usr/bin/env python3
"""
Test script to reproduce the "épée" search error
"""

import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

def test_epee_search():
    """Test searching for 'épée' to reproduce the error"""
    
    try:
        from fantasy_rag.embeddings.multilingual_search import MultilingualSearchEngine, MultilingualSearchQuery
        from fantasy_rag.description.prompt_templates import DescriptionLanguage
        
        print("🔍 Testing 'épée' search...")
        
        # Initialize search engine with the same configuration as the web app
        search_engine = MultilingualSearchEngine(
            db_path="ressources/data/fantasy_rag_db",
            collection_name="asset_descriptions"
        )
        
        if not search_engine.initialize():
            print("❌ Failed to initialize search engine")
            return False
        
        print("✅ Search engine initialized")
        
        # Check collection status
        stats = search_engine.get_collection_stats()
        print(f"📊 Collection stats: {stats}")
        
        # Create the same query that was failing
        search_query = MultilingualSearchQuery(
            text="épée",
            source_language=DescriptionLanguage.FRENCH,
            max_results=10,
            min_similarity=-1.0  # Accept negative similarities like in the web app
        )
        
        print(f"🔎 Searching for: '{search_query.text}'")
        
        # Perform search
        try:
            search_result = search_engine.search_multilingual(search_query)
            
            print(f"✅ Search completed successfully!")
            print(f"   Query: {search_result.query}")
            print(f"   Total found: {search_result.total_found}")
            print(f"   Processing time: {search_result.processing_time:.3f}s")
            
            for i, result in enumerate(search_result.results[:3]):
                print(f"   {i+1}. {result.asset_id} (sim: {result.similarity_score:.3f})")
                print(f"      {result.description[:80]}...")
            
            return True
            
        except Exception as e:
            print(f"❌ Search failed with error: {str(e)}")
            logger.error(f"Search error: {str(e)}", exc_info=True)
            return False
        
    except ImportError as e:
        print(f"❌ Import error: {str(e)}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return False

def test_web_app_search():
    """Test the web app search endpoint directly"""

    try:
        from fantasy_rag.web import app
        import json

        print("\n🌐 Testing web app search...")

        # Initialize components like the web app does
        if not app.initialize_components():
            print("❌ Failed to initialize web app components")
            return False

        print("✅ Web app components initialized")

        # Check if search_engine is properly set
        print(f"🔍 Search engine type: {type(app.search_engine)}")
        if app.search_engine is None:
            print("❌ Search engine is None after initialization")
            return False

        # Simulate the search request data
        data = {
            'query': 'épée',
            'language': 'français',
            'max_results': 10,
            'min_similarity': -1.0
        }

        print(f"🔎 Simulating web search for: '{data['query']}'")

        # Test the search logic directly
        from fantasy_rag.description.prompt_templates import DescriptionLanguage
        from fantasy_rag.embeddings.multilingual_search import MultilingualSearchQuery

        # Convert language string to enum
        lang_enum = DescriptionLanguage.FRENCH if data['language'].lower() in ['français', 'french', 'fr'] else DescriptionLanguage.ENGLISH

        # Create search query
        search_query = MultilingualSearchQuery(
            text=data['query'],
            source_language=lang_enum,
            max_results=data['max_results'],
            min_similarity=data['min_similarity']
        )

        # Perform search using the app's search engine
        search_result = app.search_engine.search_multilingual(search_query)

        print(f"✅ Web app search completed successfully!")
        print(f"   Query: {search_result.query}")
        print(f"   Total found: {search_result.total_found}")
        print(f"   Processing time: {search_result.processing_time:.3f}s")

        return True

    except Exception as e:
        print(f"❌ Web app search failed: {str(e)}")
        logger.error(f"Web app search error: {str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    print("🧪 Testing 'épée' search error reproduction")
    print("=" * 60)
    
    success1 = test_epee_search()
    success2 = test_web_app_search()
    
    print("\n" + "=" * 60)
    if success1 and success2:
        print("🎉 All tests passed! The 'épée' search error has been fixed.")
    else:
        print("❌ Some tests failed. Check the logs above for details.")
        sys.exit(1)
