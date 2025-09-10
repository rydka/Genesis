#!/usr/bin/env python3
"""
Script to debug description truncation issues
"""

import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_descriptions():
    """Debug description truncation issues"""
    
    print("🔍 Debugging Description Truncation")
    print("=" * 60)
    
    try:
        from fantasy_rag.database.chromadb_manager import ChromaDBManager, DatabaseConfig
        
        # Initialize database
        config = DatabaseConfig(db_path="ressources/data/fantasy_rag_db")
        db_manager = ChromaDBManager(config)
        
        if not db_manager.initialize():
            print("❌ Failed to initialize database")
            return False
        
        print("✅ Database initialized")
        
        # Get descriptions directly from the collection
        desc_collection = db_manager.collections[db_manager.config.descriptions_collection]
        all_docs = desc_collection.get()
        
        if not all_docs['ids']:
            print("📝 No descriptions found in database")
            return True
        
        print(f"📝 Found {len(all_docs['ids'])} descriptions")
        print()
        
        # Analyze each description
        for i, doc_id in enumerate(all_docs['ids'], 1):
            metadata = all_docs['metadatas'][i-1]
            document = all_docs['documents'][i-1]
            
            print(f"📄 Description {i}: {doc_id}")
            print(f"   Asset ID: {metadata.get('asset_id', 'N/A')}")
            print(f"   Language: {metadata.get('language', 'N/A')}")
            print(f"   Word count: {metadata.get('word_count', 'N/A')}")
            print(f"   Target length: {metadata.get('target_length', 'N/A')}")
            print(f"   Model: {metadata.get('model_name', 'N/A')}")
            
            # Analyze the text
            text_length = len(document)
            word_count_actual = len(document.split())
            lines = document.split('\n')
            
            print(f"   📊 Text Analysis:")
            print(f"      Character length: {text_length}")
            print(f"      Actual word count: {word_count_actual}")
            print(f"      Number of lines: {len(lines)}")
            print(f"      Ends with period: {document.strip().endswith('.')}")
            print(f"      Last 50 chars: '{document[-50:]}'")
            
            # Check if text seems truncated
            is_truncated = False
            truncation_indicators = [
                not document.strip().endswith('.'),
                not document.strip().endswith('!'),
                not document.strip().endswith('?'),
                document.strip().endswith(','),
                document.strip().endswith(' '),
                'évident' in document[-20:],  # Common truncation point
                'visible' in document[-20:],
                'détaillé' in document[-20:]
            ]
            
            if any(truncation_indicators):
                is_truncated = True
                print(f"   ⚠️ LIKELY TRUNCATED: Text doesn't end properly")
            else:
                print(f"   ✅ Text appears complete")
            
            print(f"   📝 Full text:")
            print(f"      {document}")
            print()
            
            # Check if this matches the user's example
            if "Ce modèle 3D est une représentation fantastique d'un personnage armé" in document:
                print(f"   🎯 This matches the user's example!")
                if is_truncated:
                    print(f"   🔧 This description needs to be regenerated")
                print()
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        logger.error(f"Debug error: {str(e)}", exc_info=True)
        return False

def check_web_interface_display():
    """Check how descriptions are displayed in the web interface"""
    
    print("🌐 Checking Web Interface Display")
    print("=" * 50)
    
    try:
        from fantasy_rag.web import app
        
        # Initialize components
        if not app.initialize_components():
            print("❌ Failed to initialize web app components")
            return False
        
        print("✅ Web app components initialized")
        
        # Test search to see how descriptions are returned
        from fantasy_rag.description.prompt_templates import DescriptionLanguage
        from fantasy_rag.embeddings.multilingual_search import MultilingualSearchQuery
        
        search_query = MultilingualSearchQuery(
            text="personnage armé",
            source_language=DescriptionLanguage.FRENCH,
            max_results=5,
            min_similarity=-1.0
        )
        
        print(f"🔎 Testing search: '{search_query.text}'")
        
        search_result = app.search_engine.search_multilingual(search_query)
        
        print(f"📊 Search results: {search_result.total_found} found")
        
        for i, result in enumerate(search_result.results, 1):
            print(f"\n📄 Result {i}:")
            print(f"   Asset ID: {result.asset_id}")
            print(f"   Similarity: {result.similarity_score:.3f}")
            print(f"   Description length: {len(result.description)} chars")
            print(f"   Word count: {len(result.description.split())} words")
            print(f"   Ends properly: {result.description.strip().endswith(('.', '!', '?'))}")
            print(f"   Last 50 chars: '{result.description[-50:]}'")
            
            # Check for truncation
            if not result.description.strip().endswith(('.', '!', '?')):
                print(f"   ⚠️ TRUNCATED: Description doesn't end properly")
            
            print(f"   📝 Full description:")
            print(f"      {result.description}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        logger.error(f"Web interface check error: {str(e)}", exc_info=True)
        return False

def main():
    """Main debug function"""
    
    print("🐛 Description Truncation Debug Tool")
    print("=" * 60)
    
    # Debug database descriptions
    success1 = debug_descriptions()
    
    print("\n" + "=" * 60)
    
    # Check web interface display
    success2 = check_web_interface_display()
    
    print("\n" + "=" * 60)
    print("📊 Debug Summary:")
    print(f"   Database check: {'✅ OK' if success1 else '❌ FAILED'}")
    print(f"   Web interface check: {'✅ OK' if success2 else '❌ FAILED'}")
    
    if success1 and success2:
        print("\n💡 If truncation was found, regenerate descriptions:")
        print("   python regenerate_descriptions_200_words.py")
    
    return 0 if success1 and success2 else 1

if __name__ == "__main__":
    exit(main())
