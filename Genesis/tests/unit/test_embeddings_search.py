#!/usr/bin/env python3
"""
Test Embeddings and Search Module

This script tests the embedding generation and semantic search functionality
for fantasy asset descriptions.
"""

import logging
from pathlib import Path
import time

def test_text_embedder():
    """Test text embedding functionality"""
    
    print("🧠 Testing Text Embedder")
    print("-" * 50)
    
    try:
        from fantasy_rag.embeddings.text_embedder import TextEmbedder, EmbeddingConfig
        
        # Test configuration
        config = EmbeddingConfig(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            batch_size=4,
            normalize_embeddings=True
        )
        
        embedder = TextEmbedder(config)
        
        # Test texts in French and English
        test_descriptions = {
            "sword_fr": "Cette épée magique brille d'une lueur mystique avec des runes anciennes gravées sur la lame d'acier poli.",
            "sword_en": "This magical sword glows with mystical light and has ancient runes carved on the polished steel blade.",
            "shield_fr": "Un bouclier robuste orné de motifs celtiques, forgé dans un métal résistant avec une surface brillante.",
            "shield_en": "A sturdy shield adorned with Celtic patterns, forged from resistant metal with a shining surface."
        }
        
        print(f"📝 Test descriptions: {len(test_descriptions)}")
        
        # Generate embeddings
        start_time = time.time()
        results = embedder.embed_descriptions(test_descriptions)
        total_time = time.time() - start_time
        
        print(f"✅ Embeddings generated in {total_time:.2f}s")
        
        # Analyze results
        for asset_id, result in results.items():
            print(f"   {asset_id}: {result.embeddings.shape} ({result.processing_time*1000:.1f}ms)")
        
        # Test similarity
        sword_fr_emb = results["sword_fr"].embeddings
        sword_en_emb = results["sword_en"].embeddings
        shield_fr_emb = results["shield_fr"].embeddings
        
        # Cross-language similarity (should be high)
        cross_lang_sim = embedder.compute_similarity(sword_fr_emb, sword_en_emb)[0,0]
        
        # Cross-category similarity (should be lower)
        cross_cat_sim = embedder.compute_similarity(sword_fr_emb, shield_fr_emb)[0,0]
        
        print(f"\n📊 Similarity Analysis:")
        print(f"   Sword (FR) vs Sword (EN): {cross_lang_sim:.3f}")
        print(f"   Sword (FR) vs Shield (FR): {cross_cat_sim:.3f}")
        
        if cross_lang_sim > 0.7:
            print("   ✅ Good cross-language similarity")
        else:
            print("   ⚠️ Cross-language similarity could be better")
        
        if cross_lang_sim > cross_cat_sim:
            print("   ✅ Language similarity > category difference")
        else:
            print("   ⚠️ Category difference > language similarity")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {str(e)}")
        print("   Install required packages: pip install sentence-transformers")
        return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def test_similarity_search():
    """Test similarity search functionality"""
    
    print("\n🔍 Testing Similarity Search")
    print("-" * 50)
    
    try:
        from fantasy_rag.embeddings.similarity_search import SimilaritySearchEngine, SearchQuery
        from fantasy_rag.embeddings.text_embedder import EmbeddingConfig
        
        # Initialize search engine
        config = EmbeddingConfig(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        
        search_engine = SimilaritySearchEngine(
            db_path="ressources/data/test_chroma_db",
            collection_name="test_fantasy_assets",
            embedding_config=config
        )
        
        if not search_engine.initialize():
            print("❌ Failed to initialize search engine")
            return False
        
        print("✅ Search engine initialized")
        
        # Clear any existing data
        search_engine.clear_collection()
        
        # Sample fantasy asset descriptions
        sample_descriptions = {
            "epic_sword_001": "Une épée légendaire forgée dans les flammes du dragon, avec une lame d'acier enchanté qui brille d'une lueur bleue mystique. La garde est ornée de runes anciennes et le pommeau contient un cristal de pouvoir.",
            "magic_staff_001": "A powerful wizard's staff carved from ancient oak, topped with a glowing crystal orb. Intricate magical symbols spiral down the shaft, and the wood seems to pulse with arcane energy.",
            "dragon_shield_001": "Un bouclier massif en écailles de dragon rouge, renforcé par des bandes de mithril. La surface reflète la lumière comme du métal poli et des motifs de flammes sont gravés sur les bords.",
            "elven_bow_001": "An elegant elven longbow crafted from silverwood, with a string made from unicorn hair. The bow is decorated with leaf patterns and glows softly in moonlight.",
            "cursed_dagger_001": "Une dague maudite à la lame noire comme l'obsidienne, gravée de symboles démoniaques. Le manche en os blanchi semble absorber la lumière environnante."
        }
        
        # Add descriptions to search index
        print(f"📚 Adding {len(sample_descriptions)} descriptions...")
        
        # Prepare metadata
        metadata = {
            "epic_sword_001": {"category": "weapon", "type": "sword", "rarity": "legendary", "language": "français"},
            "magic_staff_001": {"category": "weapon", "type": "staff", "rarity": "rare", "language": "english"},
            "dragon_shield_001": {"category": "armor", "type": "shield", "rarity": "epic", "language": "français"},
            "elven_bow_001": {"category": "weapon", "type": "bow", "rarity": "rare", "language": "english"},
            "cursed_dagger_001": {"category": "weapon", "type": "dagger", "rarity": "cursed", "language": "français"}
        }
        
        if not search_engine.add_descriptions(sample_descriptions, metadata):
            print("❌ Failed to add descriptions")
            return False
        
        print("✅ Descriptions added to search index")
        
        # Test searches
        test_queries = [
            ("magical sword", "english", None),
            ("épée magique", "français", None),
            ("bow weapon", "english", "weapon"),
            ("bouclier dragon", "français", "armor")
        ]
        
        print(f"\n🔎 Testing {len(test_queries)} search queries:")
        
        for query_text, language, category in test_queries:
            print(f"\n   Query: '{query_text}' (lang: {language}, cat: {category})")
            
            query = SearchQuery(
                text=query_text,
                language=language,
                category_filter=category,
                max_results=3,
                min_similarity=0.1
            )
            
            response = search_engine.search(query)
            
            print(f"   Results: {response.total_found} found in {response.processing_time:.2f}s")
            
            for i, result in enumerate(response.results[:2]):  # Show top 2
                print(f"     {i+1}. {result.asset_id} (sim: {result.similarity_score:.3f})")
                print(f"        {result.description[:80]}...")
        
        # Get collection statistics
        stats = search_engine.get_collection_stats()
        print(f"\n📊 Collection Statistics:")
        print(f"   Total documents: {stats.get('total_documents', 0)}")
        print(f"   Categories: {stats.get('categories', {})}")
        print(f"   Languages: {stats.get('languages', {})}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {str(e)}")
        print("   Install required packages: pip install chromadb")
        return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def test_multilingual_search():
    """Test multilingual search functionality"""
    
    print("\n🌍 Testing Multilingual Search")
    print("-" * 50)
    
    try:
        from fantasy_rag.embeddings.multilingual_search import MultilingualSearchEngine, MultilingualSearchQuery
        from fantasy_rag.description.prompt_templates import DescriptionLanguage
        
        # Initialize multilingual search engine
        search_engine = MultilingualSearchEngine(
            db_path="ressources/data/test_multilingual_db",
            collection_name="test_multilingual_assets"
        )
        
        if not search_engine.initialize():
            print("❌ Failed to initialize multilingual search engine")
            return False
        
        print("✅ Multilingual search engine initialized")
        
        # Clear existing data
        search_engine.clear_collection()
        
        # Sample multilingual descriptions
        multilingual_descriptions = {
            "epic_sword_001": {
                "français": "Une épée légendaire forgée dans les flammes du dragon, avec une lame d'acier enchanté qui brille d'une lueur bleue mystique.",
                "english": "A legendary sword forged in dragon flames, with an enchanted steel blade that glows with mystical blue light."
            },
            "magic_staff_001": {
                "français": "Un bâton de magicien puissant sculpté dans du chêne ancien, surmonté d'un orbe de cristal lumineux.",
                "english": "A powerful wizard's staff carved from ancient oak, topped with a glowing crystal orb."
            },
            "dragon_shield_001": {
                "français": "Un bouclier massif en écailles de dragon rouge, renforcé par des bandes de mithril brillant.",
                "english": "A massive shield made from red dragon scales, reinforced with shining mithril bands."
            }
        }
        
        # Prepare metadata
        metadata = {
            "epic_sword_001": {"category": "weapon", "type": "sword", "rarity": "legendary"},
            "magic_staff_001": {"category": "weapon", "type": "staff", "rarity": "rare"},
            "dragon_shield_001": {"category": "armor", "type": "shield", "rarity": "epic"}
        }
        
        # Add multilingual descriptions
        print(f"📚 Adding multilingual descriptions for {len(multilingual_descriptions)} assets...")
        
        if not search_engine.add_multilingual_descriptions(multilingual_descriptions, metadata):
            print("❌ Failed to add multilingual descriptions")
            return False
        
        print("✅ Multilingual descriptions added")
        
        # Test cross-language search
        print(f"\n🔄 Testing cross-language search:")
        
        # Search in French for English results
        response = search_engine.search_cross_language(
            query_text="épée magique dragon",
            source_language=DescriptionLanguage.FRENCH,
            target_language=DescriptionLanguage.ENGLISH,
            max_results=3
        )
        
        print(f"   French query → English results: {response.total_found} found")
        for result in response.results[:2]:
            print(f"     {result.asset_id} (sim: {result.similarity_score:.3f})")
            print(f"     {result.description[:80]}...")
        
        # Search in English for French results
        response = search_engine.search_cross_language(
            query_text="magical dragon sword",
            source_language=DescriptionLanguage.ENGLISH,
            target_language=DescriptionLanguage.FRENCH,
            max_results=3
        )
        
        print(f"   English query → French results: {response.total_found} found")
        for result in response.results[:2]:
            print(f"     {result.asset_id} (sim: {result.similarity_score:.3f})")
            print(f"     {result.description[:80]}...")
        
        # Get language statistics
        lang_stats = search_engine.get_language_statistics()
        print(f"\n📊 Language Statistics:")
        print(f"   Total documents: {lang_stats.get('total_documents', 0)}")
        print(f"   Unique assets: {lang_stats.get('unique_assets', 0)}")
        print(f"   Language distribution: {lang_stats.get('language_distribution', {})}")
        print(f"   Multilingual assets: {lang_stats.get('multilingual_assets', 0)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def main():
    """Run complete embeddings and search test"""
    
    # Setup logging
    logging.basicConfig(level=logging.WARNING)  # Reduce noise
    
    print("🔍 Fantasy Asset Embeddings & Search Test")
    print("=" * 60)
    
    tests = [
        ("Text Embedder", test_text_embedder),
        ("Similarity Search", test_similarity_search),
        ("Multilingual Search", test_multilingual_search)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🧪 Testing: {test_name}")
        try:
            if test_func():
                print(f"✅ {test_name}: PASSED")
                passed += 1
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {str(e)}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary:")
    print(f"   Tests passed: {passed}/{total}")
    print(f"   Success rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("🎉 All embedding and search tests passed!")
        print("   Ready for integration with description generation")
        
        print("\n🚀 Next Steps:")
        print("   1. Generate descriptions: python test_description_generation.py")
        print("   2. Index descriptions: python test_full_pipeline.py")
        print("   3. Test search interface: python test_search_interface.py")
        
        return 0
    else:
        print("⚠️ Some tests failed. Check the output above.")
        return 1

if __name__ == "__main__":
    exit(main())
