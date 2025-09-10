#!/usr/bin/env python3
"""
Simple test of description generation infrastructure without requiring vision model
"""

import logging
from pathlib import Path

def test_prompt_generation():
    """Test prompt template generation"""
    
    print("📝 Testing Prompt Generation")
    print("-" * 40)
    
    try:
        from fantasy_rag.description.prompt_templates import (
            FantasyPromptTemplates, PromptContext, DescriptionLanguage, AssetCategory
        )
        
        templates = FantasyPromptTemplates()
        
        # Test French prompt
        context_fr = PromptContext(
            asset_name="Épée Magique",
            view_count=6,
            target_length=200,
            language=DescriptionLanguage.FRENCH,
            category_hint=AssetCategory.WEAPON
        )
        
        prompts_fr = templates.generate_complete_prompt(context_fr)
        
        print(f"✅ French prompts generated:")
        print(f"   System: {len(prompts_fr['system'])} chars")
        print(f"   User: {len(prompts_fr['user'])} chars")
        
        # Test English prompt
        context_en = PromptContext(
            asset_name="Magic Sword",
            view_count=6,
            target_length=200,
            language=DescriptionLanguage.ENGLISH,
            category_hint=AssetCategory.WEAPON
        )
        
        prompts_en = templates.generate_complete_prompt(context_en)
        
        print(f"✅ English prompts generated:")
        print(f"   System: {len(prompts_en['system'])} chars")
        print(f"   User: {len(prompts_en['user'])} chars")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def test_description_parsing():
    """Test description parsing and quality assessment"""
    
    print("\n🔍 Testing Description Parsing")
    print("-" * 40)
    
    try:
        from fantasy_rag.description.description_parser import DescriptionParser
        from fantasy_rag.description.quality_assessor import QualityAssessor
        from fantasy_rag.description.prompt_templates import DescriptionLanguage
        
        # Sample description for testing
        sample_description = """
        Cette épée fantastique présente une lame élégante forgée dans un acier brillant, 
        ornée de gravures runiques qui scintillent d'une lueur mystique. La garde 
        finement ciselée arbore des motifs entrelacés, tandis que la poignée en cuir 
        tressé offre une prise ferme et confortable. Le pommeau, serti d'une gemme 
        bleue translucide, complète harmonieusement cette arme d'exception. Les détails 
        méticuleux révèlent un artisanat de maître, alliant beauté esthétique et 
        efficacité martiale. Cette lame semble vibrer d'une énergie ancienne, 
        témoignant de son héritage légendaire et de sa puissance magique latente.
        """
        
        # Test parser
        parser = DescriptionParser(target_length=200)
        parsed = parser.parse_description(sample_description)
        
        print(f"✅ Description parsed:")
        print(f"   Word count: {parsed.word_count}")
        print(f"   Sentences: {len(parsed.sentences)}")
        print(f"   Quality score: {parsed.quality_score:.1f}")
        print(f"   Issues: {len(parsed.issues)}")
        
        # Test quality assessor
        assessor = QualityAssessor(target_length=200)
        quality = assessor.assess_quality(sample_description, DescriptionLanguage.FRENCH)
        
        print(f"✅ Quality assessment:")
        print(f"   Overall score: {quality.overall_score:.1f}")
        print(f"   Length score: {quality.length_score:.1f}")
        print(f"   Language score: {quality.language_score:.1f}")
        print(f"   Content score: {quality.content_score:.1f}")
        print(f"   Structure score: {quality.structure_score:.1f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def test_category_classification():
    """Test category classification"""
    
    print("\n🏷️ Testing Category Classification")
    print("-" * 40)
    
    try:
        from fantasy_rag.description.category_classifier import CategoryClassifier
        
        classifier = CategoryClassifier()
        
        # Test different asset names
        test_cases = [
            "Epic-Sword-of-Power",
            "Dragon-Scale-Armor",
            "Elven-Warrior-Character",
            "Ancient-Crystal-Orb",
            "Demon-Beast-Creature"
        ]
        
        for asset_name in test_cases:
            result = classifier.classify_by_name(asset_name)
            print(f"✅ {asset_name}:")
            print(f"   Category: {result.category.value}")
            print(f"   Confidence: {result.confidence:.2f}")
            print(f"   Keywords: {result.keywords_found}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def test_multilingual_support():
    """Test multilingual support"""
    
    print("\n🌍 Testing Multilingual Support")
    print("-" * 40)
    
    try:
        from fantasy_rag.description.multilingual_handler import MultilingualHandler
        
        handler = MultilingualHandler()
        
        # Test language detection
        test_texts = [
            ("Cette épée magique est très belle et puissante.", "French"),
            ("This magical sword is very beautiful and powerful.", "English"),
            ("Le dragon rouge vole dans le ciel étoilé.", "French"),
            ("The red dragon flies in the starry sky.", "English")
        ]
        
        for text, expected in test_texts:
            detection = handler.detect_language(text)
            print(f"✅ Text: '{text[:30]}...'")
            print(f"   Detected: {detection.detected_language.value}")
            print(f"   Expected: {expected}")
            print(f"   Confidence: {detection.confidence:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def check_rendered_assets():
    """Check if we have rendered assets for testing"""
    
    print("\n📸 Checking Rendered Assets")
    print("-" * 40)
    
    renders_dir = Path("ressources/data/renders")
    
    if not renders_dir.exists():
        print("❌ Renders directory not found")
        print("   Run: python test_rendering.py")
        return False
    
    # Count assets with complete view sets
    asset_dirs = [d for d in renders_dir.iterdir() if d.is_dir()]
    complete_assets = 0
    
    for asset_dir in asset_dirs:
        view_files = list(asset_dir.glob("*.png"))
        if len(view_files) >= 6:
            complete_assets += 1
            print(f"✅ {asset_dir.name}: {len(view_files)} views")
    
    if complete_assets > 0:
        print(f"✅ Found {complete_assets} assets ready for description generation")
        return True
    else:
        print("❌ No complete asset view sets found")
        print("   Run: python test_rendering.py")
        return False

def main():
    """Run simple description infrastructure tests"""
    
    # Setup logging
    logging.basicConfig(level=logging.WARNING)  # Reduce noise
    
    print("🧠 Fantasy Asset Description Infrastructure Test")
    print("=" * 60)
    
    tests = [
        ("Prompt Generation", test_prompt_generation),
        ("Description Parsing", test_description_parsing),
        ("Category Classification", test_category_classification),
        ("Multilingual Support", test_multilingual_support),
        ("Rendered Assets Check", check_rendered_assets)
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
        print("🎉 All infrastructure tests passed!")
        print("   Ready for vision model integration")
        
        if passed >= 4:  # All except vision model
            print("\n🚀 Next Steps:")
            print("   1. Install vision model: python setup_ollama_vision.py")
            print("   2. Test full pipeline: python test_description_generation.py")
        
        return 0
    else:
        print("⚠️ Some tests failed. Check the output above.")
        return 1

if __name__ == "__main__":
    exit(main())
