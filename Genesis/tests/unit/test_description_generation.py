#!/usr/bin/env python3
"""
Test description generation with vision model
"""

import logging
from pathlib import Path
import time

from fantasy_rag.description.llm_interface import OllamaVisionInterface
from fantasy_rag.description.prompt_templates import PromptContext, DescriptionLanguage

def find_rendered_views():
    """Find rendered view images for testing"""
    
    print("🔍 Looking for rendered views...")
    
    renders_dir = Path("ressources/data/renders")
    
    if not renders_dir.exists():
        print("❌ Renders directory not found")
        return None
    
    # Look for asset directories with 6 views
    asset_dirs = [d for d in renders_dir.iterdir() if d.is_dir()]
    
    for asset_dir in asset_dirs:
        print(f"📁 Checking: {asset_dir.name}")
        
        # Look for the 6 standard views
        view_files = {}
        view_names = ["front", "back", "left", "right", "top", "iso_3d"]
        
        for view_name in view_names:
            # Find files containing the view name
            matching_files = list(asset_dir.glob(f"*{view_name}.png"))
            if matching_files:
                view_files[view_name] = matching_files[0]
                print(f"   ✅ {view_name}: {matching_files[0].name}")
            else:
                print(f"   ❌ {view_name}: not found")
        
        if len(view_files) >= 4:  # At least 4 views required
            print(f"✅ Found {len(view_files)} views in {asset_dir.name}")
            return asset_dir.name, view_files
    
    print("❌ No complete view sets found")
    return None

def test_single_description():
    """Test generating a single description"""
    
    print("\n🧠 Testing Single Description Generation")
    print("-" * 50)
    
    # Find rendered views
    result = find_rendered_views()
    if not result:
        print("❌ No rendered views available for testing")
        print("   Run rendering first: python test_rendering.py")
        return False
    
    asset_name, view_paths = result
    
    print(f"📦 Testing with asset: {asset_name}")
    print(f"🖼️ Views available: {list(view_paths.keys())}")
    
    # Initialize interface
    interface = OllamaVisionInterface(
        model_name="llama3.2-vision:latest",
        temperature=0.7,
        timeout=120
    )
    
    # Test connection
    if not interface.verify_connection():
        print("❌ Cannot connect to Ollama")
        return False
    
    # Create context for French description
    context = PromptContext(
        asset_name=asset_name,
        view_count=len(view_paths),
        target_length=200,
        language=DescriptionLanguage.FRENCH
    )
    
    print(f"\n🎯 Generating description...")
    print(f"   Language: {context.language.value}")
    print(f"   Target length: {context.target_length} words")
    
    start_time = time.time()
    
    # Generate description
    response = interface.generate_description(view_paths, context)

    total_time = time.time() - start_time

    print(f"\n📊 Results:")
    print(f"   Success: {response.success}")
    print(f"   Processing time: {response.processing_time:.1f}s")
    print(f"   Total time: {total_time:.1f}s")

    if response.success:
        print(f"   Word count: {response.word_count}")
        print(f"   Target: {context.target_length} words")

        # Show metadata if available
        if hasattr(response, 'metadata') and response.metadata:
            metadata = response.metadata
            if 'analysis_mode' in metadata:
                print(f"   Analysis mode: {metadata['analysis_mode']}")
            if 'primary_view' in metadata:
                print(f"   Primary view: {metadata['primary_view']}")

        # Show description
        print(f"\n📝 Generated Description:")
        print("-" * 50)
        print(response.description)
        print("-" * 50)

        # Analyze quality
        word_diff = abs(response.word_count - context.target_length)
        word_accuracy = max(0, 100 - (word_diff / context.target_length * 100))

        print(f"\n📈 Quality Analysis:")
        print(f"   Word count accuracy: {word_accuracy:.1f}%")
        print(f"   Length deviation: {word_diff} words")

        if word_accuracy >= 90:
            print("   ✅ Excellent length control")
        elif word_accuracy >= 75:
            print("   ✅ Good length control")
        else:
            print("   ⚠️ Length control needs improvement")

        return True
    else:
        print(f"   ❌ Error: {response.error_message}")
        return False

def test_multilingual_descriptions():
    """Test generating descriptions in multiple languages"""
    
    print("\n🌍 Testing Multilingual Description Generation")
    print("-" * 50)
    
    # Find rendered views
    result = find_rendered_views()
    if not result:
        print("❌ No rendered views available for testing")
        return False
    
    asset_name, view_paths = result
    
    # Initialize interface
    interface = OllamaVisionInterface(
        model_name="llama3.2-vision:latest",
        temperature=0.7,
        timeout=120
    )
    
    print(f"📦 Testing with asset: {asset_name}")
    print(f"🎯 Generating descriptions in French and English...")
    
    # Generate multilingual descriptions
    start_time = time.time()

    results = interface.generate_multilingual_descriptions(
        view_paths=view_paths,
        asset_name=asset_name,
        target_length=200  # Standard length
    )

    total_time = time.time() - start_time

    print(f"\n📊 Multilingual Results:")
    print(f"   Total time: {total_time:.1f}s")
    print(f"   Languages processed: {len(results)}")

    for language, response in results.items():
        print(f"\n🔤 {language.upper()}:")
        print(f"   Success: {response.success}")
        print(f"   Processing time: {response.processing_time:.1f}s")

        if response.success:
            print(f"   Word count: {response.word_count}")
            print(f"   Preview: {response.description[:100]}...")

            # Show metadata if available
            if hasattr(response, 'metadata') and response.metadata:
                metadata = response.metadata
                if 'analysis_mode' in metadata:
                    print(f"   Analysis mode: {metadata['analysis_mode']}")
        else:
            print(f"   Error: {response.error_message}")

    # Check if both languages succeeded
    success_count = sum(1 for r in results.values() if r.success)

    if success_count == len(results):
        print(f"\n✅ All {len(results)} languages generated successfully!")
        return True
    else:
        print(f"\n⚠️ Only {success_count}/{len(results)} languages succeeded")
        return False

def test_prompt_templates():
    """Test prompt template generation"""
    
    print("\n📝 Testing Prompt Templates")
    print("-" * 50)
    
    from fantasy_rag.description.prompt_templates import FantasyPromptTemplates, AssetCategory
    
    templates = FantasyPromptTemplates()
    
    # Test basic context
    context = PromptContext(
        asset_name="Test Sword",
        view_count=6,
        target_length=200,
        language=DescriptionLanguage.FRENCH,
        category_hint=AssetCategory.WEAPON
    )
    
    # Generate prompts
    prompts = templates.generate_complete_prompt(context)
    
    print(f"✅ System prompt generated: {len(prompts['system'])} chars")
    print(f"✅ User prompt generated: {len(prompts['user'])} chars")
    
    print(f"\n📋 System Prompt Preview:")
    print(prompts['system'][:200] + "...")
    
    print(f"\n📋 User Prompt Preview:")
    print(prompts['user'][:300] + "...")
    
    return True

def main():
    """Run complete description generation test"""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    print("🧠 Fantasy Asset Description Generation Test")
    print("=" * 60)
    
    success_count = 0
    total_tests = 3
    
    # Test 1: Prompt templates
    print("\n🧪 Test 1/3: Prompt Templates")
    if test_prompt_templates():
        success_count += 1
        print("✅ Prompt templates test passed")
    else:
        print("❌ Prompt templates test failed")
    
    # Test 2: Single description
    print("\n🧪 Test 2/3: Single Description Generation")
    if test_single_description():
        success_count += 1
        print("✅ Single description test passed")
    else:
        print("❌ Single description test failed")
    
    # Test 3: Multilingual descriptions
    print("\n🧪 Test 3/3: Multilingual Description Generation")
    if test_multilingual_descriptions():
        success_count += 1
        print("✅ Multilingual description test passed")
    else:
        print("❌ Multilingual description test failed")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary:")
    print(f"   Tests passed: {success_count}/{total_tests}")
    print(f"   Success rate: {success_count/total_tests*100:.1f}%")
    
    if success_count == total_tests:
        print("🎉 All tests passed! Description generation is ready!")
        return 0
    else:
        print("⚠️ Some tests failed. Check the output above.")
        return 1

if __name__ == "__main__":
    exit(main())
