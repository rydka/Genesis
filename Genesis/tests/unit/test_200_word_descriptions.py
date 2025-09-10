#!/usr/bin/env python3
"""
Test script to verify that descriptions are generated with 200 words target length
"""

import logging
import time
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_description_length():
    """Test that descriptions are generated with 200 words target"""
    
    print("🧪 Testing 200-word Description Generation")
    print("=" * 60)
    
    try:
        from fantasy_rag.description.llm_interface import OllamaVisionInterface
        from fantasy_rag.description.prompt_templates import PromptContext, DescriptionLanguage
        
        # Initialize vision interface
        print("🔧 Initializing vision interface...")
        interface = OllamaVisionInterface(model_name="llama3.2-vision:latest")
        
        if not interface.verify_connection():
            print("❌ Failed to connect to Ollama")
            print("   Make sure Ollama is running with llama3.2-vision model")
            return False

        print("✅ Connected to Ollama")
        
        # Find test images
        renders_dir = Path("ressources/data/renders")
        if not renders_dir.exists():
            print("❌ No renders directory found")
            print("   Run rendering first: python test_rendering.py")
            return False
        
        # Find any asset with renders
        asset_dirs = [d for d in renders_dir.iterdir() if d.is_dir()]
        if not asset_dirs:
            print("❌ No asset renders found")
            return False
        
        test_asset_dir = asset_dirs[0]
        asset_name = test_asset_dir.name
        
        # Find view images
        view_files = list(test_asset_dir.glob("*.png"))
        if not view_files:
            print(f"❌ No PNG files found in {test_asset_dir}")
            return False
        
        # Create view paths dictionary
        view_paths = {}
        for view_file in view_files[:6]:  # Use up to 6 views
            view_name = view_file.stem
            view_paths[view_name] = view_file
        
        print(f"📦 Testing with asset: {asset_name}")
        print(f"🖼️ Using {len(view_paths)} views: {list(view_paths.keys())}")
        
        # Test with default context (should be 200 words)
        print("\n🎯 Testing default target length...")
        context_default = PromptContext(
            asset_name=asset_name,
            view_count=len(view_paths),
            language=DescriptionLanguage.FRENCH
            # target_length should default to 200
        )
        
        print(f"   Target length: {context_default.target_length} words")
        
        start_time = time.time()
        response = interface.generate_description(view_paths, context_default)
        generation_time = time.time() - start_time
        
        if response.success:
            print(f"✅ Description generated successfully!")
            print(f"   Processing time: {response.processing_time:.1f}s")
            print(f"   Total time: {generation_time:.1f}s")
            print(f"   Word count: {response.word_count} words")
            print(f"   Target: {context_default.target_length} words")
            
            # Check if word count is close to target
            target = context_default.target_length
            actual = response.word_count
            diff = abs(actual - target)
            percentage_diff = (diff / target) * 100
            
            print(f"   Difference: {diff} words ({percentage_diff:.1f}%)")
            
            if percentage_diff <= 10:  # Within 10%
                print(f"✅ Word count is within acceptable range!")
            elif percentage_diff <= 20:  # Within 20%
                print(f"⚠️ Word count is acceptable but could be better")
            else:
                print(f"❌ Word count is too far from target")
            
            # Show first few lines of description
            lines = response.description.split('\n')
            print(f"\n📝 Description preview:")
            for i, line in enumerate(lines[:3]):
                if line.strip():
                    print(f"   {line.strip()}")
            if len(lines) > 3:
                print("   ...")
            
            return percentage_diff <= 20
            
        else:
            print(f"❌ Description generation failed: {response.error_message}")
            return False
        
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        logger.error(f"Test error: {str(e)}", exc_info=True)
        return False

def test_explicit_200_words():
    """Test with explicitly set 200 words"""
    
    print("\n🎯 Testing explicit 200-word target...")
    
    try:
        from fantasy_rag.description.llm_interface import OllamaVisionInterface
        from fantasy_rag.description.prompt_templates import PromptContext, DescriptionLanguage
        
        interface = OllamaVisionInterface(model_name="llama3.2-vision:latest")
        if not interface.verify_connection():
            return False
        
        # Find test images
        renders_dir = Path("ressources/data/renders")
        asset_dirs = [d for d in renders_dir.iterdir() if d.is_dir()]
        test_asset_dir = asset_dirs[0]
        asset_name = test_asset_dir.name
        
        view_files = list(test_asset_dir.glob("*.png"))
        view_paths = {}
        for view_file in view_files[:6]:
            view_name = view_file.stem
            view_paths[view_name] = view_file
        
        # Test with explicit 200 words
        context_200 = PromptContext(
            asset_name=asset_name,
            view_count=len(view_paths),
            target_length=200,  # Explicitly set to 200
            language=DescriptionLanguage.FRENCH
        )
        
        print(f"   Target length: {context_200.target_length} words")
        
        response = interface.generate_description(view_paths, context_200)
        
        if response.success:
            target = context_200.target_length
            actual = response.word_count
            diff = abs(actual - target)
            percentage_diff = (diff / target) * 100
            
            print(f"✅ Generated: {actual} words (target: {target})")
            print(f"   Difference: {diff} words ({percentage_diff:.1f}%)")
            
            return percentage_diff <= 20
        else:
            print(f"❌ Failed: {response.error_message}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def main():
    """Main test function"""
    
    print("🧪 Testing 200-Word Description Target")
    print("=" * 60)
    
    # Test 1: Default target length
    success1 = test_description_length()
    
    # Test 2: Explicit 200 words
    success2 = test_explicit_200_words()
    
    print("\n" + "=" * 60)
    print("📊 Test Results:")
    print(f"   Default target: {'✅ PASS' if success1 else '❌ FAIL'}")
    print(f"   Explicit 200: {'✅ PASS' if success2 else '❌ FAIL'}")
    
    if success1 and success2:
        print("\n🎉 All tests passed! Descriptions are targeting 200 words.")
        print("\n💡 Next steps:")
        print("   1. Generate new descriptions: python process_selected_assets.py")
        print("   2. Check web interface: python -m fantasy_rag.web.app")
        return 0
    else:
        print("\n❌ Some tests failed. Check the configuration.")
        return 1

if __name__ == "__main__":
    exit(main())
