#!/usr/bin/env python3
"""
Simple test to generate a description manually
"""

import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_simple_description():
    """Test simple description generation"""
    
    print("🧪 Testing Simple Description Generation")
    print("=" * 50)
    
    try:
        from fantasy_rag.description.llm_interface import OllamaVisionInterface
        from fantasy_rag.description.prompt_templates import PromptContext, DescriptionLanguage
        
        # Initialize interface
        interface = OllamaVisionInterface(
            model_name="llama3.2-vision:latest",
            max_tokens=500,
            temperature=0.7,
            timeout=60
        )
        
        if not interface.verify_connection():
            print("❌ Failed to connect to Ollama")
            return False
        
        print("✅ Connected to Ollama")
        
        # Find render files
        render_dir = Path("ressources/data/renders")
        if not render_dir.exists():
            print(f"❌ Render directory not found: {render_dir}")
            return False
        
        # Look for any image files
        image_files = list(render_dir.glob("**/*.png")) + list(render_dir.glob("**/*.jpg"))
        
        if not image_files:
            print(f"❌ No image files found in {render_dir}")
            return False
        
        # Use the first image found
        test_image = image_files[0]
        print(f"🖼️ Using test image: {test_image}")
        
        # Create simple view paths
        view_paths = {"front": test_image}
        
        # Create simple context
        context = PromptContext(
            asset_name="Test Asset",
            view_count=1,
            target_length=100,  # Start small
            language=DescriptionLanguage.FRENCH,
            special_instructions="Décrivez simplement ce que vous voyez en français."
        )
        
        # Generate description
        print("🎯 Generating description...")
        response = interface.generate_description(view_paths, context)
        
        if response.success:
            print(f"✅ Success!")
            print(f"   Word count: {response.word_count}")
            print(f"   Processing time: {response.processing_time:.1f}s")
            print(f"📝 Description:")
            print(f"   {response.description}")
            return True
        else:
            print(f"❌ Failed: {response.error_message}")
            return False
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        logger.error(f"Test error: {str(e)}", exc_info=True)
        return False

def main():
    """Main function"""
    
    success = test_simple_description()
    
    if success:
        print("\n✅ Simple description test passed!")
        return 0
    else:
        print("\n❌ Simple description test failed!")
        return 1

if __name__ == "__main__":
    exit(main())
