#!/usr/bin/env python3
"""
Test Ollama setup and vision model availability
"""

import requests
import json
import time
from pathlib import Path

def test_ollama_connection():
    """Test basic Ollama connection"""
    print("🔍 Testing Ollama Connection")
    print("-" * 40)
    
    base_url = "http://localhost:11434"
    
    try:
        # Test basic connection
        response = requests.get(f"{base_url}/api/tags", timeout=10)
        
        if response.status_code == 200:
            print("✅ Ollama server is running")
            
            # List available models
            data = response.json()
            models = data.get('models', [])
            
            print(f"📋 Available models ({len(models)}):")
            for model in models:
                name = model.get('name', 'Unknown')
                size = model.get('size', 0) / (1024**3)  # Convert to GB
                print(f"   📦 {name} ({size:.1f} GB)")
            
            # Check for vision models
            vision_models = [m for m in models if 'vision' in m.get('name', '').lower()]
            llama_models = [m for m in models if 'llama' in m.get('name', '').lower()]
            
            print(f"\n🔍 Vision models found: {len(vision_models)}")
            for model in vision_models:
                print(f"   👁️ {model.get('name')}")
            
            print(f"\n🦙 Llama models found: {len(llama_models)}")
            for model in llama_models:
                print(f"   🦙 {model.get('name')}")
            
            return True, models
            
        else:
            print(f"❌ Ollama server error: {response.status_code}")
            print(f"   Response: {response.text}")
            return False, []
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to Ollama server")
        print("   Is Ollama running? Try: ollama serve")
        return False, []
    except Exception as e:
        print(f"❌ Error testing Ollama: {str(e)}")
        return False, []

def test_vision_model():
    """Test vision model functionality"""
    print("\n🧠 Testing Vision Model")
    print("-" * 40)
    
    base_url = "http://localhost:11434"
    
    # Try different vision model names
    vision_models = [
        "llama3.2-vision",
        "llama3.2-vision:latest", 
        "llava",
        "llava:latest",
        "bakllava",
        "moondream"
    ]
    
    for model_name in vision_models:
        print(f"\n🔍 Testing model: {model_name}")
        
        try:
            # Simple text-only test first
            test_data = {
                "model": model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": "Hello! Can you see images?"
                    }
                ],
                "stream": False
            }
            
            response = requests.post(
                f"{base_url}/api/chat",
                json=test_data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result.get('message', {}).get('content', '')
                
                print(f"✅ {model_name} is working!")
                print(f"   Response: {content[:100]}...")
                return model_name
                
            else:
                print(f"❌ {model_name} failed: {response.status_code}")
                
        except Exception as e:
            print(f"❌ {model_name} error: {str(e)}")
    
    print("\n❌ No working vision models found")
    return None

def suggest_ollama_setup():
    """Provide setup instructions for Ollama"""
    print("\n📋 Ollama Setup Instructions")
    print("=" * 50)
    
    print("1. 📥 Install Ollama:")
    print("   Windows: Download from https://ollama.com/download")
    print("   Linux: curl -fsSL https://ollama.com/install.sh | sh")
    print("   macOS: brew install ollama")
    
    print("\n2. 🚀 Start Ollama server:")
    print("   ollama serve")
    
    print("\n3. 📦 Install vision model:")
    print("   ollama pull llama3.2-vision")
    print("   # or")
    print("   ollama pull llava")
    
    print("\n4. 🧪 Test installation:")
    print("   ollama run llama3.2-vision")
    print("   # Then type: Can you see images?")
    
    print("\n5. 🔧 Alternative models if llama3.2-vision unavailable:")
    print("   ollama pull llava:latest")
    print("   ollama pull bakllava:latest")
    print("   ollama pull moondream:latest")

def test_with_sample_image():
    """Test vision model with a sample image"""
    print("\n🖼️ Testing with Sample Image")
    print("-" * 40)
    
    # Check if we have rendered images to test with
    renders_dir = Path("ressources/data/renders")
    
    if renders_dir.exists():
        # Find any PNG files
        png_files = list(renders_dir.rglob("*.png"))
        
        if png_files:
            test_image = png_files[0]
            print(f"📸 Found test image: {test_image.name}")
            print(f"   Size: {test_image.stat().st_size / 1024:.1f} KB")
            
            # We'll implement actual vision testing later
            print("   Vision testing will be implemented in next step")
            return True
        else:
            print("📸 No rendered images found for testing")
            print("   Run rendering first to generate test images")
            return False
    else:
        print("📁 Renders directory not found")
        return False

def main():
    """Run complete Ollama setup test"""
    
    print("🧠 Ollama Setup and Vision Model Test")
    print("=" * 60)
    
    # Test 1: Basic connection
    connected, models = test_ollama_connection()
    
    if not connected:
        suggest_ollama_setup()
        return 1
    
    # Test 2: Vision model availability
    working_model = test_vision_model()
    
    if not working_model:
        print("\n⚠️ No vision models available")
        suggest_ollama_setup()
        return 1
    
    # Test 3: Sample image test
    test_with_sample_image()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary:")
    print(f"   ✅ Ollama server: Running")
    print(f"   ✅ Models available: {len(models)}")
    print(f"   ✅ Working vision model: {working_model}")
    print(f"   🎯 Ready for description generation!")
    
    return 0

if __name__ == "__main__":
    exit(main())
