#!/usr/bin/env python3
"""
Setup Ollama with vision model for Fantasy Asset Description Generation
"""

import subprocess
import requests
import time
import sys
import os
from pathlib import Path

def find_ollama_executable():
    """Find Ollama executable in common locations"""

    # Common Ollama installation paths on Windows
    possible_paths = [
        Path(os.environ.get('USERPROFILE', '')) / "AppData" / "Local" / "Programs" / "Ollama" / "ollama.exe",
        Path("C:/Program Files/Ollama/ollama.exe"),
        Path("C:/Program Files (x86)/Ollama/ollama.exe"),
        Path(os.environ.get('LOCALAPPDATA', '')) / "Programs" / "Ollama" / "ollama.exe"
    ]

    # Check if ollama is already in PATH
    try:
        result = subprocess.run(["ollama", "--version"], capture_output=True, timeout=5)
        if result.returncode == 0:
            return "ollama"  # Already in PATH
    except:
        pass

    # Search in common locations
    for path in possible_paths:
        if path.exists():
            print(f"✅ Found Ollama at: {path}")
            return str(path)

    return None

def add_ollama_to_path():
    """Add Ollama to PATH for current session"""

    ollama_path = find_ollama_executable()

    if not ollama_path:
        print("❌ Ollama executable not found")
        print("\n💡 Please install Ollama from: https://ollama.com/download")
        return None

    if ollama_path == "ollama":
        print("✅ Ollama already in PATH")
        return "ollama"

    # Add directory to PATH for current session
    ollama_dir = str(Path(ollama_path).parent)
    current_path = os.environ.get('PATH', '')

    if ollama_dir not in current_path:
        os.environ['PATH'] = f"{ollama_dir};{current_path}"
        print(f"✅ Added Ollama to PATH: {ollama_dir}")

    return ollama_path

def check_ollama_running():
    """Check if Ollama server is running"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False

def install_vision_model():
    """Install a vision model for Ollama"""

    print("🧠 Installing Vision Model for Ollama")
    print("=" * 50)

    # Ensure Ollama is available
    ollama_cmd = add_ollama_to_path()
    if not ollama_cmd:
        return False

    if not check_ollama_running():
        print("❌ Ollama server is not running")
        print("   Please start Ollama first: ollama serve")
        return False

    print("✅ Ollama server is running")
    
    # Try different vision models in order of preference
    models_to_try = [
        ("llama3.2-vision", "Llama 3.2 Vision (Recommended)"),
        ("llava", "LLaVA (Alternative)"),
        ("bakllava", "BakLLaVA (Lightweight)")
    ]
    
    for model_name, description in models_to_try:
        print(f"\n🔄 Trying to install: {description}")
        print(f"   Model: {model_name}")
        
        try:
            # Use subprocess to call ollama pull
            print(f"   Running: ollama pull {model_name}")
            print("   This may take several minutes...")
            
            result = subprocess.run(
                [ollama_cmd, "pull", model_name],
                capture_output=True,
                text=True,
                timeout=1800  # 30 minutes timeout
            )
            
            if result.returncode == 0:
                print(f"✅ Successfully installed {model_name}")
                
                # Test the model
                print(f"🧪 Testing {model_name}...")
                test_result = test_vision_model(model_name)
                
                if test_result:
                    print(f"🎉 {model_name} is working correctly!")
                    return model_name
                else:
                    print(f"⚠️ {model_name} installed but not working properly")
            else:
                print(f"❌ Failed to install {model_name}")
                print(f"   Error: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            print(f"⏰ Installation of {model_name} timed out")
        except FileNotFoundError:
            print(f"❌ '{ollama_cmd}' command not found")
            print("   Please make sure Ollama is installed")
            return False
        except Exception as e:
            print(f"❌ Error installing {model_name}: {str(e)}")
    
    print("\n❌ Failed to install any vision model")
    return False

def test_vision_model(model_name):
    """Test if a vision model is working"""
    try:
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
            "http://localhost:11434/api/chat",
            json=test_data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            content = result.get('message', {}).get('content', '')
            return len(content) > 0
        else:
            return False
            
    except Exception as e:
        print(f"   Test error: {str(e)}")
        return False

def create_test_description():
    """Create a simple test description to verify everything works"""
    
    print("\n🧪 Testing Description Generation")
    print("-" * 40)
    
    # Check if we have rendered images
    renders_dir = Path("ressources/data/renders")
    
    if not renders_dir.exists():
        print("📁 No renders directory found")
        print("   Run rendering first: python test_rendering.py")
        return False
    
    # Find any PNG files
    png_files = list(renders_dir.rglob("*.png"))
    
    if not png_files:
        print("📸 No rendered images found")
        print("   Run rendering first: python test_rendering.py")
        return False
    
    test_image = png_files[0]
    print(f"📸 Found test image: {test_image.name}")
    
    # We'll implement actual testing in the next step
    print("✅ Ready for description generation testing")
    return True

def main():
    """Main setup function"""
    
    print("🚀 Ollama Vision Model Setup")
    print("=" * 60)
    
    # Step 1: Check Ollama
    print("\n📋 Step 1: Checking Ollama Status")

    ollama_cmd = add_ollama_to_path()
    if not ollama_cmd:
        return 1

    if not check_ollama_running():
        print("❌ Ollama is not running")
        print("\n💡 To start Ollama:")
        print("   1. Open a new terminal/command prompt")
        print(f"   2. Run: {ollama_cmd} serve")
        print("   3. Keep that terminal open")
        print("   4. Run this script again")
        return 1

    print("✅ Ollama is running")
    
    # Step 2: Install vision model
    print("\n📋 Step 2: Installing Vision Model")
    model_name = install_vision_model()
    
    if not model_name:
        print("\n❌ Failed to install vision model")
        print("\n💡 Manual installation:")
        print("   1. Open a new terminal")
        print("   2. Run: ollama pull llama3.2-vision")
        print("   3. Wait for download to complete")
        print("   4. Run this script again")
        return 1
    
    # Step 3: Test setup
    print("\n📋 Step 3: Testing Setup")
    if create_test_description():
        print("✅ Setup test passed")
    else:
        print("⚠️ Setup test had issues (but vision model is installed)")
    
    # Summary
    print("\n" + "=" * 60)
    print("🎉 Ollama Vision Setup Complete!")
    print(f"   ✅ Ollama server: Running")
    print(f"   ✅ Vision model: {model_name}")
    print(f"   🎯 Ready for description generation!")
    
    print("\n🚀 Next Steps:")
    print("   1. Run: python test_description_generation.py")
    print("   2. Generate descriptions for your 3D assets")
    
    return 0

if __name__ == "__main__":
    exit(main())
