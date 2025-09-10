#!/usr/bin/env python3
"""
Environment setup script for Fantasy Assets RAG system
Uses .venv environment
"""

import sys
import subprocess
from pathlib import Path
import os

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def create_venv():
    """Create .venv environment if it doesn't exist"""
    venv_path = Path(".venv")
    if venv_path.exists():
        print("✅ .venv environment already exists")
        return True

    print("📦 Creating .venv environment...")
    try:
        subprocess.check_call([sys.executable, "-m", "venv", ".venv"])
        print("✅ .venv environment created successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error creating .venv: {e}")
        return False

def get_venv_python():
    """Get path to Python executable in .venv"""
    if sys.platform == "win32":
        return Path(".venv/Scripts/python.exe")
    else:
        return Path(".venv/bin/python")

def install_requirements():
    """Install Python requirements in .venv"""
    print("📦 Installing Python requirements in .venv...")

    venv_python = get_venv_python()
    if not venv_python.exists():
        print("❌ .venv Python not found. Please create .venv first.")
        return False

    try:
        subprocess.check_call([str(venv_python), "-m", "pip", "install", "--upgrade", "pip"])
        subprocess.check_call([str(venv_python), "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully in .venv")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False

def check_blender():
    """Check if Blender is available"""
    try:
        result = subprocess.run(["blender", "--version"], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            print(f"✅ Blender found: {version_line}")
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    print("⚠️  Blender not found in PATH")
    print("   Please install Blender and add it to your PATH, or update config/settings.py")
    print("   Download from: https://www.blender.org/download/")
    return False

def check_ollama():
    """Check if Ollama is available"""
    try:
        result = subprocess.run(["ollama", "version"], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ Ollama found")
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    print("⚠️  Ollama not found")
    print("   Please install Ollama from: https://ollama.ai/")
    print("   Then run: ollama pull llama3.2-vision")
    return False

def create_directories():
    """Create necessary directories"""
    print("📁 Creating directories...")
    
    from fantasy_rag.config import create_directories as create_dirs
    create_dirs()
    
    print("✅ Directories created")

def setup_test_environment():
    """Setup test environment with sample data"""
    print("🧪 Setting up test environment...")
    
    # Import after requirements are installed
    from fantasy_rag.config import ASSETS_DIR
    
    # Create test assets directory
    test_assets_dir = ASSETS_DIR / "test"
    test_assets_dir.mkdir(exist_ok=True)
    
    # Create a simple test info file
    test_info = {
        "description": "Test assets directory for Fantasy RAG system",
        "expected_files": "Place 10 STL files here for testing",
        "supported_formats": [".stl", ".STL"]
    }
    
    from fantasy_rag.utils import save_json
    save_json(test_info, test_assets_dir / "README.json")
    
    print(f"✅ Test environment ready at: {test_assets_dir}")
    print(f"   Please place your 10 test STL files in: {ASSETS_DIR}")

def create_activation_script():
    """Create activation script for .venv"""
    if sys.platform == "win32":
        script_content = """@echo off
echo 🚀 Activating Fantasy RAG environment...
call .venv\\Scripts\\activate.bat
echo ✅ Environment activated!
echo 📁 STL files go in: ressources\\data\\assets
echo 🎯 Run: python -m fantasy_rag.setup to initialize
"""
        script_path = Path("activate_fantasy_rag.bat")
    else:
        script_content = """#!/bin/bash
echo "🚀 Activating Fantasy RAG environment..."
source .venv/bin/activate
echo "✅ Environment activated!"
echo "📁 STL files go in: ressources/data/assets"
echo "🎯 Run: python -m fantasy_rag.setup to initialize"
"""
        script_path = Path("activate_fantasy_rag.sh")

    try:
        with open(script_path, 'w') as f:
            f.write(script_content)

        if sys.platform != "win32":
            os.chmod(script_path, 0o755)

        print(f"✅ Activation script created: {script_path}")
        return True
    except Exception as e:
        print(f"❌ Error creating activation script: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 Setting up Fantasy Assets RAG Environment (.venv)")
    print("=" * 50)

    # Check Python version
    if not check_python_version():
        return False

    # Create .venv environment
    if not create_venv():
        return False

    # Install requirements in .venv
    if not install_requirements():
        return False

    # Create directories (using .venv Python)
    venv_python = get_venv_python()
    try:
        subprocess.check_call([str(venv_python), "-c",
                             "from fantasy_rag.config import create_directories; create_directories()"])
        print("✅ Directories created")
    except subprocess.CalledProcessError:
        print("⚠️  Could not create directories yet (will be created on first run)")

    # Check external dependencies
    blender_ok = check_blender()
    ollama_ok = check_ollama()

    # Create activation script
    create_activation_script()

    print("\n" + "=" * 50)
    print("🎯 Setup Summary:")
    print(f"✅ .venv environment: Created and configured")
    print(f"✅ Python packages: Installed")
    print(f"{'✅' if blender_ok else '⚠️ '} Blender: {'Ready' if blender_ok else 'Needs installation'}")
    print(f"{'✅' if ollama_ok else '⚠️ '} Ollama: {'Ready' if ollama_ok else 'Needs installation'}")

    print("\n🚀 Next Steps:")
    if sys.platform == "win32":
        print("1. Run: activate_fantasy_rag.bat")
    else:
        print("1. Run: ./activate_fantasy_rag.sh")
    print("2. Place STL files in: ressources/data/assets")
    print("3. Run: python -m fantasy_rag.asset_management.stl_loader")

    if not (blender_ok and ollama_ok):
        print("\n⚠️  Install missing dependencies:")
        if not blender_ok:
            print("   - Blender: https://www.blender.org/download/")
        if not ollama_ok:
            print("   - Ollama: https://ollama.ai/")

    return True

if __name__ == "__main__":
    main()
