#!/usr/bin/env python3
"""
Test script for Fantasy Assets RAG environment
"""

import sys
from pathlib import Path

def test_imports():
    """Test if all required packages can be imported"""
    print("🧪 Testing package imports...")
    
    packages = [
        ("numpy", "np"),
        ("pandas", "pd"), 
        ("PIL", "Image"),
        ("cv2", None),
        ("trimesh", None),
        ("torch", None),
        ("transformers", None),
        ("sentence_transformers", None),
        ("sklearn", None),
        ("chromadb", None),
        ("streamlit", "st"),
        ("tqdm", None)
    ]
    
    success_count = 0
    for package, alias in packages:
        try:
            if alias:
                exec(f"import {package} as {alias}")
            else:
                exec(f"import {package}")
            print(f"  ✅ {package}")
            success_count += 1
        except ImportError as e:
            print(f"  ❌ {package}: {e}")
    
    print(f"\n📊 Import Results: {success_count}/{len(packages)} packages available")
    return success_count == len(packages)

def test_directories():
    """Test if project directories exist"""
    print("\n📁 Testing directory structure...")
    
    directories = [
        "fantasy_rag",
        "fantasy_rag/config",
        "fantasy_rag/utils",
        "ressources",
        "ressources/data",
        "ressources/data/assets"
    ]
    
    success_count = 0
    for directory in directories:
        path = Path(directory)
        if path.exists():
            print(f"  ✅ {directory}")
            success_count += 1
        else:
            print(f"  ❌ {directory} (missing)")
    
    print(f"\n📊 Directory Results: {success_count}/{len(directories)} directories found")
    return success_count == len(directories)

def test_config():
    """Test if configuration can be loaded"""
    print("\n⚙️  Testing configuration...")
    
    try:
        from fantasy_rag.config import settings, ASSETS_DIR
        print(f"  ✅ Configuration loaded")
        print(f"  📁 Assets directory: {ASSETS_DIR}")
        print(f"  🎨 Render resolution: {settings.rendering.resolution}")
        print(f"  🧠 LLM model: {settings.llm.model_name}")
        return True
    except Exception as e:
        print(f"  ❌ Configuration error: {e}")
        return False

def test_utilities():
    """Test utility functions"""
    print("\n🔧 Testing utilities...")
    
    try:
        from fantasy_rag.utils import file_utils, mesh_utils, image_utils
        
        # Test file utils
        test_files = file_utils.get_stl_files(Path("ressources/data/assets"))
        print(f"  ✅ File utils: Found {len(test_files)} STL files")
        
        # Test mesh utils (basic import)
        print(f"  ✅ Mesh utils: Available")
        
        # Test image utils (basic import)
        print(f"  ✅ Image utils: Available")
        
        return True
    except Exception as e:
        print(f"  ❌ Utilities error: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 Fantasy Assets RAG Environment Test")
    print("=" * 50)
    
    print(f"🐍 Python version: {sys.version}")
    print(f"📁 Working directory: {Path.cwd()}")
    
    # Run tests
    imports_ok = test_imports()
    directories_ok = test_directories()
    config_ok = test_config()
    utils_ok = test_utilities()
    
    # Summary
    print("\n" + "=" * 50)
    print("🎯 Test Summary:")
    print(f"{'✅' if imports_ok else '❌'} Package imports")
    print(f"{'✅' if directories_ok else '❌'} Directory structure")
    print(f"{'✅' if config_ok else '❌'} Configuration")
    print(f"{'✅' if utils_ok else '❌'} Utilities")
    
    all_ok = imports_ok and directories_ok and config_ok and utils_ok
    
    if all_ok:
        print("\n🎉 Environment test PASSED! Ready for Fantasy RAG development.")
    else:
        print("\n⚠️  Environment test FAILED. Please check errors above.")
        
    return all_ok

if __name__ == "__main__":
    main()
