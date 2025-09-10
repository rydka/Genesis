#!/usr/bin/env python3
"""
Quick test script for Fantasy Assets Rendering Engine
Uses reduced settings for faster testing
"""

import sys
import logging
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from fantasy_rag.asset_management.stl_loader import STLLoader
from fantasy_rag.rendering.view_generator import ViewGenerator
from fantasy_rag.rendering.blender_renderer import BlenderRenderer

def main():
    """Quick test of the Rendering Engine"""
    
    # Setup logging
    logging.basicConfig(level=logging.WARNING)  # Reduce log noise
    
    print("🎨 Quick Rendering Engine Test")
    print("=" * 50)
    
    try:
        # Step 1: Quick Blender verification
        print("🔧 Verifying Blender...")
        
        # Use reduced settings for quick test
        renderer = BlenderRenderer(
            resolution=(512, 512),  # Smaller resolution
            samples=32  # Fewer samples
        )
        
        if not renderer.verify_blender():
            print("❌ Blender verification failed!")
            return 1
        
        print("✅ Blender OK (512x512, 32 samples)")
        
        # Step 2: Load one test asset
        print("\n📂 Loading test asset...")
        loader = STLLoader()
        
        # Just discover files, don't load all meshes
        stl_files = loader.discover_stl_files()
        if not stl_files:
            print("❌ No STL files found")
            return 1
        
        # Create asset object for first file (without loading mesh)
        test_file = stl_files[0]
        test_asset = loader.create_asset_from_file(test_file)
        
        # Load just this one asset
        if not loader.load_mesh_data(test_asset):
            print(f"❌ Failed to load {test_asset.file_name}")
            return 1
        
        print(f"✅ Loaded: {test_asset.file_name}")
        print(f"   Complexity: {test_asset.vertices_count:,} vertices")
        
        # Step 3: Quick render test
        print(f"\n🎯 Testing render (this may take 1-2 minutes)...")
        
        generator = ViewGenerator(renderer)
        result = generator.generate_views_for_asset(test_asset)
        
        if result.success:
            print(f"✅ Render successful!")
            print(f"   Time: {result.render_time:.1f}s")
            print(f"   Views: {len(result.views)}/6")
            
            # Show file sizes
            total_size = 0
            for view_name, view_path in result.views.items():
                size_mb = view_path.stat().st_size / 1024 / 1024
                total_size += size_mb
                print(f"   📸 {view_name}: {size_mb:.1f} MB")
            
            print(f"   Total: {total_size:.1f} MB")
            print(f"\n📁 Output: {renderer.output_dir / test_asset.asset_id}")
            
        else:
            print(f"❌ Render failed: {result.error_message}")
            return 1
        
        print(f"\n🎉 Quick test completed successfully!")
        print(f"   Ready to render all {len(stl_files)} assets")
        
        return 0
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Test interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main())
