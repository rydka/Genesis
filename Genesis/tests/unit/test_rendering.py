#!/usr/bin/env python3
"""
Test script for Fantasy Assets Rendering Engine
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
    """Test the Rendering Engine with fantasy assets"""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("🎨 Fantasy Assets Rendering Engine Test")
    print("=" * 60)
    
    try:
        # Step 1: Verify Blender
        print("🔧 Step 1: Verifying Blender...")
        renderer = BlenderRenderer()
        
        if not renderer.verify_blender():
            print("❌ Blender verification failed!")
            print("Please ensure Blender is installed and accessible")
            return 1
        
        print("✅ Blender verification successful")
        print(f"   Output directory: {renderer.output_dir}")
        print(f"   Resolution: {renderer.resolution[0]}x{renderer.resolution[1]}")
        print(f"   Samples: {renderer.samples}")
        
        # Step 2: Load STL assets
        print("\n📂 Step 2: Loading STL assets...")
        loader = STLLoader()
        assets = loader.load_all_assets()
        
        loaded_assets = loader.get_loaded_assets()
        print(f"✅ Loaded {len(loaded_assets)} STL assets")
        
        if not loaded_assets:
            print("❌ No assets loaded. Cannot test rendering.")
            return 1
        
        # Step 3: Test single asset rendering
        print("\n🎯 Step 3: Testing single asset rendering...")
        test_asset = loaded_assets[0]  # Use first asset for test
        print(f"   Testing with: {test_asset.file_name}")
        print(f"   Asset complexity: {test_asset.vertices_count:,} vertices")
        
        generator = ViewGenerator(renderer)
        result = generator.generate_views_for_asset(test_asset)
        
        if result.success:
            print(f"✅ Single asset render successful!")
            print(f"   Render time: {result.render_time:.1f}s")
            print(f"   Views generated: {len(result.views)}/6")
            
            # List generated views
            for view_name, view_path in result.views.items():
                file_size = view_path.stat().st_size / 1024 / 1024  # MB
                print(f"   📸 {view_name}: {view_path.name} ({file_size:.1f} MB)")
        else:
            print(f"❌ Single asset render failed: {result.error_message}")
            return 1
        
        # Step 4: Ask user about batch rendering
        print(f"\n🚀 Step 4: Batch rendering option")
        print(f"   Found {len(loaded_assets)} assets total")
        print(f"   Estimated time: ~{len(loaded_assets) * result.render_time / 60:.1f} minutes")
        
        response = input("\n   Render all assets? (y/N): ").strip().lower()
        
        if response == 'y':
            print(f"\n🎨 Batch rendering {len(loaded_assets)} assets...")
            
            batch_result = generator.generate_views_batch(loaded_assets)
            summary = generator.get_render_summary(batch_result)
            
            print(f"\n📊 Batch Rendering Results:")
            print(f"   Total assets: {summary['total_assets']}")
            print(f"   Successful: {summary['successful_renders']}")
            print(f"   Failed: {summary['failed_renders']}")
            print(f"   Total time: {summary['total_time']:.1f}s ({summary['total_time']/60:.1f} min)")
            print(f"   Total images: {summary['total_images']}")
            print(f"   Average render time: {summary['avg_render_time']:.1f}s per asset")
            print(f"   Render speed: {summary['images_per_second']:.2f} images/second")
            
            if batch_result.errors:
                print(f"\n❌ Errors encountered:")
                for error in batch_result.errors[:5]:  # Show first 5 errors
                    print(f"   - {error}")
                if len(batch_result.errors) > 5:
                    print(f"   ... and {len(batch_result.errors) - 5} more errors")
            
            if summary['successful_renders'] > 0:
                print(f"\n✅ Batch rendering completed!")
                print(f"   Generated {summary['total_images']} images")
                print(f"   Output directory: {renderer.output_dir}")
            else:
                print(f"\n❌ Batch rendering failed completely")
                return 1
        else:
            print("   Skipping batch rendering")
        
        # Step 5: Verification
        print(f"\n🔍 Step 5: Verifying rendered views...")
        
        # Check the test asset views
        view_status = generator.verify_rendered_views(test_asset.asset_id)
        missing_views = [view for view, exists in view_status.items() if not exists]
        
        if not missing_views:
            print(f"✅ All 6 views verified for test asset")
        else:
            print(f"⚠️  Missing views for test asset: {missing_views}")
        
        print(f"\n🎉 Rendering Engine test completed!")
        print(f"   Ready for LLM description generation")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error during rendering test: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
