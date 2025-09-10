#!/usr/bin/env python3
"""
Test script for STL Loader
"""

import sys
import logging
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from fantasy_rag.asset_management.stl_loader import STLLoader

def main():
    """Test the STL Loader with our 12 fantasy assets"""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("🎮 Fantasy Assets STL Loader Test")
    print("=" * 50)
    
    try:
        # Initialize loader
        loader = STLLoader()
        
        # Load all assets
        print("📂 Loading STL assets...")
        assets = loader.load_all_assets()
        
        # Get summary
        summary = loader.get_asset_summary()
        
        print(f"\n📊 Loading Results:")
        print(f"  Total assets: {summary['total_assets']}")
        print(f"  Successfully loaded: {summary['loaded_assets']}")
        print(f"  Failed to load: {summary['failed_assets']}")
        
        if summary['loaded_assets'] > 0:
            print(f"\n📈 Mesh Statistics:")
            print(f"  Total vertices: {summary['total_vertices']:,}")
            print(f"  Total faces: {summary['total_faces']:,}")
            print(f"  Average vertices per asset: {summary['avg_vertices']:,}")
            print(f"  Average faces per asset: {summary['avg_faces']:,}")
            print(f"  Total volume: {summary['total_volume']:.2f} cubic units")
            print(f"  Total file size: {summary['total_file_size'] / 1024 / 1024:.1f} MB")
            
            print(f"\n📏 Complexity Range:")
            complexity = summary['complexity_range']
            print(f"  Vertices: {complexity['min_vertices']:,} - {complexity['max_vertices']:,}")
            print(f"  Faces: {complexity['min_faces']:,} - {complexity['max_faces']:,}")
            
            print(f"\n🎯 Loaded Assets Details:")
            for asset in loader.get_loaded_assets():
                size_mb = asset.file_size / 1024 / 1024
                print(f"  📄 {asset.file_name}")
                print(f"     ID: {asset.asset_id}")
                print(f"     Vertices: {asset.vertices_count:,}")
                print(f"     Faces: {asset.faces_count:,}")
                print(f"     Volume: {asset.volume:.2f}")
                print(f"     File size: {size_mb:.1f} MB")
                print(f"     Bounding box: {asset.bounding_box}")
                print()
        
        print("✅ STL Loader test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during STL loading: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
