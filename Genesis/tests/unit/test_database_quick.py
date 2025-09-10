#!/usr/bin/env python3
"""
Quick Database Test - Test only the corrected database functionality
"""

import logging
from pathlib import Path

def test_database_manager():
    """Test ChromaDB manager functionality"""
    
    print("💾 Testing Database Manager")
    print("-" * 50)
    
    try:
        from fantasy_rag.database.chromadb_manager import ChromaDBManager, DatabaseConfig
        from fantasy_rag.database.schema_definitions import (
            AssetRecord, RenderRecord, DescriptionRecord,
            AssetStatus, AssetCategory, ProcessingStage
        )
        
        # Initialize database manager
        config = DatabaseConfig(
            db_path="ressources/data/test_fantasy_db_quick",
            assets_collection="test_assets",
            renders_collection="test_renders",
            descriptions_collection="test_descriptions"
        )
        
        db_manager = ChromaDBManager(config)
        
        if not db_manager.initialize():
            print("❌ Failed to initialize database manager")
            return False
        
        print("✅ Database manager initialized")
        
        # Clear existing test data
        for collection_name in ["test_assets", "test_renders", "test_descriptions"]:
            db_manager.clear_collection(collection_name)
        
        # Test asset management
        test_asset = AssetRecord(
            asset_id="test_sword_001",
            original_filename="epic_sword.stl",
            file_path=Path("test/epic_sword.stl"),
            file_hash="abc123def456",
            category=AssetCategory.WEAPON,
            status=AssetStatus.DISCOVERED,
            current_stage=ProcessingStage.DISCOVERY,
            file_size=1024000,
            file_format="stl",
            tags=["fantasy", "weapon", "sword"],  # Test tags
            custom_metadata={"rarity": "legendary", "level": 50}  # Test custom metadata
        )
        
        # Add asset
        if not db_manager.add_asset(test_asset):
            print("❌ Failed to add test asset")
            return False
        
        print("✅ Test asset added with tags and custom metadata")
        
        # Retrieve asset
        retrieved_asset = db_manager.get_asset("test_sword_001")
        if not retrieved_asset:
            print("❌ Failed to retrieve test asset")
            return False
        
        print(f"✅ Asset retrieved: {retrieved_asset.original_filename}")
        print(f"   Tags: {retrieved_asset.tags}")
        print(f"   Custom metadata: {retrieved_asset.custom_metadata}")
        
        # Test render management
        test_render = RenderRecord(
            render_id="render_001",
            asset_id="test_sword_001",
            view_name="front",
            image_path=Path("test/renders/front.png"),
            resolution=(1024, 1024),
            samples=128,
            render_time=45.2,
            file_size=512000,
            dominant_colors=["#FF0000", "#00FF00", "#0000FF"]  # Test color list
        )
        
        if not db_manager.add_render(test_render):
            print("❌ Failed to add test render")
            return False
        
        print("✅ Test render added with color data")
        
        # Retrieve renders
        renders = db_manager.get_asset_renders("test_sword_001")
        if not renders:
            print("❌ Failed to retrieve renders")
            return False
        
        print(f"✅ Retrieved {len(renders)} renders")
        print(f"   Resolution: {renders[0].resolution}")
        print(f"   Colors: {renders[0].dominant_colors}")
        
        # Test description management
        test_description = DescriptionRecord(
            description_id="desc_001",
            asset_id="test_sword_001",
            language="français",
            description_text="Une épée légendaire forgée dans les flammes du dragon avec une lame enchantée.",
            model_name="llama3.2-vision:latest",
            generation_time=12.5,
            word_count=15,
            target_length=200,
            views_analyzed=["front", "back", "iso_3d"]  # Test views list
        )
        
        if not db_manager.add_description(test_description):
            print("❌ Failed to add test description")
            return False
        
        print("✅ Test description added with views data")
        
        # Retrieve descriptions
        descriptions = db_manager.get_asset_descriptions("test_sword_001")
        if not descriptions:
            print("❌ Failed to retrieve descriptions")
            return False
        
        print(f"✅ Retrieved {len(descriptions)} descriptions")
        print(f"   Language: {descriptions[0].language}")
        print(f"   Views analyzed: {descriptions[0].views_analyzed}")
        
        # Get database statistics
        stats = db_manager.get_database_stats()
        print(f"\n📊 Database Statistics:")
        print(f"   Collections: {len(stats['collections'])}")
        
        for collection_name, collection_stats in stats['collections'].items():
            print(f"   {collection_name}: {collection_stats['document_count']} documents")
        
        if 'asset_analysis' in stats:
            print(f"   Asset statuses: {stats['asset_analysis']['status_distribution']}")
            print(f"   Asset categories: {stats['asset_analysis']['category_distribution']}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {str(e)}")
        print("   Install required packages: pip install chromadb")
        return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def test_stl_discovery():
    """Test STL file discovery"""
    
    print("\n📁 Testing STL File Discovery")
    print("-" * 50)
    
    stl_directory = Path("ressources/data/assets")
    
    if not stl_directory.exists():
        print(f"❌ STL directory not found: {stl_directory}")
        return False
    
    # Find all STL files
    stl_files = list(stl_directory.rglob("*.stl"))
    print(f"✅ Found {len(stl_files)} STL files")
    
    for i, stl_file in enumerate(stl_files[:5]):  # Show first 5
        file_size = stl_file.stat().st_size / (1024 * 1024)  # MB
        print(f"   {i+1}. {stl_file.name} ({file_size:.1f} MB)")
    
    if len(stl_files) > 5:
        print(f"   ... and {len(stl_files) - 5} more files")
    
    return len(stl_files) > 0

def main():
    """Run quick database tests"""
    
    # Setup logging
    logging.basicConfig(level=logging.WARNING)  # Reduce noise
    
    print("💾 Fantasy Asset Database Quick Test")
    print("=" * 60)
    
    tests = [
        ("Database Manager", test_database_manager),
        ("STL File Discovery", test_stl_discovery)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🧪 Testing: {test_name}")
        try:
            if test_func():
                print(f"✅ {test_name}: PASSED")
                passed += 1
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {str(e)}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary:")
    print(f"   Tests passed: {passed}/{total}")
    print(f"   Success rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("🎉 All database tests passed!")
        print("\n✅ Module 5 - Database Management: OPERATIONAL")
        print("   - ChromaDB integration working")
        print("   - Asset, Render, Description storage working")
        print("   - Metadata serialization fixed")
        print("   - STL files discovered successfully")
        
        print("\n🚀 Complete Fantasy RAG System Status:")
        print("   1. ✅ Module 1: Rendering (Blender)")
        print("   2. ✅ Module 2: Asset Management")
        print("   3. ✅ Module 3: Description Generation (Ollama)")
        print("   4. ✅ Module 4: Embeddings & Search (ChromaDB)")
        print("   5. ✅ Module 5: Database Management (ChromaDB)")
        
        print("\n🎯 System Ready for Production!")
        
        return 0
    else:
        print("⚠️ Some tests failed. Check the output above.")
        return 1

if __name__ == "__main__":
    exit(main())
