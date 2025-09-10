#!/usr/bin/env python3
"""
STL Loader for Fantasy Assets RAG
Handles loading and basic processing of STL files
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from datetime import datetime

import trimesh
import numpy as np

from ..config.settings import ASSETS_DIR
from ..utils.file_utils import get_stl_files, get_file_hash
from ..utils.mesh_utils import load_stl_mesh, extract_mesh_metadata

logger = logging.getLogger(__name__)

@dataclass
class STLAsset:
    """Represents a loaded STL asset with metadata"""
    
    # File information
    file_path: Path
    file_name: str
    file_hash: str
    file_size: int
    
    # Mesh data
    mesh: Optional[trimesh.Trimesh] = None
    
    # Metadata
    vertices_count: int = 0
    faces_count: int = 0
    volume: float = 0.0
    surface_area: float = 0.0
    bounding_box: tuple = None
    center_mass: tuple = None
    
    # Processing status
    loaded: bool = False
    load_error: Optional[str] = None
    load_time: Optional[datetime] = None
    
    # Asset ID for database
    asset_id: str = ""

class STLLoader:
    """
    Loads and processes STL files for the Fantasy RAG system
    """
    
    def __init__(self, assets_directory: Optional[Path] = None):
        """
        Initialize STL Loader
        
        Args:
            assets_directory: Directory containing STL files
        """
        self.assets_directory = assets_directory or ASSETS_DIR
        self.loaded_assets: Dict[str, STLAsset] = {}
        
        logger.info(f"STL Loader initialized with directory: {self.assets_directory}")
    
    def discover_stl_files(self) -> List[Path]:
        """
        Discover all STL files in the assets directory
        
        Returns:
            List of STL file paths
        """
        logger.info(f"Discovering STL files in: {self.assets_directory}")
        
        stl_files = get_stl_files(self.assets_directory)
        
        logger.info(f"Found {len(stl_files)} STL files")
        for file_path in stl_files:
            logger.debug(f"  - {file_path.name}")
            
        return stl_files
    
    def create_asset_from_file(self, file_path: Path) -> STLAsset:
        """
        Create an STLAsset object from a file path
        
        Args:
            file_path: Path to STL file
            
        Returns:
            STLAsset object with basic file information
        """
        # Generate asset ID from file hash
        file_hash = get_file_hash(file_path)
        asset_id = f"stl_{file_hash[:12]}"
        
        # Get file stats
        file_stats = file_path.stat()
        
        asset = STLAsset(
            file_path=file_path,
            file_name=file_path.name,
            file_hash=file_hash,
            file_size=file_stats.st_size,
            asset_id=asset_id
        )
        
        logger.debug(f"Created asset: {asset.asset_id} for {asset.file_name}")
        return asset
    
    def load_mesh_data(self, asset: STLAsset) -> bool:
        """
        Load mesh data for an STL asset
        
        Args:
            asset: STLAsset to load mesh data for
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.debug(f"Loading mesh data for: {asset.file_name}")
            start_time = datetime.now()
            
            # Load mesh using utility function
            mesh = load_stl_mesh(asset.file_path)
            if mesh is None:
                asset.load_error = "Failed to load mesh"
                return False
            
            # Get metadata
            metadata = extract_mesh_metadata(mesh)
            
            # Update asset with mesh data
            asset.mesh = mesh
            asset.vertices_count = metadata.vertices_count
            asset.faces_count = metadata.faces_count
            asset.volume = metadata.volume
            asset.surface_area = metadata.surface_area
            asset.bounding_box = metadata.bounding_box
            asset.center_mass = metadata.center
            
            # Mark as loaded
            asset.loaded = True
            asset.load_time = datetime.now()
            
            load_duration = (asset.load_time - start_time).total_seconds()
            logger.info(f"Loaded {asset.file_name} in {load_duration:.2f}s - "
                       f"{asset.vertices_count:,} vertices, {asset.faces_count:,} faces")
            
            return True
            
        except Exception as e:
            error_msg = f"Error loading mesh data: {str(e)}"
            logger.error(f"Failed to load {asset.file_name}: {error_msg}")
            asset.load_error = error_msg
            return False
    
    def load_all_assets(self) -> Dict[str, STLAsset]:
        """
        Discover and load all STL assets
        
        Returns:
            Dictionary of loaded assets (asset_id -> STLAsset)
        """
        logger.info("Starting batch loading of all STL assets")
        
        # Discover files
        stl_files = self.discover_stl_files()
        
        if not stl_files:
            logger.warning("No STL files found")
            return {}
        
        # Process each file
        loaded_count = 0
        failed_count = 0
        
        for file_path in stl_files:
            try:
                # Create asset
                asset = self.create_asset_from_file(file_path)
                
                # Load mesh data
                if self.load_mesh_data(asset):
                    self.loaded_assets[asset.asset_id] = asset
                    loaded_count += 1
                else:
                    failed_count += 1
                    
            except Exception as e:
                logger.error(f"Failed to process {file_path.name}: {str(e)}")
                failed_count += 1
        
        logger.info(f"Asset loading complete: {loaded_count} loaded, {failed_count} failed")
        return self.loaded_assets
    
    def get_asset_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of loaded assets
        
        Returns:
            Dictionary with summary statistics
        """
        if not self.loaded_assets:
            return {"total_assets": 0}
        
        loaded_assets = [a for a in self.loaded_assets.values() if a.loaded]
        
        if not loaded_assets:
            return {"total_assets": len(self.loaded_assets), "loaded_assets": 0}
        
        # Calculate statistics
        total_vertices = sum(a.vertices_count for a in loaded_assets)
        total_faces = sum(a.faces_count for a in loaded_assets)
        total_volume = sum(a.volume for a in loaded_assets)
        total_size = sum(a.file_size for a in loaded_assets)
        
        avg_vertices = total_vertices / len(loaded_assets)
        avg_faces = total_faces / len(loaded_assets)
        
        return {
            "total_assets": len(self.loaded_assets),
            "loaded_assets": len(loaded_assets),
            "failed_assets": len(self.loaded_assets) - len(loaded_assets),
            "total_vertices": total_vertices,
            "total_faces": total_faces,
            "total_volume": total_volume,
            "total_file_size": total_size,
            "avg_vertices": int(avg_vertices),
            "avg_faces": int(avg_faces),
            "complexity_range": {
                "min_vertices": min(a.vertices_count for a in loaded_assets),
                "max_vertices": max(a.vertices_count for a in loaded_assets),
                "min_faces": min(a.faces_count for a in loaded_assets),
                "max_faces": max(a.faces_count for a in loaded_assets)
            }
        }
    
    def get_asset_by_name(self, name: str) -> Optional[STLAsset]:
        """
        Get asset by file name
        
        Args:
            name: File name to search for
            
        Returns:
            STLAsset if found, None otherwise
        """
        for asset in self.loaded_assets.values():
            if asset.file_name == name or asset.file_name.replace('.stl', '') == name:
                return asset
        return None
    
    def get_loaded_assets(self) -> List[STLAsset]:
        """
        Get list of successfully loaded assets
        
        Returns:
            List of loaded STLAsset objects
        """
        return [asset for asset in self.loaded_assets.values() if asset.loaded]


def main():
    """Test the STL Loader"""
    logging.basicConfig(level=logging.INFO)
    
    loader = STLLoader()
    assets = loader.load_all_assets()
    
    print(f"\n🎯 STL Loader Test Results:")
    print(f"{'='*50}")
    
    summary = loader.get_asset_summary()
    print(f"📊 Assets loaded: {summary['loaded_assets']}/{summary['total_assets']}")
    
    if summary['loaded_assets'] > 0:
        print(f"📈 Total vertices: {summary['total_vertices']:,}")
        print(f"📈 Total faces: {summary['total_faces']:,}")
        print(f"📏 Average complexity: {summary['avg_vertices']:,} vertices, {summary['avg_faces']:,} faces")
        print(f"💾 Total file size: {summary['total_file_size'] / 1024 / 1024:.1f} MB")
        
        print(f"\n🎮 Loaded Assets:")
        for asset in loader.get_loaded_assets():
            print(f"  - {asset.file_name}: {asset.vertices_count:,} vertices")


if __name__ == "__main__":
    main()
