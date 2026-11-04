"""
Mesh utilities for STL file processing
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import trimesh
from dataclasses import dataclass

@dataclass
class MeshMetadata:
    """Metadata extracted from mesh"""
    vertices_count: int
    faces_count: int
    volume: float
    surface_area: float
    bounding_box: Dict[str, Any]
    center: Tuple[float, float, float]
    is_watertight: bool
    is_valid: bool
    bounds: Dict[str, float]

def load_stl_mesh(file_path: Path) -> Optional[trimesh.Trimesh]:
    """
    Load STL file as trimesh object
    
    Args:
        file_path: Path to STL file
        
    Returns:
        Trimesh object or None if error
    """
    try:
        mesh = trimesh.load(str(file_path))
        
        # Handle case where trimesh returns a Scene instead of Mesh
        if isinstance(mesh, trimesh.Scene):
            # Get the first mesh from the scene
            if len(mesh.geometry) > 0:
                mesh = list(mesh.geometry.values())[0]
            else:
                print(f"❌ No geometry found in STL file: {file_path}")
                return None
                
        if not isinstance(mesh, trimesh.Trimesh):
            print(f"❌ Invalid mesh type in STL file: {file_path}")
            return None
            
        return mesh
        
    except Exception as e:
        print(f"❌ Error loading STL file {file_path}: {e}")
        return None

def extract_mesh_metadata(mesh: trimesh.Trimesh) -> MeshMetadata:
    """
    Extract comprehensive metadata from mesh
    
    Args:
        mesh: Trimesh object
        
    Returns:
        MeshMetadata object
    """
    try:
        # Basic counts
        vertices_count = len(mesh.vertices)
        faces_count = len(mesh.faces)
        
        # Volume and surface area
        volume = float(mesh.volume) if mesh.is_volume else 0.0
        surface_area = float(mesh.area)
        
        # Bounding box
        bounds = mesh.bounds
        bbox_size = bounds[1] - bounds[0]  # max - min
        
        bounding_box = {
            "min": bounds[0].tolist(),
            "max": bounds[1].tolist(),
            "size": bbox_size.tolist(),
            "dimensions": {
                "width": float(bbox_size[0]),
                "height": float(bbox_size[1]), 
                "depth": float(bbox_size[2])
            }
        }
        
        # Center point
        center = tuple(mesh.center_mass.tolist())
        
        # Quality checks
        is_watertight = mesh.is_watertight
        is_valid = mesh.is_valid if hasattr(mesh, 'is_valid') else True
        
        # Bounds dictionary
        bounds_dict = {
            "x_min": float(bounds[0][0]),
            "y_min": float(bounds[0][1]),
            "z_min": float(bounds[0][2]),
            "x_max": float(bounds[1][0]),
            "y_max": float(bounds[1][1]),
            "z_max": float(bounds[1][2])
        }
        
        return MeshMetadata(
            vertices_count=vertices_count,
            faces_count=faces_count,
            volume=volume,
            surface_area=surface_area,
            bounding_box=bounding_box,
            center=center,
            is_watertight=is_watertight,
            is_valid=is_valid,
            bounds=bounds_dict
        )
        
    except Exception as e:
        print(f"❌ Error extracting mesh metadata: {e}")
        # Return default metadata
        return MeshMetadata(
            vertices_count=0,
            faces_count=0,
            volume=0.0,
            surface_area=0.0,
            bounding_box={},
            center=(0.0, 0.0, 0.0),
            is_watertight=False,
            is_valid=False,
            bounds={}
        )

def normalize_mesh(mesh: trimesh.Trimesh, target_size: float = 1.0) -> trimesh.Trimesh:
    """
    Normalize mesh to fit within target size
    
    Args:
        mesh: Input mesh
        target_size: Target maximum dimension
        
    Returns:
        Normalized mesh
    """
    try:
        # Get current bounding box
        bounds = mesh.bounds
        current_size = np.max(bounds[1] - bounds[0])
        
        if current_size > 0:
            # Calculate scale factor
            scale_factor = target_size / current_size
            
            # Apply scaling
            mesh.apply_scale(scale_factor)
            
        # Center the mesh
        mesh.apply_translation(-mesh.center_mass)
        
        return mesh
        
    except Exception as e:
        print(f"❌ Error normalizing mesh: {e}")
        return mesh

def repair_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Attempt to repair mesh issues
    
    Args:
        mesh: Input mesh
        
    Returns:
        Repaired mesh
    """
    try:
        # Remove duplicate vertices
        mesh.remove_duplicate_faces()
        mesh.remove_degenerate_faces()
        
        # Fill holes if possible
        if hasattr(mesh, 'fill_holes'):
            mesh.fill_holes()
            
        # Remove unreferenced vertices
        mesh.remove_unreferenced_vertices()
        
        return mesh
        
    except Exception as e:
        print(f"❌ Error repairing mesh: {e}")
        return mesh

def get_mesh_complexity_score(metadata: MeshMetadata) -> float:
    """
    Calculate complexity score for mesh (0-1 scale)
    
    Args:
        metadata: Mesh metadata
        
    Returns:
        Complexity score between 0 and 1
    """
    try:
        # Factors for complexity
        vertex_factor = min(metadata.vertices_count / 50000, 1.0)  # Normalize to 50k vertices
        face_factor = min(metadata.faces_count / 100000, 1.0)     # Normalize to 100k faces
        
        # Surface area to volume ratio (higher = more complex surface)
        if metadata.volume > 0:
            sa_vol_ratio = metadata.surface_area / metadata.volume
            ratio_factor = min(sa_vol_ratio / 100, 1.0)  # Normalize
        else:
            ratio_factor = 0.5  # Default for non-volume meshes
            
        # Combine factors
        complexity = (vertex_factor * 0.4 + face_factor * 0.4 + ratio_factor * 0.2)
        
        return min(max(complexity, 0.0), 1.0)  # Clamp to [0, 1]
        
    except Exception as e:
        print(f"❌ Error calculating complexity score: {e}")
        return 0.5  # Default medium complexity

def validate_stl_file(file_path: Path) -> Dict[str, Any]:
    """
    Validate STL file and return comprehensive report
    
    Args:
        file_path: Path to STL file
        
    Returns:
        Validation report dictionary
    """
    report = {
        "file_path": str(file_path),
        "is_valid": False,
        "errors": [],
        "warnings": [],
        "metadata": None,
        "complexity_score": 0.0
    }
    
    try:
        # Check file exists
        if not file_path.exists():
            report["errors"].append("File does not exist")
            return report
            
        # Check file size
        file_size = file_path.stat().st_size
        if file_size == 0:
            report["errors"].append("File is empty")
            return report
            
        if file_size > 500 * 1024 * 1024:  # 500 MB
            report["warnings"].append("File is very large (>500MB)")
            
        # Load mesh
        mesh = load_stl_mesh(file_path)
        if mesh is None:
            report["errors"].append("Could not load mesh from file")
            return report
            
        # Extract metadata
        metadata = extract_mesh_metadata(mesh)
        report["metadata"] = metadata
        
        # Validation checks
        if metadata.vertices_count == 0:
            report["errors"].append("Mesh has no vertices")
            
        if metadata.faces_count == 0:
            report["errors"].append("Mesh has no faces")
            
        if not metadata.is_valid:
            report["warnings"].append("Mesh failed trimesh validation")
            
        if not metadata.is_watertight:
            report["warnings"].append("Mesh is not watertight")
            
        # Calculate complexity
        report["complexity_score"] = get_mesh_complexity_score(metadata)
        
        # Mark as valid if no critical errors
        report["is_valid"] = len(report["errors"]) == 0
        
    except Exception as e:
        report["errors"].append(f"Validation error: {str(e)}")
        
    return report
