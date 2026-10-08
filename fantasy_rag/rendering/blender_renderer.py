#!/usr/bin/env python3
"""
Blender Renderer for Fantasy Assets RAG
Handles 6-view rendering of STL assets using Blender
"""

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import json
import time

from ..config.settings import RENDERS_DIR
from ..config.settings import settings

logger = logging.getLogger(__name__)

@dataclass
class RenderView:
    """Represents a single rendered view"""
    name: str
    camera_position: Tuple[float, float, float]
    camera_rotation: Tuple[float, float, float]
    description: str
    priority: int = 1  # 1=highest, 5=lowest
    
@dataclass
class RenderResult:
    """Result of a rendering operation"""
    asset_id: str
    asset_name: str
    views: Dict[str, Path]  # view_name -> image_path
    render_time: float
    success: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None

class BlenderRenderer:
    """
    Handles 6-view rendering of STL assets using Blender
    """
    
    # 6-view configuration optimized for LLM analysis and MeshFormer
    RENDER_VIEWS = {
        "front": RenderView(
            name="front",
            camera_position=(0, -8, 0),
            camera_rotation=(90, 0, 0),
            description="Vue de face principale",
            priority=1
        ),
        "back": RenderView(
            name="back", 
            camera_position=(0, 8, 0),
            camera_rotation=(90, 0, 180),
            description="Vue arrière",
            priority=3
        ),
        "left": RenderView(
            name="left",
            camera_position=(-8, 0, 0),
            camera_rotation=(90, 0, 90),
            description="Profil gauche",
            priority=2
        ),
        "right": RenderView(
            name="right",
            camera_position=(8, 0, 0),
            camera_rotation=(90, 0, -90),
            description="Profil droit", 
            priority=2
        ),
        "top": RenderView(
            name="top",
            camera_position=(0, 0, 8),
            camera_rotation=(0, 0, 0),
            description="Vue du dessus",
            priority=3
        ),
        "iso_3d": RenderView(
            name="iso_3d",
            camera_position=(6, -6, 4),
            camera_rotation=(60, 0, 45),
            description="Isométrique 3/4 avant (référence MeshFormer)",
            priority=1
        )
    }
    
    def __init__(self, 
                 blender_executable: Optional[str] = None,
                 output_dir: Optional[Path] = None,
                 resolution: Tuple[int, int] = (1024, 1024),
                 samples: int = 128):
        """
        Initialize Blender Renderer
        
        Args:
            blender_executable: Path to Blender executable
            output_dir: Directory for rendered images
            resolution: Render resolution (width, height)
            samples: Cycles render samples
        """
        self.blender_executable = blender_executable or settings.blender.blender_executable
        self.output_dir = output_dir or RENDERS_DIR
        self.resolution = resolution
        self.samples = samples
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Blender Renderer initialized:")
        logger.info(f"  Blender: {self.blender_executable}")
        logger.info(f"  Output: {self.output_dir}")
        logger.info(f"  Resolution: {self.resolution[0]}x{self.resolution[1]}")
        logger.info(f"  Samples: {self.samples}")
    
    def verify_blender(self) -> bool:
        """
        Verify that Blender is accessible and working
        
        Returns:
            True if Blender is working, False otherwise
        """
        try:
            result = subprocess.run(
                [self.blender_executable, "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                version_info = result.stdout.strip()
                logger.info(f"Blender verification successful: {version_info.split()[1]}")
                return True
            else:
                logger.error(f"Blender verification failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error verifying Blender: {str(e)}")
            return False
    
    def create_blender_script(self, 
                            stl_path: Path, 
                            output_dir: Path,
                            asset_id: str) -> str:
        """
        Create Blender Python script for 6-view rendering
        
        Args:
            stl_path: Path to STL file
            output_dir: Directory for output images
            asset_id: Unique asset identifier
            
        Returns:
            Blender Python script as string
        """
        script = f'''
import bpy
import bmesh
import mathutils
import os
from pathlib import Path

# Clear existing mesh objects
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# Import STL (Blender 4.5+ uses wm.stl_import)
stl_path = r"{stl_path}"
bpy.ops.wm.stl_import(filepath=stl_path)

# Get imported object
obj = bpy.context.selected_objects[0]
bpy.context.view_layer.objects.active = obj

# Center object at origin
bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
obj.location = (0, 0, 0)

# Calculate bounding box for camera distance
bbox = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
max_dim = max(max(bbox, key=lambda x: x[i])[i] - min(bbox, key=lambda x: x[i])[i] for i in range(3))
camera_distance = max_dim * 1.5

# Setup scene
scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.render.resolution_x = {self.resolution[0]}
scene.render.resolution_y = {self.resolution[1]}
scene.cycles.samples = {self.samples}

# Setup lighting
bpy.ops.object.light_add(type='SUN', location=(5, 5, 10))
sun = bpy.context.object
sun.data.energy = 3

# Add fill light
bpy.ops.object.light_add(type='AREA', location=(-5, -5, 5))
area = bpy.context.object
area.data.energy = 2
area.data.size = 10

# Setup camera
bpy.ops.object.camera_add()
camera = bpy.context.object
scene.camera = camera

# Material setup for better visibility
mat = bpy.data.materials.new(name="AssetMaterial")
mat.use_nodes = True
bsdf = mat.node_tree.nodes["Principled BSDF"]
bsdf.inputs[0].default_value = (0.8, 0.8, 0.8, 1.0)  # Base Color
bsdf.inputs[7].default_value = 0.3  # Roughness
obj.data.materials.append(mat)

# Render views
views = {{
    "front": ((0, -camera_distance, 0), (1.5708, 0, 0)),
    "back": ((0, camera_distance, 0), (1.5708, 0, 3.14159)),
    "left": ((-camera_distance, 0, 0), (1.5708, 0, 1.5708)),
    "right": ((camera_distance, 0, 0), (1.5708, 0, -1.5708)),
    "top": ((0, 0, camera_distance), (0, 0, 0)),
    "iso_3d": ((camera_distance*0.75, -camera_distance*0.75, camera_distance*0.5), (1.0472, 0, 0.7854))
}}

output_dir = Path(r"{output_dir}")
output_dir.mkdir(parents=True, exist_ok=True)

for view_name, (location, rotation) in views.items():
    # Position camera
    camera.location = location
    camera.rotation_euler = rotation
    
    # Point camera at object
    direction = mathutils.Vector((0, 0, 0)) - mathutils.Vector(location)
    camera.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
    
    # Render
    output_path = output_dir / f"{asset_id}_{{view_name}}.png"
    scene.render.filepath = str(output_path)
    bpy.ops.render.render(write_still=True)

    print(f"Rendered {{view_name}}: {{output_path}}")

print("All views rendered successfully")
'''
        return script
    
    def render_asset_views(self, 
                          stl_path: Path, 
                          asset_id: str,
                          asset_name: str) -> RenderResult:
        """
        Render 6 views of an STL asset
        
        Args:
            stl_path: Path to STL file
            asset_id: Unique asset identifier
            asset_name: Human-readable asset name
            
        Returns:
            RenderResult with paths to rendered images
        """
        start_time = time.time()
        
        logger.info(f"Starting 6-view render for: {asset_name}")
        
        # Create output directory for this asset
        asset_output_dir = self.output_dir / asset_id
        asset_output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Create Blender script
            script_content = self.create_blender_script(stl_path, asset_output_dir, asset_id)
            
            # Write script to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as script_file:
                script_file.write(script_content)
                script_path = script_file.name
            
            try:
                # Run Blender with script
                cmd = [
                    self.blender_executable,
                    "--background",
                    "--python", script_path
                ]
                
                logger.debug(f"Running Blender command: {' '.join(cmd)}")
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minutes timeout
                )
                
                if result.returncode != 0:
                    error_msg = f"Blender rendering failed: {result.stderr}"
                    logger.error(error_msg)
                    return RenderResult(
                        asset_id=asset_id,
                        asset_name=asset_name,
                        views={},
                        render_time=time.time() - start_time,
                        success=False,
                        error_message=error_msg
                    )
                
                # Collect rendered images
                rendered_views = {}
                for view_name in self.RENDER_VIEWS.keys():
                    image_path = asset_output_dir / f"{asset_id}_{view_name}.png"
                    if image_path.exists():
                        rendered_views[view_name] = image_path
                        logger.debug(f"Found rendered view: {view_name}")
                    else:
                        logger.warning(f"Missing rendered view: {view_name}")
                
                render_time = time.time() - start_time
                
                if len(rendered_views) == 6:
                    logger.info(f"Successfully rendered 6 views for {asset_name} in {render_time:.1f}s")
                    return RenderResult(
                        asset_id=asset_id,
                        asset_name=asset_name,
                        views=rendered_views,
                        render_time=render_time,
                        success=True,
                        metadata={
                            "resolution": self.resolution,
                            "samples": self.samples,
                            "views_count": len(rendered_views)
                        }
                    )
                else:
                    error_msg = f"Only {len(rendered_views)}/6 views rendered successfully"
                    logger.error(error_msg)
                    return RenderResult(
                        asset_id=asset_id,
                        asset_name=asset_name,
                        views=rendered_views,
                        render_time=render_time,
                        success=False,
                        error_message=error_msg
                    )
                    
            finally:
                # Clean up script file
                try:
                    Path(script_path).unlink()
                except:
                    pass
                    
        except Exception as e:
            error_msg = f"Error during rendering: {str(e)}"
            logger.error(error_msg)
            return RenderResult(
                asset_id=asset_id,
                asset_name=asset_name,
                views={},
                render_time=time.time() - start_time,
                success=False,
                error_message=error_msg
            )
    
    def get_view_priority_order(self) -> List[str]:
        """
        Get view names ordered by priority (highest first)
        
        Returns:
            List of view names in priority order
        """
        return sorted(self.RENDER_VIEWS.keys(), 
                     key=lambda x: self.RENDER_VIEWS[x].priority)


def main():
    """Test the Blender Renderer"""
    logging.basicConfig(level=logging.INFO)
    
    renderer = BlenderRenderer()
    
    print("🎨 Blender Renderer Test")
    print("=" * 50)
    
    # Verify Blender
    if not renderer.verify_blender():
        print("❌ Blender verification failed")
        return 1
    
    print("✅ Blender verification successful")
    print(f"📁 Output directory: {renderer.output_dir}")
    print(f"🎯 Configured for 6-view rendering at {renderer.resolution[0]}x{renderer.resolution[1]}")
    
    return 0

if __name__ == "__main__":
    exit(main())
