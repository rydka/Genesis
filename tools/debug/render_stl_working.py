#!/usr/bin/env python3
"""
Working STL renderer based on successful debug
"""

import subprocess
import tempfile
from pathlib import Path
import time

def render_stl_working(stl_path, output_dir, asset_name):
    """Render STL with 6 views using proven working approach"""
    
    print(f"🎨 Rendering: {stl_path.name}")
    
    # Ensure paths are absolute and properly formatted
    stl_path = Path(stl_path).absolute()
    output_dir = Path(output_dir).absolute()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"   STL: {stl_path}")
    print(f"   Output: {output_dir}")
    
    # Working Blender script based on successful debug
    script_content = f'''
import bpy
import os
import sys
import mathutils

def log_print(msg):
    print(f"[RENDER] {{msg}}")
    sys.stdout.flush()

try:
    log_print("=== STL RENDERING START ===")
    
    # Clear scene
    log_print("Clearing scene...")
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    
    # Import STL
    stl_file = r"{stl_path}"
    log_print(f"Importing STL: {{stl_file}}")
    
    if not os.path.exists(stl_file):
        log_print(f"ERROR: STL file not found: {{stl_file}}")
        sys.exit(1)
    
    # Import the STL
    bpy.ops.import_mesh.stl(filepath=stl_file)
    
    # Check if import was successful
    if not bpy.context.selected_objects:
        log_print("ERROR: No objects imported from STL")
        sys.exit(1)
    
    obj = bpy.context.selected_objects[0]
    bpy.context.view_layer.objects.active = obj
    log_print(f"STL imported successfully: {{obj.name}}")
    
    # Center and scale object
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    obj.location = (0, 0, 0)
    
    # Calculate bounding box for camera positioning
    bbox_corners = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
    dimensions = [
        max(bbox_corners, key=lambda x: x[0])[0] - min(bbox_corners, key=lambda x: x[0])[0],
        max(bbox_corners, key=lambda x: x[1])[1] - min(bbox_corners, key=lambda x: x[1])[1],
        max(bbox_corners, key=lambda x: x[2])[2] - min(bbox_corners, key=lambda x: x[2])[2]
    ]
    max_dim = max(dimensions)
    camera_distance = max_dim * 2.0
    log_print(f"Object dimensions: {{dimensions}}")
    log_print(f"Camera distance: {{camera_distance}}")
    
    # Setup render settings
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.render.resolution_x = 512
    scene.render.resolution_y = 512
    scene.cycles.samples = 32
    scene.render.image_settings.file_format = 'PNG'
    log_print("Render settings configured")
    
    # Add lighting
    bpy.ops.object.light_add(type='SUN', location=(camera_distance, camera_distance, camera_distance))
    sun = bpy.context.object
    sun.data.energy = 5
    log_print("Lighting added")
    
    # Add camera
    bpy.ops.object.camera_add()
    camera = bpy.context.object
    scene.camera = camera
    log_print("Camera added")
    
    # Add simple material to object
    if not obj.data.materials:
        mat = bpy.data.materials.new(name="STL_Material")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        bsdf.inputs[0].default_value = (0.7, 0.7, 0.7, 1.0)  # Gray color
        obj.data.materials.append(mat)
        log_print("Material added")
    
    # Define 6 camera positions
    views = {{
        "front": ((0, -camera_distance, 0), (1.5708, 0, 0)),
        "back": ((0, camera_distance, 0), (1.5708, 0, 3.14159)),
        "left": ((-camera_distance, 0, 0), (1.5708, 0, 1.5708)),
        "right": ((camera_distance, 0, 0), (1.5708, 0, -1.5708)),
        "top": ((0, 0, camera_distance), (0, 0, 0)),
        "iso_3d": ((camera_distance*0.7, -camera_distance*0.7, camera_distance*0.5), (1.0472, 0, 0.7854))
    }}
    
    output_dir = r"{output_dir}"
    rendered_count = 0
    
    for view_name, (location, rotation) in views.items():
        log_print(f"Rendering view: {{view_name}}")
        
        # Position camera
        camera.location = location
        camera.rotation_euler = rotation
        
        # Set output path
        output_file = os.path.join(output_dir, f"{asset_name}_{{view_name}}.png")
        scene.render.filepath = output_file
        
        try:
            # Render
            bpy.ops.render.render(write_still=True)
            
            # Verify file was created
            if os.path.exists(output_file):
                file_size = os.path.getsize(output_file)
                log_print(f"SUCCESS: {{view_name}} - {{file_size}} bytes")
                rendered_count += 1
            else:
                log_print(f"ERROR: {{view_name}} file not created")
                
        except Exception as e:
            log_print(f"ERROR rendering {{view_name}}: {{str(e)}}")
    
    log_print(f"=== RENDERING COMPLETE: {{rendered_count}}/6 views ===")
    
    if rendered_count == 6:
        log_print("SUCCESS: All views rendered successfully")
    else:
        log_print(f"WARNING: Only {{rendered_count}}/6 views rendered")
    
except Exception as e:
    log_print(f"FATAL ERROR: {{str(e)}}")
    import traceback
    log_print(f"TRACEBACK: {{traceback.format_exc()}}")
    sys.exit(1)
'''
    
    # Execute Blender script
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
        f.write(script_content)
        script_path = f.name
    
    try:
        blender_exe = r"C:\Program Files\Blender Foundation\Blender 4.5\blender.exe"
        cmd = [blender_exe, "--background", "--python", script_path]
        
        print(f"   Executing Blender...")
        start_time = time.time()
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300
        )
        
        render_time = time.time() - start_time
        
        # Parse results
        success_count = 0
        if result.stdout:
            for line in result.stdout.split('\n'):
                if '[RENDER] SUCCESS:' in line:
                    success_count += 1
        
        print(f"   Completed in {render_time:.1f}s")
        print(f"   Views rendered: {success_count}/6")
        
        # Verify output files
        rendered_files = list(output_dir.glob(f"{asset_name}_*.png"))
        
        if len(rendered_files) == 6:
            total_size = sum(f.stat().st_size for f in rendered_files) / 1024
            print(f"   ✅ SUCCESS: All 6 views ({total_size:.1f} KB total)")
            return True, rendered_files
        else:
            print(f"   ⚠️  Only {len(rendered_files)}/6 files created")
            return False, rendered_files
            
    except subprocess.TimeoutExpired:
        print(f"   ❌ Timeout after 300s")
        return False, []
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False, []
    finally:
        try:
            Path(script_path).unlink()
        except:
            pass

def main():
    """Test the working STL renderer"""
    
    print("🎨 Working STL Renderer Test")
    print("=" * 50)
    
    # Find first STL
    assets_dir = Path("ressources/data/assets")
    stl_files = list(assets_dir.glob("*.stl"))
    
    if not stl_files:
        print("❌ No STL files found")
        return 1
    
    # Test with smallest file first
    stl_files.sort(key=lambda x: x.stat().st_size)
    test_stl = stl_files[0]
    
    output_dir = Path("ressources/data/renders/working_test")
    asset_name = test_stl.stem
    
    print(f"Testing with: {test_stl.name} ({test_stl.stat().st_size/1024/1024:.1f} MB)")
    
    success, files = render_stl_working(test_stl, output_dir, asset_name)
    
    if success:
        print(f"\n🎉 STL rendering SUCCESS!")
        print(f"   Files created:")
        for f in files:
            size_kb = f.stat().st_size / 1024
            print(f"     📸 {f.name}: {size_kb:.1f} KB")
        
        print(f"\n🚀 Ready to render all {len(stl_files)} fantasy assets!")
        return 0
    else:
        print(f"\n❌ STL rendering failed")
        return 1

if __name__ == "__main__":
    exit(main())
