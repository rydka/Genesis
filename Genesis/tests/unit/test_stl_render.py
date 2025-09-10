#!/usr/bin/env python3
"""
Test STL rendering with working Blender setup
"""

import subprocess
import tempfile
from pathlib import Path
import time

def render_stl_simple(stl_path, output_dir, asset_id):
    """Render a single STL with 6 views using working Blender setup"""
    
    print(f"🎨 Rendering STL: {stl_path.name}")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Simple working Blender script
    script_content = f'''
import bpy
import bmesh
import mathutils
import os
from pathlib import Path

print("=== STL Rendering Script Started ===")

# Clear scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# Import STL
stl_path = r"{stl_path}"
print(f"Importing STL: {{stl_path}}")

try:
    # Use correct STL import method for Blender 4.5+
    bpy.ops.wm.stl_import(filepath=stl_path)
    print("STL imported successfully")
except Exception as e:
    print(f"ERROR importing STL: {{e}}")
    exit(1)

# Get imported object
if not bpy.context.selected_objects:
    print("ERROR: No objects selected after import")
    exit(1)

obj = bpy.context.selected_objects[0]
bpy.context.view_layer.objects.active = obj
print(f"Object imported: {{obj.name}}")

# Center object
bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
obj.location = (0, 0, 0)

# Calculate camera distance
bbox = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
max_dim = max(max(bbox, key=lambda x: x[i])[i] - min(bbox, key=lambda x: x[i])[i] for i in range(3))
camera_distance = max_dim * 1.5
print(f"Camera distance: {{camera_distance}}")

# Setup scene
scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.render.resolution_x = 512
scene.render.resolution_y = 512
scene.cycles.samples = 32

# Add lighting
bpy.ops.object.light_add(type='SUN', location=(5, 5, 10))
sun = bpy.context.object
sun.data.energy = 3

# Setup camera
bpy.ops.object.camera_add()
camera = bpy.context.object
scene.camera = camera

# Simple material
mat = bpy.data.materials.new(name="AssetMaterial")
mat.use_nodes = True
bsdf = mat.node_tree.nodes["Principled BSDF"]
bsdf.inputs[0].default_value = (0.8, 0.8, 0.8, 1.0)
obj.data.materials.append(mat)

# Define 6 views
views = {{
    "front": ((0, -camera_distance, 0), (1.5708, 0, 0)),
    "back": ((0, camera_distance, 0), (1.5708, 0, 3.14159)),
    "left": ((-camera_distance, 0, 0), (1.5708, 0, 1.5708)),
    "right": ((camera_distance, 0, 0), (1.5708, 0, -1.5708)),
    "top": ((0, 0, camera_distance), (0, 0, 0)),
    "iso_3d": ((camera_distance*0.75, -camera_distance*0.75, camera_distance*0.5), (1.0472, 0, 0.7854))
}}

output_dir = Path(r"{output_dir}")
rendered_count = 0

for view_name, (location, rotation) in views.items():
    print(f"Rendering view: {{view_name}}")
    
    # Position camera
    camera.location = location
    camera.rotation_euler = rotation
    
    # Render
    output_path = output_dir / f"{asset_id}_{{view_name}}.png"
    scene.render.filepath = str(output_path)
    
    try:
        bpy.ops.render.render(write_still=True)
        if os.path.exists(str(output_path)):
            file_size = os.path.getsize(str(output_path))
            print(f"SUCCESS: {{view_name}} rendered - {{file_size}} bytes")
            rendered_count += 1
        else:
            print(f"ERROR: {{view_name}} file not created")
    except Exception as e:
        print(f"ERROR rendering {{view_name}}: {{e}}")

print(f"=== Rendering Complete: {{rendered_count}}/6 views ===")
'''
    
    # Write script
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
        f.write(script_content)
        script_path = f.name
    
    try:
        # Run Blender
        blender_exe = r"C:\Program Files\Blender Foundation\Blender 4.5\blender.exe"
        cmd = [blender_exe, "--background", "--python", script_path]
        
        print(f"Running Blender...")
        start_time = time.time()
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300
        )
        
        render_time = time.time() - start_time
        
        print(f"Blender completed in {render_time:.1f}s")
        
        if result.returncode == 0:
            # Count rendered files
            rendered_files = list(output_dir.glob(f"{asset_id}_*.png"))
            
            print(f"\\n📊 Render Results:")
            print(f"   Views rendered: {len(rendered_files)}/6")
            print(f"   Render time: {render_time:.1f}s")
            
            total_size = 0
            for img_path in rendered_files:
                size_kb = img_path.stat().st_size / 1024
                total_size += size_kb
                print(f"   📸 {img_path.name}: {size_kb:.1f} KB")
            
            print(f"   Total size: {total_size:.1f} KB")
            
            if len(rendered_files) == 6:
                print(f"\\n✅ SUCCESS: All 6 views rendered!")
                return True
            else:
                print(f"\\n⚠️  WARNING: Only {len(rendered_files)}/6 views rendered")
                return False
        else:
            print(f"\\n❌ Blender failed:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"\\n❌ Error: {e}")
        return False
    finally:
        # Cleanup
        try:
            Path(script_path).unlink()
        except:
            pass

def main():
    """Test STL rendering"""
    
    print("🎨 STL Rendering Test")
    print("=" * 50)
    
    # Find first STL file
    assets_dir = Path("ressources/data/assets")
    stl_files = list(assets_dir.glob("*.stl"))
    
    if not stl_files:
        print("❌ No STL files found")
        return 1
    
    test_stl = stl_files[0]
    output_dir = Path("ressources/data/renders/test")
    asset_id = "test_render"
    
    print(f"Testing with: {test_stl.name}")
    print(f"Output: {output_dir}")
    
    success = render_stl_simple(test_stl, output_dir, asset_id)
    
    if success:
        print(f"\\n🎉 STL rendering test successful!")
        print(f"   Ready to render all {len(stl_files)} assets")
        return 0
    else:
        print(f"\\n❌ STL rendering test failed")
        return 1

if __name__ == "__main__":
    exit(main())
