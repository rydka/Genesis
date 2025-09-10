#!/usr/bin/env python3
"""
Test STL rendering with corrected import method
"""

import subprocess
import tempfile
from pathlib import Path
import time

def test_stl_render_fixed():
    """Test STL rendering with correct import method"""
    
    print("🎨 Testing STL Render with Fixed Import")
    print("=" * 50)
    
    # Find STL file
    stl_dir = Path("ressources/data/assets")
    stl_files = list(stl_dir.glob("*.stl"))
    
    if not stl_files:
        print("❌ No STL files found")
        return False
    
    test_stl = stl_files[0]
    output_dir = Path("ressources/data/renders/test_fixed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Testing with: {test_stl.name}")
    print(f"Size: {test_stl.stat().st_size / 1024 / 1024:.1f} MB")
    
    # Fixed STL rendering script
    script_content = f'''
import bpy
import bmesh
import mathutils
import os
import sys

def log_print(msg):
    print(f"[STL RENDER] {{msg}}")
    sys.stdout.flush()

try:
    log_print("=== STL RENDER TEST START ===")
    
    # Clear scene
    log_print("Clearing scene...")
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    
    # Import STL with correct method
    stl_path = r"{test_stl.absolute()}"
    log_print(f"Importing STL: {{os.path.basename(stl_path)}}")
    
    bpy.ops.wm.stl_import(filepath=stl_path)
    log_print("STL imported successfully")
    
    # Get imported object
    imported_objects = bpy.context.selected_objects
    if not imported_objects:
        log_print("ERROR: No objects imported")
        sys.exit(1)
    
    obj = imported_objects[0]
    bpy.context.view_layer.objects.active = obj
    log_print(f"Object: {{obj.name}} ({{len(obj.data.vertices)}} vertices)")
    
    # Center object
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    obj.location = (0, 0, 0)
    
    # Calculate camera distance
    bbox = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
    max_dim = max(max(bbox, key=lambda x: x[i])[i] - min(bbox, key=lambda x: x[i])[i] for i in range(3))
    camera_distance = max_dim * 1.5
    log_print(f"Camera distance: {{camera_distance:.2f}}")
    
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
    
    # Add material
    if not obj.data.materials:
        mat = bpy.data.materials.new(name="STLMaterial")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        bsdf.inputs[0].default_value = (0.8, 0.8, 0.8, 1.0)
        obj.data.materials.append(mat)
    
    # Test 3 views
    views = {{
        "front": ((0, -camera_distance, 0), (1.5708, 0, 0)),
        "iso": ((camera_distance*0.7, -camera_distance*0.7, camera_distance*0.5), (1.0472, 0, 0.7854)),
        "top": ((0, 0, camera_distance), (0, 0, 0))
    }}
    
    output_dir = r"{output_dir.absolute()}"
    rendered_count = 0
    
    for view_name, (location, rotation) in views.items():
        log_print(f"Rendering {{view_name}} view...")
        
        # Position camera
        camera.location = location
        camera.rotation_euler = rotation
        
        # Render
        output_path = os.path.join(output_dir, f"stl_{{view_name}}.png")
        scene.render.filepath = output_path
        
        try:
            bpy.ops.render.render(write_still=True)
            
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                log_print(f"SUCCESS: {{view_name}} - {{file_size}} bytes")
                rendered_count += 1
            else:
                log_print(f"ERROR: {{view_name}} file not created")
                
        except Exception as e:
            log_print(f"ERROR rendering {{view_name}}: {{e}}")
    
    log_print(f"=== RENDER COMPLETE: {{rendered_count}}/3 views ===")
    
    if rendered_count == 3:
        log_print("SUCCESS: All views rendered!")
    else:
        log_print(f"PARTIAL: Only {{rendered_count}}/3 views rendered")
    
except Exception as e:
    log_print(f"FATAL ERROR: {{str(e)}}")
    import traceback
    log_print(f"TRACEBACK: {{traceback.format_exc()}}")
    sys.exit(1)
'''
    
    # Execute script
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
        f.write(script_content)
        script_path = f.name
    
    try:
        blender_exe = r"C:\Program Files\Blender Foundation\Blender 4.5\blender.exe"
        cmd = [blender_exe, "--background", "--python", script_path]
        
        print(f"Executing STL render test...")
        
        start_time = time.time()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=180  # 3 minutes
        )
        duration = time.time() - start_time
        
        print(f"\n⏱️ Execution time: {duration:.1f}s")
        print(f"📊 Exit code: {result.returncode}")
        
        # Parse output
        print(f"\n📝 Render Logs:")
        if result.stdout:
            lines = result.stdout.split('\n')
            render_logs = [line for line in lines if '[STL RENDER]' in line]
            
            if render_logs:
                for log in render_logs:
                    print(f"   {log}")
            else:
                print("   No render logs found")
        
        if result.stderr:
            print(f"\n⚠️ Errors:")
            for line in result.stderr.split('\n')[:5]:
                if line.strip():
                    print(f"   {line}")
        
        # Check output files
        rendered_files = list(output_dir.glob("stl_*.png"))
        print(f"\n📸 Render Results:")
        print(f"   Files rendered: {len(rendered_files)}/3")
        
        total_size = 0
        for img_path in rendered_files:
            size_kb = img_path.stat().st_size / 1024
            total_size += size_kb
            print(f"   📸 {img_path.name}: {size_kb:.1f} KB")
        
        if rendered_files:
            print(f"   Total size: {total_size:.1f} KB")
        
        success = len(rendered_files) == 3 and result.returncode == 0
        
        if success:
            print(f"\n✅ STL RENDER TEST SUCCESSFUL!")
        else:
            print(f"\n❌ STL RENDER TEST FAILED")
        
        return success
            
    except subprocess.TimeoutExpired:
        print(f"\n⏰ STL render test timed out after 3 minutes")
        return False
    except Exception as e:
        print(f"\n❌ Execution error: {e}")
        return False
    finally:
        try:
            Path(script_path).unlink()
        except:
            pass

def main():
    """Run STL render test"""
    
    success = test_stl_render_fixed()
    
    print(f"\n" + "=" * 50)
    if success:
        print("🎉 STL RENDERING FIXED!")
        print("   Ready to test full 6-view rendering")
    else:
        print("❌ STL RENDERING STILL HAS ISSUES")
        print("   Need further debugging")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
