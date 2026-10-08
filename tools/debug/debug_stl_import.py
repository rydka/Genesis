#!/usr/bin/env python3
"""
Debug STL import and rendering specifically
"""

import subprocess
import tempfile
from pathlib import Path
import time

def debug_stl_import():
    """Debug STL import process step by step"""
    
    print("🔍 STL Import Debug")
    print("=" * 50)
    
    # Find STL file
    stl_dir = Path("ressources/data/assets")
    stl_files = list(stl_dir.glob("*.stl"))
    
    if not stl_files:
        print("❌ No STL files found")
        return False
    
    test_stl = stl_files[0]
    output_dir = Path("ressources/data/renders/debug")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Testing STL: {test_stl.name}")
    print(f"Size: {test_stl.stat().st_size / 1024 / 1024:.1f} MB")
    
    # Step-by-step STL debug script
    script_content = f'''
import bpy
import bmesh
import mathutils
import os
import sys
import traceback

def log_print(msg):
    print(f"[STL DEBUG] {{msg}}")
    sys.stdout.flush()

try:
    log_print("=== STL IMPORT DEBUG START ===")
    
    # Clear scene
    log_print("Step 1: Clearing scene...")
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    log_print("Scene cleared successfully")
    
    # Check STL file
    stl_path = r"{test_stl.absolute()}"
    log_print(f"Step 2: Checking STL file...")
    log_print(f"STL path: {{stl_path}}")
    log_print(f"File exists: {{os.path.exists(stl_path)}}")
    log_print(f"File size: {{os.path.getsize(stl_path) / 1024 / 1024:.1f}} MB")
    
    if not os.path.exists(stl_path):
        log_print("ERROR: STL file not found!")
        sys.exit(1)
    
    # Import STL
    log_print("Step 3: Importing STL...")
    try:
        bpy.ops.import_mesh.stl(filepath=stl_path)
        log_print("STL import operation completed")
    except Exception as e:
        log_print(f"ERROR during STL import: {{e}}")
        log_print(f"Traceback: {{traceback.format_exc()}}")
        sys.exit(1)
    
    # Check imported objects
    log_print("Step 4: Checking imported objects...")
    selected_objects = bpy.context.selected_objects
    log_print(f"Selected objects count: {{len(selected_objects)}}")
    
    if not selected_objects:
        log_print("ERROR: No objects selected after import")
        all_objects = list(bpy.data.objects)
        log_print(f"All objects in scene: {{[obj.name for obj in all_objects]}}")
        sys.exit(1)
    
    obj = selected_objects[0]
    bpy.context.view_layer.objects.active = obj
    log_print(f"Active object: {{obj.name}}")
    log_print(f"Object type: {{obj.type}}")
    
    if obj.type == 'MESH':
        mesh = obj.data
        log_print(f"Mesh vertices: {{len(mesh.vertices)}}")
        log_print(f"Mesh faces: {{len(mesh.polygons)}}")
        log_print(f"Mesh edges: {{len(mesh.edges)}}")
    else:
        log_print(f"WARNING: Object is not a mesh, type is {{obj.type}}")
    
    # Center object
    log_print("Step 5: Centering object...")
    try:
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
        obj.location = (0, 0, 0)
        log_print("Object centered successfully")
    except Exception as e:
        log_print(f"ERROR centering object: {{e}}")
    
    # Calculate bounding box
    log_print("Step 6: Calculating bounding box...")
    try:
        bbox = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
        dimensions = []
        for i in range(3):
            min_coord = min(bbox, key=lambda x: x[i])[i]
            max_coord = max(bbox, key=lambda x: x[i])[i]
            dimensions.append(max_coord - min_coord)
        
        max_dim = max(dimensions)
        camera_distance = max_dim * 1.5
        
        log_print(f"Bounding box dimensions: {{dimensions}}")
        log_print(f"Max dimension: {{max_dim}}")
        log_print(f"Camera distance: {{camera_distance}}")
    except Exception as e:
        log_print(f"ERROR calculating bounding box: {{e}}")
        camera_distance = 10  # fallback
        log_print(f"Using fallback camera distance: {{camera_distance}}")
    
    # Setup scene for rendering
    log_print("Step 7: Setting up scene...")
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.render.resolution_x = 256
    scene.render.resolution_y = 256
    scene.cycles.samples = 16
    log_print("Scene setup completed")
    
    # Add lighting
    log_print("Step 8: Adding lighting...")
    bpy.ops.object.light_add(type='SUN', location=(5, 5, 10))
    sun = bpy.context.object
    sun.data.energy = 3
    log_print("Lighting added")
    
    # Setup camera
    log_print("Step 9: Setting up camera...")
    bpy.ops.object.camera_add()
    camera = bpy.context.object
    scene.camera = camera
    log_print("Camera added")
    
    # Add material
    log_print("Step 10: Adding material...")
    if not obj.data.materials:
        mat = bpy.data.materials.new(name="STLMaterial")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        bsdf.inputs[0].default_value = (0.8, 0.8, 0.8, 1.0)
        obj.data.materials.append(mat)
        log_print("Material added successfully")
    else:
        log_print("Object already has materials")
    
    # Test single render
    log_print("Step 11: Testing single render...")
    output_dir = r"{output_dir.absolute()}"
    camera.location = (0, -camera_distance, 0)
    camera.rotation_euler = (1.5708, 0, 0)
    
    output_path = os.path.join(output_dir, "stl_debug_render.png")
    scene.render.filepath = output_path
    
    log_print(f"Rendering to: {{output_path}}")
    
    try:
        bpy.ops.render.render(write_still=True)
        log_print("Render operation completed")
        
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            log_print(f"SUCCESS: Render file created - {{file_size}} bytes")
        else:
            log_print("ERROR: Render file not created")
            log_print(f"Directory contents: {{os.listdir(output_dir)}}")
            
    except Exception as e:
        log_print(f"ERROR during render: {{e}}")
        log_print(f"Traceback: {{traceback.format_exc()}}")
    
    log_print("=== STL IMPORT DEBUG END ===")
    
except Exception as e:
    log_print(f"FATAL ERROR: {{str(e)}}")
    log_print(f"TRACEBACK: {{traceback.format_exc()}}")
    sys.exit(1)
'''
    
    # Execute debug script
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
        f.write(script_content)
        script_path = f.name
    
    try:
        blender_exe = r"C:\Program Files\Blender Foundation\Blender 4.5\blender.exe"
        cmd = [blender_exe, "--background", "--python", script_path]
        
        print(f"Executing STL debug script...")
        
        start_time = time.time()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minutes for large STL
        )
        duration = time.time() - start_time
        
        print(f"\n⏱️ Execution time: {duration:.1f}s")
        print(f"📊 Exit code: {result.returncode}")
        
        # Parse output
        print(f"\n📝 STL Debug Logs:")
        if result.stdout:
            lines = result.stdout.split('\n')
            debug_logs = [line for line in lines if '[STL DEBUG]' in line]
            
            if debug_logs:
                for log in debug_logs:
                    print(f"   {log}")
            else:
                print("   No debug logs found")
                print("   Raw output (first 20 lines):")
                for line in lines[:20]:
                    if line.strip():
                        print(f"     {line}")
        
        if result.stderr:
            print(f"\n⚠️ Errors:")
            for line in result.stderr.split('\n')[:10]:
                if line.strip():
                    print(f"   {line}")
        
        # Check output
        debug_file = output_dir / "stl_debug_render.png"
        print(f"\n📸 Render Result:")
        print(f"   File: {debug_file}")
        print(f"   Exists: {debug_file.exists()}")
        
        if debug_file.exists():
            size_kb = debug_file.stat().st_size / 1024
            print(f"   Size: {size_kb:.1f} KB")
            print(f"✅ STL RENDER SUCCESSFUL!")
            return True
        else:
            print(f"❌ STL RENDER FAILED")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"\n⏰ STL debug timed out after 5 minutes")
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
    """Run STL import debug"""
    
    success = debug_stl_import()
    
    print(f"\n" + "=" * 50)
    if success:
        print("🎉 STL IMPORT DEBUG SUCCESSFUL!")
        print("   STL files can be imported and rendered")
    else:
        print("❌ STL IMPORT DEBUG FAILED")
        print("   Found issues with STL import/render process")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
