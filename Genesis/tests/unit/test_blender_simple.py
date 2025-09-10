#!/usr/bin/env python3
"""
Simple Blender test to debug rendering issues
"""

import subprocess
import tempfile
from pathlib import Path

def test_blender_basic():
    """Test basic Blender functionality"""
    
    print("🔧 Testing Blender Basic Functionality")
    print("=" * 50)
    
    # Simple Blender script to test
    script_content = '''
import bpy
import os

print("=== Blender Python Script Started ===")
print(f"Blender version: {bpy.app.version_string}")
print(f"Working directory: {os.getcwd()}")

# Clear scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# Add a simple cube
bpy.ops.mesh.primitive_cube_add()
cube = bpy.context.object

# Setup camera
bpy.ops.object.camera_add(location=(4, -4, 3))
camera = bpy.context.object
bpy.context.scene.camera = camera

# Point camera at cube
import mathutils
direction = mathutils.Vector((0, 0, 0)) - camera.location
camera.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()

# Setup render
scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.render.resolution_x = 256
scene.render.resolution_y = 256
scene.cycles.samples = 16

# Render test
output_path = r"C:\\Users\\Shadow\\Desktop\\Work\\Genesis\\Genesis\\test_render.png"
scene.render.filepath = output_path
print(f"Rendering to: {output_path}")

try:
    bpy.ops.render.render(write_still=True)
    print("SUCCESS: Render completed successfully")

    # Check if file exists
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path)
        print(f"SUCCESS: Output file created: {file_size} bytes")
    else:
        print("ERROR: Output file not found")

except Exception as e:
    print(f"ERROR: Render failed: {e}")

print("=== Blender Python Script Completed ===")
'''
    
    # Write script to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
        f.write(script_content)
        script_path = f.name
    
    try:
        # Run Blender
        blender_exe = r"C:\Program Files\Blender Foundation\Blender 4.5\blender.exe"
        cmd = [blender_exe, "--background", "--python", script_path]
        
        print(f"Running: {' '.join(cmd)}")
        print("This may take 30-60 seconds...")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120
        )
        
        print(f"\n📊 Blender Exit Code: {result.returncode}")
        
        if result.stdout:
            print(f"\n📝 Blender Output:")
            print(result.stdout)
        
        if result.stderr:
            print(f"\n⚠️ Blender Errors:")
            print(result.stderr)
        
        # Check if test render was created
        test_file = Path("test_render.png")
        if test_file.exists():
            size_kb = test_file.stat().st_size / 1024
            print(f"\nSUCCESS: Test render successful: {size_kb:.1f} KB")
            test_file.unlink()  # Clean up
            return True
        else:
            print(f"\nERROR: Test render failed: No output file")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"\n⏰ Blender timed out after 120 seconds")
        return False
    except Exception as e:
        print(f"\n❌ Error running Blender: {e}")
        return False
    finally:
        # Clean up script
        try:
            Path(script_path).unlink()
        except:
            pass

if __name__ == "__main__":
    success = test_blender_basic()
    exit(0 if success else 1)
