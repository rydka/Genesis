#!/usr/bin/env python3
"""
Debug script for Blender rendering issues
"""

import subprocess
import tempfile
from pathlib import Path
import time
import os

def debug_blender_detailed():
    """Detailed debug of Blender rendering"""
    
    print("🔍 Blender Debug Analysis")
    print("=" * 60)
    
    # Test 1: Basic Blender info
    print("\n📋 Test 1: Blender Information")
    print("-" * 30)
    
    blender_exe = r"C:\Program Files\Blender Foundation\Blender 4.5\blender.exe"
    
    try:
        result = subprocess.run([blender_exe, "--version"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ Blender accessible")
            version_line = result.stdout.split('\n')[0]
            print(f"   Version: {version_line}")
        else:
            print("❌ Blender version check failed")
            return False
    except Exception as e:
        print(f"❌ Error accessing Blender: {e}")
        return False
    
    # Test 2: Directory permissions
    print("\n📁 Test 2: Directory Permissions")
    print("-" * 30)
    
    test_dirs = [
        Path("ressources/data/renders"),
        Path("ressources/data/renders/debug"),
        Path.cwd()
    ]
    
    for test_dir in test_dirs:
        try:
            test_dir.mkdir(parents=True, exist_ok=True)
            test_file = test_dir / "test_write.txt"
            test_file.write_text("test")
            test_file.unlink()
            print(f"✅ {test_dir}: Write permissions OK")
        except Exception as e:
            print(f"❌ {test_dir}: Write error - {e}")
    
    # Test 3: STL file access
    print("\n📄 Test 3: STL File Access")
    print("-" * 30)
    
    stl_dir = Path("ressources/data/assets")
    stl_files = list(stl_dir.glob("*.stl"))
    
    if stl_files:
        test_stl = stl_files[0]
        print(f"✅ Found {len(stl_files)} STL files")
        print(f"   Test file: {test_stl.name}")
        print(f"   Size: {test_stl.stat().st_size / 1024 / 1024:.1f} MB")
        print(f"   Full path: {test_stl.absolute()}")
    else:
        print("❌ No STL files found")
        return False
    
    # Test 4: Minimal render test with full logging
    print("\n🎨 Test 4: Minimal Render with Full Logging")
    print("-" * 30)
    
    output_dir = Path("ressources/data/renders/debug")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Ultra-simple Blender script with extensive logging
    script_content = f'''
import bpy
import os
import sys
import traceback

def log_print(msg):
    print(f"[BLENDER LOG] {{msg}}")
    sys.stdout.flush()

try:
    log_print("=== BLENDER DEBUG SCRIPT START ===")
    log_print(f"Python version: {{sys.version}}")
    log_print(f"Blender version: {{bpy.app.version_string}}")
    log_print(f"Working directory: {{os.getcwd()}}")
    
    # Check output directory
    output_dir = r"{output_dir.absolute()}"
    log_print(f"Output directory: {{output_dir}}")
    log_print(f"Output dir exists: {{os.path.exists(output_dir)}}")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        log_print("Created output directory")
    
    # Clear scene
    log_print("Clearing scene...")
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    log_print("Scene cleared")
    
    # Add simple cube
    log_print("Adding cube...")
    bpy.ops.mesh.primitive_cube_add()
    cube = bpy.context.object
    log_print(f"Cube added: {{cube.name}}")
    
    # Setup camera
    log_print("Setting up camera...")
    bpy.ops.object.camera_add(location=(4, -4, 3))
    camera = bpy.context.object
    bpy.context.scene.camera = camera
    log_print(f"Camera added: {{camera.name}}")
    
    # Setup render settings
    log_print("Configuring render settings...")
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.render.resolution_x = 256
    scene.render.resolution_y = 256
    scene.cycles.samples = 8  # Very low for speed
    log_print("Render settings configured")
    
    # Test render
    output_file = os.path.join(output_dir, "debug_render.png")
    log_print(f"Rendering to: {{output_file}}")
    
    scene.render.filepath = output_file
    
    log_print("Starting render...")
    bpy.ops.render.render(write_still=True)
    log_print("Render operation completed")
    
    # Check result
    if os.path.exists(output_file):
        file_size = os.path.getsize(output_file)
        log_print(f"SUCCESS: File created - {{file_size}} bytes")
    else:
        log_print("ERROR: Output file not found")
        log_print(f"Directory contents: {{os.listdir(output_dir)}}")
    
    log_print("=== BLENDER DEBUG SCRIPT END ===")
    
except Exception as e:
    log_print(f"EXCEPTION: {{str(e)}}")
    log_print(f"TRACEBACK: {{traceback.format_exc()}}")
'''
    
    # Write and execute script
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
        f.write(script_content)
        script_path = f.name
    
    try:
        print(f"Executing Blender debug script...")
        print(f"Script: {script_path}")
        
        cmd = [blender_exe, "--background", "--python", script_path]
        
        start_time = time.time()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120
        )
        duration = time.time() - start_time
        
        print(f"\n⏱️ Execution time: {duration:.1f}s")
        print(f"📊 Exit code: {result.returncode}")
        
        # Analyze output
        print(f"\n📝 STDOUT Analysis:")
        if result.stdout:
            lines = result.stdout.split('\n')
            blender_logs = [line for line in lines if '[BLENDER LOG]' in line]
            
            if blender_logs:
                print("   Blender script logs:")
                for log in blender_logs:
                    print(f"     {log}")
            else:
                print("   No Blender script logs found")
                print("   Raw stdout (first 10 lines):")
                for line in lines[:10]:
                    if line.strip():
                        print(f"     {line}")
        else:
            print("   No stdout")
        
        print(f"\n⚠️ STDERR Analysis:")
        if result.stderr:
            print("   Errors/warnings:")
            for line in result.stderr.split('\n')[:10]:
                if line.strip():
                    print(f"     {line}")
        else:
            print("   No stderr")
        
        # Check output file
        debug_file = output_dir / "debug_render.png"
        print(f"\n📸 Output File Check:")
        print(f"   Expected: {debug_file}")
        print(f"   Exists: {debug_file.exists()}")
        
        if debug_file.exists():
            size_kb = debug_file.stat().st_size / 1024
            print(f"   Size: {size_kb:.1f} KB")
            print(f"✅ DEBUG RENDER SUCCESSFUL!")
            return True
        else:
            print(f"❌ DEBUG RENDER FAILED - No output file")
            
            # List directory contents
            print(f"\n📁 Directory contents:")
            try:
                for item in output_dir.iterdir():
                    print(f"     {item.name}")
            except:
                print("     (Could not list directory)")
            
            return False
            
    except subprocess.TimeoutExpired:
        print(f"\n⏰ Blender timed out after 120s")
        return False
    except Exception as e:
        print(f"\n❌ Execution error: {e}")
        return False
    finally:
        # Cleanup
        try:
            Path(script_path).unlink()
        except:
            pass

def main():
    """Run comprehensive Blender debug"""
    
    success = debug_blender_detailed()
    
    print(f"\n" + "=" * 60)
    if success:
        print("🎉 DEBUG SUCCESSFUL - Blender rendering works!")
        print("   The issue may be in our complex STL rendering script")
    else:
        print("❌ DEBUG FAILED - Found fundamental Blender issues")
        print("   Need to resolve basic rendering problems first")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
