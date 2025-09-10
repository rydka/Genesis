#!/usr/bin/env python3
"""
Test correct STL import method for Blender 4.5
"""

import subprocess
import tempfile
from pathlib import Path

def test_stl_import_methods():
    """Test different STL import methods"""
    
    print("🔍 Testing STL Import Methods for Blender 4.5")
    print("=" * 60)
    
    # Find STL file
    stl_dir = Path("ressources/data/assets")
    stl_files = list(stl_dir.glob("*.stl"))
    
    if not stl_files:
        print("❌ No STL files found")
        return False
    
    test_stl = stl_files[0]
    output_dir = Path("ressources/data/renders/debug")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Testing with: {test_stl.name}")
    
    # Test script with multiple import methods
    script_content = f'''
import bpy
import os
import sys

def log_print(msg):
    print(f"[IMPORT TEST] {{msg}}")
    sys.stdout.flush()

try:
    log_print("=== STL IMPORT METHOD TEST ===")
    log_print(f"Blender version: {{bpy.app.version_string}}")
    
    # Clear scene
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    log_print("Scene cleared")
    
    stl_path = r"{test_stl.absolute()}"
    log_print(f"STL file: {{stl_path}}")
    log_print(f"File exists: {{os.path.exists(stl_path)}}")
    
    # Method 1: Try old import_mesh.stl
    log_print("\\nMethod 1: Testing bpy.ops.import_mesh.stl...")
    try:
        bpy.ops.import_mesh.stl(filepath=stl_path)
        log_print("SUCCESS: import_mesh.stl worked")
        imported_objects = bpy.context.selected_objects
        log_print(f"Imported objects: {{len(imported_objects)}}")
        if imported_objects:
            log_print(f"Object name: {{imported_objects[0].name}}")
            log_print("Method 1 SUCCESSFUL")
        else:
            log_print("Method 1 FAILED: No objects imported")
    except Exception as e:
        log_print(f"Method 1 FAILED: {{e}}")
    
    # Clear scene for next test
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    
    # Method 2: Try wm.stl_import (new method)
    log_print("\\nMethod 2: Testing bpy.ops.wm.stl_import...")
    try:
        bpy.ops.wm.stl_import(filepath=stl_path)
        log_print("SUCCESS: wm.stl_import worked")
        imported_objects = bpy.context.selected_objects
        log_print(f"Imported objects: {{len(imported_objects)}}")
        if imported_objects:
            log_print(f"Object name: {{imported_objects[0].name}}")
            log_print("Method 2 SUCCESSFUL")
        else:
            log_print("Method 2 FAILED: No objects imported")
    except Exception as e:
        log_print(f"Method 2 FAILED: {{e}}")
    
    # Clear scene for next test
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    
    # Method 3: Try import_scene.stl
    log_print("\\nMethod 3: Testing bpy.ops.import_scene.stl...")
    try:
        bpy.ops.import_scene.stl(filepath=stl_path)
        log_print("SUCCESS: import_scene.stl worked")
        imported_objects = bpy.context.selected_objects
        log_print(f"Imported objects: {{len(imported_objects)}}")
        if imported_objects:
            log_print(f"Object name: {{imported_objects[0].name}}")
            log_print("Method 3 SUCCESSFUL")
        else:
            log_print("Method 3 FAILED: No objects imported")
    except Exception as e:
        log_print(f"Method 3 FAILED: {{e}}")
    
    # List all available import operators
    log_print("\\nAvailable import operators:")
    import_ops = [op for op in dir(bpy.ops) if 'import' in op.lower()]
    for op_name in import_ops[:10]:  # Show first 10
        log_print(f"  {{op_name}}")
    
    # Check if STL addon is enabled
    log_print("\\nChecking STL addon status...")
    addon_utils = bpy.utils
    if hasattr(addon_utils, 'addon_utils'):
        addon_utils = addon_utils.addon_utils
    
    # Try to enable STL addon
    log_print("Attempting to enable STL addon...")
    try:
        bpy.ops.preferences.addon_enable(module="io_mesh_stl")
        log_print("STL addon enabled")
    except Exception as e:
        log_print(f"Could not enable STL addon: {{e}}")
    
    # List enabled addons
    log_print("\\nEnabled addons (first 10):")
    try:
        enabled_addons = bpy.context.preferences.addons.keys()
        for addon in list(enabled_addons)[:10]:
            log_print(f"  {{addon}}")
    except Exception as e:
        log_print(f"Could not list addons: {{e}}")
    
    log_print("=== IMPORT METHOD TEST COMPLETE ===")
    
except Exception as e:
    log_print(f"FATAL ERROR: {{str(e)}}")
    import traceback
    log_print(f"TRACEBACK: {{traceback.format_exc()}}")
'''
    
    # Execute test script
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
        f.write(script_content)
        script_path = f.name
    
    try:
        blender_exe = r"C:\Program Files\Blender Foundation\Blender 4.5\blender.exe"
        cmd = [blender_exe, "--background", "--python", script_path]
        
        print(f"Executing import method test...")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120
        )
        
        print(f"\n📊 Exit code: {result.returncode}")
        
        # Parse output
        print(f"\n📝 Import Test Results:")
        if result.stdout:
            lines = result.stdout.split('\n')
            test_logs = [line for line in lines if '[IMPORT TEST]' in line]
            
            if test_logs:
                for log in test_logs:
                    print(f"   {log}")
            else:
                print("   No test logs found")
                print("   Raw output (first 20 lines):")
                for line in lines[:20]:
                    if line.strip():
                        print(f"     {line}")
        
        if result.stderr:
            print(f"\n⚠️ Errors:")
            for line in result.stderr.split('\n')[:5]:
                if line.strip():
                    print(f"   {line}")
        
        return result.returncode == 0
            
    except Exception as e:
        print(f"\n❌ Execution error: {e}")
        return False
    finally:
        try:
            Path(script_path).unlink()
        except:
            pass

def main():
    """Run STL import method test"""
    
    success = test_stl_import_methods()
    
    print(f"\n" + "=" * 60)
    if success:
        print("✅ STL IMPORT METHOD TEST COMPLETED")
        print("   Check the logs above to see which method works")
    else:
        print("❌ STL IMPORT METHOD TEST FAILED")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
