"""Render GLB meshes from 4 angles with Blender (Workbench, orthographic), for visual comparison.

Usage:
    blender -b -P tools/debug/render_glb_views.py -- OUT_DIR a.glb b.glb ...

Writes OUT_DIR/<stem>__{front,three_quarter,side,back}.png (512x512).
"""
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector

args = sys.argv[sys.argv.index("--") + 1:]
out_dir = Path(args[0]).resolve()  # Blender resolves relative render paths against its own cwd
out_dir.mkdir(parents=True, exist_ok=True)
glbs = [Path(p).resolve() for p in args[1:]]

VIEWS = {"front": 0, "three_quarter": 45, "side": 90, "back": 180}

for glb in glbs:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(glb))
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    pts = [o.matrix_world @ Vector(c) for o in meshes for c in o.bound_box]
    lo = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    hi = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    center, size = (lo + hi) / 2, max(hi - lo)

    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "STUDIO"
    scene.display.shading.color_type = "SINGLE"
    scene.display.shading.single_color = (0.75, 0.72, 0.8)
    scene.display.shading.show_cavity = True
    scene.render.resolution_x = scene.render.resolution_y = 512
    scene.render.film_transparent = False
    scene.world = bpy.data.worlds.new("w")

    cam_data = bpy.data.cameras.new("cam")
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = size * 1.15
    cam = bpy.data.objects.new("cam", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam

    # glTF is Y-up, imported as Z-up; Hunyuan front faces -Y after import
    for name, deg in VIEWS.items():
        a = math.radians(deg)
        d = size * 3
        cam.location = center + Vector((d * math.sin(a), -d * math.cos(a), 0))
        direction = center - cam.location
        cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
        scene.render.filepath = str(out_dir / f"{glb.stem}__{name}.png")
        bpy.ops.render.render(write_still=True)
