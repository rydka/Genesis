"""Render GLB meshes from 4 angles with Blender (Workbench, orthographic), for visual comparison.

Usage:
    blender -b -P tools/debug/render_glb_views.py -- OUT_DIR a.glb b.glb ...
        [--res 512] [--views front,three_quarter,side,back] [--frame REF.glb]

Writes OUT_DIR/<stem>__{front,three_quarter,side,back}.png (512x512 by default). Each mesh is framed
on its own bounding box, unless --frame gives a reference mesh whose box frames them all: crops at
fixed image coordinates then show the same region of every mesh (zoom sheets of tools/bench/).
"""
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector

ALL_VIEWS = {"front": 0, "three_quarter": 45, "side": 90, "back": 180}

args = sys.argv[sys.argv.index("--") + 1:]
res, views, frame = 512, list(ALL_VIEWS), None
positional = []
i = 0
while i < len(args):
    if args[i] == "--res":
        res, i = int(args[i + 1]), i + 2
    elif args[i] == "--views":
        views, i = args[i + 1].split(","), i + 2
    elif args[i] == "--frame":
        frame, i = Path(args[i + 1]).resolve(), i + 2
    else:
        positional.append(args[i])
        i += 1
out_dir = Path(positional[0]).resolve()  # Blender resolves relative render paths against its own cwd
out_dir.mkdir(parents=True, exist_ok=True)
glbs = [Path(p).resolve() for p in positional[1:]]


def load(glb: Path):
    """Reset the scene, import `glb`, return (centre, size) of its bounding box."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(glb))
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    pts = [o.matrix_world @ Vector(c) for o in meshes for c in o.bound_box]
    lo = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    hi = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    return (lo + hi) / 2, max(hi - lo)


framing = load(frame) if frame is not None else None

for glb in glbs:
    center, size = load(glb)
    if framing is not None:
        center, size = framing

    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "STUDIO"
    scene.display.shading.color_type = "SINGLE"
    scene.display.shading.single_color = (0.75, 0.72, 0.8)
    scene.display.shading.show_cavity = True
    scene.render.resolution_x = scene.render.resolution_y = res
    scene.render.film_transparent = False
    scene.world = bpy.data.worlds.new("w")

    cam_data = bpy.data.cameras.new("cam")
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = size * 1.15
    cam = bpy.data.objects.new("cam", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam

    # glTF is Y-up, imported as Z-up; Hunyuan front faces -Y after import
    for name in views:
        a = math.radians(ALL_VIEWS[name])
        d = size * 3
        cam.location = center + Vector((d * math.sin(a), -d * math.cos(a), 0))
        direction = center - cam.location
        cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
        scene.render.filepath = str(out_dir / f"{glb.stem}__{name}.png")
        bpy.ops.render.render(write_still=True)
