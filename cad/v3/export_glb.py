"""
Headless Blender export of the v3 assembly as the GLB the software reads.

Usage:
    blender -b <master.blend> -P export_glb.py -- <out.glb>

    master.blend  the v3 assembly scene built by the Blender project in the
                  private hardware tree (`build/master.blend`, not part of
                  this repo): one mesh object per Fusion occurrence, named
                  '<group> / <Part>:<n>', under the rig empties below.
    out.glb       where the GLB goes; a `<out>.json` report is written next
                  to it.

The scene carries the rig the software needs:

    ORB_ROOT                 parent of everything
    ORB_ROTOR_base           turntable: spindle, pulley, platform insert;
                             turns about its local Z, origin on the axis
    ORB_ROTOR_elevation      shaft, pulley, bar insert, orbit arm; turns
                             about its local X, origin on the axis
    ORB_ROTOR_*_balls/motor  ball sets and motor pulleys (static here)

The server (`server/orbiter_server/geom/rig_model.py`) takes the pivots
from the origins of the two rotor nodes and the viewer spins their subtrees,
so the node names are kept exactly as they are in the scene.

What the export changes:
  - ORB_ROOT is shifted in X and Y so that the azimuth axis is the world Z
    axis; the machine still stands on z = 0. The server's math frame and
    the viewer's grid are centred on the turntable axis.
  - The hidden tools (extractors, wrench) are left out.
  - Meshes over 4 000 faces are decimated (bearing rings, pulleys) to keep
    the file small; placements and silhouettes stay (checked: bounds move
    less than 0.5 mm).
  - glTF 2.0 binary, metres, Y up (the Blender exporter's default), no
    compression: trimesh on the server reads it as is.
"""

import json
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

argv = sys.argv
user_args = argv[argv.index("--") + 1:] if "--" in argv else []
if len(user_args) != 1:
    raise SystemExit("usage: blender -b master.blend -P export_glb.py -- <out.glb>")
destination = Path(user_args[0]).resolve()
destination.parent.mkdir(parents=True, exist_ok=True)

ROTORS = ("ORB_ROTOR_base", "ORB_ROTOR_elevation")
MAX_FACES = 4000

scene = bpy.context.scene
scene.frame_set(1)
root = bpy.data.objects["ORB_ROOT"]
assert root.parent is None and root.matrix_world.is_identity, "ORB_ROOT must sit at the origin"
for name in ROTORS:
    rotor = bpy.data.objects[name]
    assert rotor.parent is root, f"{name} must be a child of ORB_ROOT"
    assert rotor.rotation_euler.to_matrix().is_identity, f"{name} must be at rest"

parts = [o for o in scene.objects
         if o.type == "MESH" and not o.hide_render and not o.get("orb_tool", False)]


def bounds(objects):
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for obj in objects:
        co = np.empty(len(obj.data.vertices) * 3)
        obj.data.vertices.foreach_get("co", co)
        m = np.array(obj.matrix_world)
        pts = co.reshape(-1, 3) @ m[:3, :3].T + m[:3, 3]
        lo, hi = np.minimum(lo, pts.min(0)), np.maximum(hi, pts.max(0))
    return lo, hi


# Azimuth axis onto world Z.
axis = bpy.data.objects["ORB_ROTOR_base"].matrix_world.translation
shift = Vector((-axis.x, -axis.y, 0.0))
root.location += shift
bpy.context.view_layer.update()
lo_before, hi_before = bounds(parts)

# Decimate the dense meshes once per shared mesh; instances keep sharing it.
faces_before = sum(len(o.data.polygons) for o in parts)
by_mesh: dict[str, list] = {}
for obj in parts:
    by_mesh.setdefault(obj.data.name, []).append(obj)
for group in by_mesh.values():
    obj = group[0]
    count = len(obj.data.polygons)
    if count <= MAX_FACES:
        continue
    bpy.ops.object.select_all(action="DESELECT")
    obj.hide_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    obj.data = obj.data.copy()
    mod = obj.modifiers.new("decimate", "DECIMATE")
    mod.ratio = max(0.3, MAX_FACES / count)
    mod.use_collapse_triangulate = True
    bpy.ops.object.modifier_apply(modifier=mod.name)
    for sibling in group[1:]:
        sibling.data = obj.data
lo, hi = bounds(parts)
deviation_mm = float(max(np.abs(lo - lo_before).max(), np.abs(hi - hi_before).max()) * 1000)
assert deviation_mm < 0.5, f"decimation moved the bounds by {deviation_mm:.3f} mm"

# Export the parts with all their ancestors (root, rotors), nothing else.
selected = set(parts)
for obj in parts:
    parent = obj.parent
    while parent:
        selected.add(parent)
        parent = parent.parent
assert all(bpy.data.objects[n] in selected for n in ROTORS)
bpy.ops.object.select_all(action="DESELECT")
for obj in selected:
    obj.hide_set(False)
    obj.select_set(True)
bpy.ops.export_scene.gltf(
    filepath=str(destination), export_format="GLB", use_selection=True,
    export_yup=True, export_animations=False, export_extras=True,
    export_cameras=False, export_lights=False, export_texcoords=False,
    export_normals=True, export_materials="EXPORT",
)


def mm(v):
    return [round(float(c) * 1000, 3) for c in v]


report = {
    "source": scene.get("orb_source", "Fusion Orbiter"),
    "frame": "metres, Y up (glTF); azimuth axis = Z axis of the Blender scene, machine on z = 0",
    "azimuth_axis_shift_mm": mm(shift),
    "rotors": {
        name: {
            "origin_mm": mm(bpy.data.objects[name].matrix_world.translation),
            "axis": bpy.data.objects[name].get("orb_axis"),
            "parts": sorted(o.name for o in parts if o.parent and o.parent.name == name),
        }
        for name in ROTORS
    },
    "parts": len(parts),
    "faces_before": faces_before,
    "faces_after": sum(len(o.data.polygons) for o in parts),
    "bounds_mm": [mm(lo), mm(hi)],
    "decimation_bounds_deviation_mm": round(deviation_mm, 4),
    "bytes": destination.stat().st_size,
}
destination.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print("GLB_EXPORT", json.dumps({k: v for k, v in report.items() if k != "rotors"}))
print("GLB_ROTORS", json.dumps({k: v["origin_mm"] for k, v in report["rotors"].items()}))
