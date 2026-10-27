"""
Headless Blender render script for the Orbiter v3 CAD screenshots.

Usage:
    blender -b <assembly.blend> -P render_screenshots.py -- <out_dir> [stl_root]

    assembly.blend  the v3 assembly scene: every occurrence is a mesh object
                    named '<group> / <Part>:<n>' with the Fusion placement
                    (the Blender project in the private hardware tree builds
                    it from the Fusion export; it is not part of this repo).
    out_dir         where the PNGs go.
    stl_root        folder with <colour>/<Part>.stl, default stl/printed next
                    to this script. Each printed part is rendered on its own
                    from its STL, the same way the assembly is.

Renders:
  - Assembly: front, side, top, iso views, 2560x1920.
  - Per-part: each printed STL, iso view.

Visual style is the one cad/render_screenshots.py set for v1, so the two
galleries read as one set: matte per-part colours from a muted palette
keyed by the part's role, a neutral mid-dark grey background, Cycles with
soft 3-point area lighting, Freestyle silhouette + crease lines.

Run-time hooks:
    ENV  BLENDER_RENDER_SAMPLES   override Cycles samples (default 96)
    ENV  BLENDER_RENDER_RESX      override resolution width
    ENV  BLENDER_RENDER_RESY      override resolution height
"""

import bpy
import math
import os
import sys
from pathlib import Path

from mathutils import Vector


# -------------------- args --------------------
argv = sys.argv
user_args = argv[argv.index("--") + 1:] if "--" in argv else []
if not user_args:
    raise SystemExit("usage: blender -b assembly.blend -P render_screenshots.py -- <out_dir> [stl_root]")

HERE = Path(__file__).resolve().parent
OUT_DIR = Path(user_args[0])
STL_ROOT = Path(user_args[1]) if len(user_args) > 1 else HERE / "stl" / "printed"

RES_X = int(os.environ.get("BLENDER_RENDER_RESX", "2560"))
RES_Y = int(os.environ.get("BLENDER_RENDER_RESY", "1920"))
SAMPLES = int(os.environ.get("BLENDER_RENDER_SAMPLES", "96"))


def srgb_to_linear(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def hex_to_linear_rgba(hexstr, alpha=1.0):
    h = hexstr.lstrip("#")
    return tuple(srgb_to_linear(int(h[i:i + 2], 16) / 255.0) for i in (0, 2, 4)) + (alpha,)


BG_COLOR = hex_to_linear_rgba("#383838")

# The v1 palette, plus two neutrals for the bought parts the v1 rig did
# not have as separate bodies (steel fasteners and bearings, black belts).
PALETTE = {
    "forest_green": "#2a4a3a",   # anchor: housings
    "brass":        "#b0853e",   # pulleys
    "terracotta":   "#a06048",   # inserts / platform
    "dusty_teal":   "#4a7a7a",   # rails
    "slate_blue":   "#4a5566",   # orbit arm
    "mustard":      "#a89048",   # spindle, shaft, clamp nuts
    "warm_grey":    "#6a6056",   # caps, wedges, spacers, mounts
    "dusty_rose":   "#9a6a72",   # sensor parts
    "steel":        "#7a7c80",   # bearings, fasteners, motors, inserts
    "rubber":       "#2e2c2a",   # belts
}

# Substring of the normalised object or STL name -> palette key.
# Most specific first. Names are normalised to lowercase alphanumerics, so
# 'BottomBody:1 / BasePulley GT2 112T:1' matches 'basepulley'.
PART_COLOR_MAP = [
    ("orbitarm",            "slate_blue"),
    ("mainrail",            "dusty_teal"),
    ("crossrail",           "dusty_teal"),
    ("vertivalbar",         "dusty_teal"),
    ("bottomhousin",        "forest_green"),
    ("basepulley",          "brass"),
    ("replacablepulley",    "brass"),
    ("gt216tpulley",        "brass"),
    ("baseplatforminsert",  "terracotta"),
    ("barmountinsert",      "terracotta"),
    ("basefootplug",        "terracotta"),
    ("basespindle",         "mustard"),
    ("elevationshaft",      "mustard"),
    ("clampnut",            "mustard"),
    ("basesensorring",      "dusty_rose"),
    ("anglesensormount",    "dusty_rose"),
    ("as5600",              "dusty_rose"),
    ("hexplug",             "dusty_rose"),
    ("belt",                "rubber"),
    ("bearing",             "steel"),
    ("ballset",             "steel"),
    ("innerring",           "steel"),
    ("outerring",           "steel"),
    ("nema17",              "steel"),
    ("iso4",                "steel"),
    ("iso7093",             "steel"),
    ("heatsetinsert",       "steel"),
    ("baseextractor",       "warm_grey"),
    ("wrench",              "warm_grey"),
]

# Rails are 'MainRail', 'CrossRail', 'VertivalBar': their caps and wedges
# carry the rail's name, so those go through the cap rule first.
CAP_WORDS = ("cap", "wedge", "spacer", "spreader", "jaws")


def normalise(name):
    return "".join(ch for ch in name.lower() if ch.isalnum())


def palette_key_for(name):
    norm = normalise(name)
    if any(w in norm for w in CAP_WORDS):
        return "warm_grey"
    for key, pal in PART_COLOR_MAP:
        if key in norm:
            return pal
    return "warm_grey"


def palette_material(key):
    mat_name = f"OrbiterV2_{key}"
    mat = bpy.data.materials.get(mat_name)
    if mat is not None:
        return mat
    mat = bpy.data.materials.new(mat_name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = hex_to_linear_rgba(PALETTE[key])
    bsdf.inputs["Roughness"].default_value = 0.6
    bsdf.inputs["Metallic"].default_value = 0.0
    for spec_key in ("Specular IOR Level", "Specular"):
        if spec_key in bsdf.inputs:
            bsdf.inputs[spec_key].default_value = 0.4
            break
    out = nodes.new("ShaderNodeOutputMaterial")
    links.new(bsdf.outputs[0], out.inputs[0])
    return mat


def assign_palette(mesh_objs, label):
    print(f"[palette] {label}:")
    for o in mesh_objs:
        key = palette_key_for(o.name)
        o.data.materials.clear()
        o.data.materials.append(palette_material(key))
        print(f"  - {o.name!r:60s} -> {key}")


# -------------------- scene --------------------
def setup_world():
    world = bpy.data.worlds.get("World") or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    tree = world.node_tree
    tree.nodes.clear()
    bg = tree.nodes.new("ShaderNodeBackground")
    bg.inputs["Color"].default_value = BG_COLOR
    bg.inputs["Strength"].default_value = 1.0
    out = tree.nodes.new("ShaderNodeOutputWorld")
    tree.links.new(bg.outputs[0], out.inputs[0])


def enable_cycles_gpu():
    prefs = bpy.context.preferences.addons.get("cycles")
    if prefs is None:
        return "CPU"
    cprefs = prefs.preferences
    for try_type in ("OPTIX", "CUDA", "HIP"):
        try:
            cprefs.compute_device_type = try_type
            cprefs.refresh_devices()
            any_on = False
            wanted = os.environ.get("ORB_GPU", "5060")  # one named card: the other one may belong to another job
            for d in cprefs.devices:
                if d.type == try_type and wanted in d.name:
                    d.use = True
                    any_on = True
                else:
                    d.use = False
            if any_on:
                return try_type
        except Exception:
            continue
    return "CPU"


def setup_render():
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.render.resolution_x = RES_X
    scene.render.resolution_y = RES_Y
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"

    scene.cycles.samples = SAMPLES
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = 0.01
    scene.cycles.adaptive_min_samples = 16
    scene.cycles.use_denoising = True
    scene.cycles.denoiser = "OPENIMAGEDENOISE"

    device_type = enable_cycles_gpu()
    scene.cycles.device = "GPU" if device_type != "CPU" else "CPU"
    print(f"[cycles] device: {device_type}")

    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0

    scene.render.use_freestyle = True
    scene.render.line_thickness_mode = "ABSOLUTE"
    scene.render.line_thickness = 1.6

    vl = bpy.context.view_layer
    vl.use_freestyle = True
    fs = vl.freestyle_settings
    fs.crease_angle = math.radians(140)
    fs.use_smoothness = True
    fs.use_culling = True
    lineset = fs.linesets.active if fs.linesets else fs.linesets.new("LineSet")
    lineset.select_silhouette = True
    lineset.select_border = True
    lineset.select_crease = True
    lineset.select_edge_mark = False
    lineset.select_contour = True
    lineset.select_external_contour = True
    lineset.select_suggestive_contour = False
    lineset.select_ridge_valley = False
    # A scene saved without Freestyle has a line set but no line style.
    if lineset.linestyle is None:
        lineset.linestyle = bpy.data.linestyles.get("LineStyle") or bpy.data.linestyles.new("LineStyle")
    ls = lineset.linestyle
    ls.color = (0.0, 0.0, 0.0)
    ls.alpha = 1.0
    ls.thickness = 1.6
    ls.thickness_position = "CENTER"


def world_bbox(mesh_objs):
    min_v = Vector((math.inf,) * 3)
    max_v = Vector((-math.inf,) * 3)
    for o in mesh_objs:
        for corner in o.bound_box:
            wc = o.matrix_world @ Vector(corner)
            for i in range(3):
                min_v[i] = min(min_v[i], wc[i])
                max_v[i] = max(max_v[i], wc[i])
    center = (min_v + max_v) * 0.5
    size = max_v - min_v
    return center, max(size), (size.x, size.y, size.z)


VIEW_DIRS = {
    "front": (Vector((0, -1, 0)), Vector((0, 0, 1))),
    "side":  (Vector((1, 0, 0)), Vector((0, 0, 1))),
    "top":   (Vector((0, 0, 1)), Vector((0, -1, 0))),
    "iso":   (Vector((0.7, -0.7, 0.55)).normalized(), Vector((0, 0, 1))),
}

RENDER_OBJECTS = (
    "KeyLight", "FillLight", "RimLight",
    "KeyLightTarget", "FillLightTarget", "RimLightTarget",
    "RenderCam", "CamTarget",
)


def add_lights(center, size_max, view):
    base = max(size_max * size_max, 0.01)
    cam_dir, up_dir = VIEW_DIRS[view]
    right = cam_dir.cross(up_dir).normalized()
    up = right.cross(cam_dir).normalized()

    def place(name, dir_vec, dist_mult, size_mult, energy_mult, color):
        loc = center + dir_vec * (size_max * dist_mult)
        bpy.ops.object.light_add(type="AREA", location=tuple(loc))
        light = bpy.context.active_object
        light.name = name
        light.data.size = size_max * size_mult
        light.data.energy = base * energy_mult
        light.data.color = color
        target = bpy.data.objects.new(f"{name}Target", None)
        target.location = center
        bpy.context.collection.objects.link(target)
        c = light.constraints.new(type="TRACK_TO")
        c.target = target
        c.track_axis = "TRACK_NEGATIVE_Z"
        c.up_axis = "UP_Y"

    place("KeyLight", (cam_dir * 0.8 + up * 1.2).normalized(), 2.0, 1.6, 320.0, (1.0, 0.98, 0.95))
    place("FillLight", (cam_dir * 0.5 - right * 1.0 + up * 0.3).normalized(), 2.2, 2.5, 180.0, (0.92, 0.95, 1.0))
    place("RimLight", (-cam_dir * 0.7 + up * 0.8 + right * 0.5).normalized(), 2.0, 1.2, 250.0, (1.0, 0.95, 0.88))


def place_camera(view, center, size_max, size_xyz):
    sx, sy, sz = size_xyz
    cam_data = bpy.data.cameras.new("RenderCam")
    cam = bpy.data.objects.new("RenderCam", cam_data)
    bpy.context.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    cam_data.type = "ORTHO"
    cam_data.clip_start = size_max * 0.001
    cam_data.clip_end = size_max * 100
    margin = 1.15
    aspect = RES_X / RES_Y
    dist = size_max * 4.0

    if view == "front":
        screen_w, screen_h = sx, sz
        cam.location = (center.x, center.y - dist, center.z)
        cam.rotation_euler = (math.radians(90), 0, 0)
    elif view == "side":
        screen_w, screen_h = sy, sz
        cam.location = (center.x + dist, center.y, center.z)
        cam.rotation_euler = (math.radians(90), 0, math.radians(90))
    elif view == "top":
        screen_w, screen_h = sx, sy
        cam.location = (center.x, center.y, center.z + dist)
        cam.rotation_euler = (0, 0, 0)
    else:
        diag = math.sqrt(sx * sx + sy * sy)
        screen_w, screen_h = diag, sz + 0.4 * diag
        cam.location = (center.x + dist * 0.7, center.y - dist * 0.7, center.z + dist * 0.55)
        target = bpy.data.objects.new("CamTarget", None)
        target.location = center
        bpy.context.collection.objects.link(target)
        c = cam.constraints.new(type="TRACK_TO")
        c.target = target
        c.track_axis = "TRACK_NEGATIVE_Z"
        c.up_axis = "UP_Y"

    cam_data.ortho_scale = max(screen_w * margin, screen_h * margin * aspect)


def remove_render_objects():
    for o in list(bpy.data.objects):
        if o.name in RENDER_OBJECTS or o.type in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(o, do_unlink=True)


def render_to(path):
    bpy.context.scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)
    print(f"[render] wrote {path}")


def shoot(mesh_objs, views, out_name):
    center, size_max, size_xyz = world_bbox(mesh_objs)
    written = []
    for view in views:
        remove_render_objects()
        add_lights(center, size_max, view)
        place_camera(view, center, size_max, size_xyz)
        out = OUT_DIR / out_name(view)
        render_to(out)
        written.append(out)
    return written


# -------------------- assembly --------------------
def render_assembly():
    """The scene the script was opened with is the assembly: keep every
    mesh that renders, drop the tools the model hides, recolour, shoot."""
    scene = bpy.context.scene
    meshes = [o for o in scene.objects if o.type == "MESH" and not o.hide_render]
    for o in scene.objects:
        if o.type == "MESH" and o.hide_render:
            o.hide_viewport = True
    print(f"[assembly] {len(meshes)} meshes")
    setup_world()
    setup_render()
    assign_palette(meshes, "assembly")
    return shoot(meshes, ("front", "side", "top", "iso"), lambda v: f"assembly_{v}.png")


# -------------------- parts --------------------
def part_file_stem(stl_path):
    """'BottomHousin(Mirror).stl' -> 'bottom_housin_mirror'."""
    stem = stl_path.stem
    out = []
    prev_lower = False
    for ch in stem:
        if ch.isalnum():
            if ch.isupper() and prev_lower:
                out.append("_")
            out.append(ch.lower())
            prev_lower = ch.islower() or ch.isdigit()
        else:
            if out and out[-1] != "_":
                out.append("_")
            prev_lower = False
    return "".join(out).strip("_")


def render_parts():
    stls = sorted(STL_ROOT.glob("*/*.stl"), key=lambda p: p.name.lower())
    print(f"[parts] {len(stls)} STL files under {STL_ROOT}")
    written = []
    for stl in stls:
        bpy.ops.wm.read_homefile(use_empty=True)
        setup_world()
        setup_render()
        bpy.ops.wm.stl_import(filepath=str(stl), global_scale=0.001)
        meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
        if not meshes:
            print(f"[parts] {stl.name}: nothing imported")
            continue
        for i, m in enumerate(meshes):
            m.name = stl.stem if i == 0 else f"{stl.stem}.{i:03d}"
        assign_palette(meshes, stl.name)
        name = part_file_stem(stl)
        written += shoot(meshes, ("iso",), lambda v: f"part_{name}.png")
    return written


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[main] out={OUT_DIR} stl_root={STL_ROOT} {RES_X}x{RES_Y} samples={SAMPLES}")
    asm = render_assembly()
    parts = render_parts()
    print(f"[main] done: {len(asm)} assembly views, {len(parts)} parts")


main()
