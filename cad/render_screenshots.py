"""
Headless Blender render script for Orbiter CAD release screenshots.

Usage:
    blender --background --python render_screenshots.py
    blender -b -P render_screenshots.py -- <project_root>

Renders:
  - Assembly: front, side, top, iso views from Orbiter.glb
  - Per-part: each printable STL on a neutral grey background

Visual style:
  - Per-part matte materials from a coherent muted designer palette
    (dark forest green anchor + warm brass, terracotta, dusty teal,
    slate blue-grey, soft mustard, warm grey, dusty rose). Each named
    part gets ONE color, consistent across assembly + per-part views.
  - Neutral mid-dark grey background (~#383838) — dark enough that the
    body pops without going pitch black.
  - Cycles engine with OptiX/CUDA acceleration when available, falling back
    to CPU gracefully.
  - Freestyle line set for crisp black silhouette + crease + border edges
    — this is what gives feature lines on orthographic views.
  - Soft 3-point area lighting + a subtle warm rim.
  - 2560x1920 output.

Run-time hooks for tuning per machine:
    ENV  BLENDER_RENDER_SAMPLES   override Cycles samples
    ENV  BLENDER_RENDER_RESX      override resolution width
    ENV  BLENDER_RENDER_RESY      override resolution height
"""

import bpy
import sys
import os
import math
from mathutils import Vector


# -------------------- args --------------------
argv = sys.argv
if "--" in argv:
    user_args = argv[argv.index("--") + 1:]
else:
    user_args = []

PROJECT_ROOT = user_args[0] if user_args else "D:/git-stack/hardware-lab/Orbiter"
PROJECT_ROOT = PROJECT_ROOT.replace("\\", "/")
SHOTS_DIR = f"{PROJECT_ROOT}/OrbiterV0.1/cad/screenshots"
ASM_DIR = f"{PROJECT_ROOT}/OrbiterV0.1/cad/assembly"
GLB_PATH = f"{PROJECT_ROOT}/Orbiter.glb"

PART_STLS = [
    ("AxisFrame.stl", "part_axis_frame.png"),
    ("BarHolderInsert.stl", "part_bar_holder_insert.png"),
    ("GT2 Pulley.stl", "part_gt2_pulley.png"),
    ("Hall Sensor Mount.stl", "part_hall_sensor_mount.png"),
    ("OrbitFrame.stl", "part_orbit_frame.png"),
    ("PlatformInsert.stl", "part_platform_insert.png"),
    ("TableSupportFrame.stl", "part_table_support_frame.png"),
]

RES_X = int(os.environ.get("BLENDER_RENDER_RESX", "2560"))
RES_Y = int(os.environ.get("BLENDER_RENDER_RESY", "1920"))
SAMPLES = int(os.environ.get("BLENDER_RENDER_SAMPLES", "96"))


# Colors are expressed as Principled-BSDF default_value tuples in
# Blender's linear scene-referred space. Blender's "Standard" view
# transform applies the sRGB encoding curve at output, so to hit a
# target sRGB hex we convert sRGB -> linear with the standard piecewise
# transform first.
def srgb_to_linear(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def hex_to_linear_rgba(hexstr, alpha=1.0):
    h = hexstr.lstrip("#")
    r = int(h[0:2], 16) / 255.0
    g = int(h[2:4], 16) / 255.0
    b = int(h[4:6], 16) / 255.0
    return (srgb_to_linear(r), srgb_to_linear(g), srgb_to_linear(b), alpha)


# Anchor color used as the fallback for unmatched part names.
BODY_COLOR_HEX = "#2a4a3a"            # dark forest green
BG_COLOR_HEX = "#383838"              # neutral mid-dark grey

BODY_COLOR = hex_to_linear_rgba(BODY_COLOR_HEX)
BG_COLOR = hex_to_linear_rgba(BG_COLOR_HEX)


# Designer palette: muted, matte, low-saturation, neighbouring on the
# colour wheel. Used per-part so adjacent components are visually
# distinct in the assembly views.
PALETTE = {
    "forest_green":   "#2a4a3a",   # anchor — heavy frame pieces
    "brass":          "#b0853e",   # rotating mechanics (pulleys)
    "terracotta":     "#a06048",   # inserts / platform surfaces
    "dusty_teal":     "#4a7a7a",   # structural supports / rails
    "slate_blue":     "#4a5566",   # large arm / orbit frame
    "mustard":        "#a89048",   # small accent parts
    "warm_grey":      "#6a6056",   # mounts / brackets
    "dusty_rose":     "#9a6a72",   # sensors (small high-contrast accents)
}


# Mapping: substring of an object's name (case-insensitive) -> palette key.
# First matching entry wins, so order from most specific to most generic.
# Names come from:
#   - GLB assembly  (e.g. 'BasePulley.001', 'OrbitArm.001', 'BaseMount.001')
#   - STL per-part  (file stems: 'AxisFrame', 'OrbitFrame', 'GT2 Pulley'...)
# Anything unmatched falls back to forest_green.
PART_COLOR_MAP = [
    # GLB names (substring match before name normalisation strips dots/digits)
    ("basemount",          "forest_green"),
    ("baseplatforminsert", "terracotta"),
    ("basepulley",         "brass"),
    ("basesensor",         "dusty_rose"),
    ("mainrail",           "dusty_teal"),
    ("orbitarm",           "slate_blue"),
    ("orbitbarinsert",     "terracotta"),
    ("orbitmount",         "warm_grey"),

    # STL filename stems (per-part renders) — keep colours consistent
    # with the assembly equivalents so the gallery reads as one set.
    ("axisframe",          "slate_blue"),     # = OrbitArm
    ("orbitframe",         "forest_green"),   # = BaseMount/main frame
    ("tablesupportframe",  "dusty_teal"),     # = MainRail
    ("barholderinsert",    "terracotta"),     # = OrbitBarInsert
    ("platforminsert",     "terracotta"),     # = BasePlatformInsert
    ("gt2pulley",          "brass"),          # = BasePulley
    ("hallsensormount",    "dusty_rose"),     # = BaseSensor
]


def _normalise_name(name):
    """Lowercase + strip dots, colons, spaces — for fuzzy matching.

    Blender suffixes duplicates with .001/.002 and the GLB carries ':1'/':2'
    instance markers; we want 'BasePulley.001' and 'BasePulley:2' to both
    hit the 'basepulley' key. STL imports may add spaces ('GT2 Pulley'
    -> 'gt2pulley'). Digits are kept so 'gt2pulley' still matches.
    """
    out = []
    for ch in name.lower():
        if ch.isalnum():
            out.append(ch)
    return "".join(out)


def palette_color_for(obj_name):
    """Return (palette_key, hex) for a given object name, or anchor fallback."""
    norm = _normalise_name(obj_name)
    for key_substr, pal_key in PART_COLOR_MAP:
        if key_substr in norm:
            return pal_key, PALETTE[pal_key]
    return "forest_green", PALETTE["forest_green"]


# -------------------- helpers --------------------
def clear_scene():
    """Remove all objects, meshes, materials, lights, cameras."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for block_collection in (
        bpy.data.meshes,
        bpy.data.materials,
        bpy.data.lights,
        bpy.data.cameras,
        bpy.data.images,
        bpy.data.armatures,
        bpy.data.curves,
    ):
        for block in list(block_collection):
            try:
                block_collection.remove(block, do_unlink=True)
            except Exception:
                pass


def setup_world():
    """Solid neutral grey world background via shader nodes.

    Cycles ignores world.color when world.use_nodes is True (which it is
    by default in 5.x), so we wire a Background shader with our target
    linear color. We also lower the background strength a touch so the
    object's own shading dominates.
    """
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


def _enable_cycles_gpu():
    """Try to enable OptiX/CUDA/HIP, returning the device type chosen.

    Falls through to CPU if no GPU is configured. Always safe to call.
    """
    prefs = bpy.context.preferences.addons.get("cycles")
    if prefs is None:
        return "CPU"
    cprefs = prefs.preferences
    for try_type in ("OPTIX", "CUDA", "HIP"):
        try:
            cprefs.compute_device_type = try_type
            cprefs.refresh_devices()
            any_on = False
            for d in cprefs.devices:
                if d.type == try_type:
                    d.use = True
                    any_on = True
                elif d.type == "CPU":
                    d.use = False
            if any_on:
                return try_type
        except Exception:
            continue
    return "CPU"


def setup_render(transparent_bg=False):
    """Configure Cycles with denoising and Freestyle line rendering.

    Cycles gives us proper colour, soft shadows from area lights, and
    plays nicely with Freestyle for crisp edge overlays. Workbench's
    cavity-only approach hides too many flat planar features on
    orthographic views — that was the previous batch's failure mode.
    """
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.render.resolution_x = RES_X
    scene.render.resolution_y = RES_Y
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = transparent_bg
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA" if transparent_bg else "RGB"
    scene.render.image_settings.color_depth = "8"

    # Cycles sampling
    scene.cycles.samples = SAMPLES
    try:
        scene.cycles.use_adaptive_sampling = True
        scene.cycles.adaptive_threshold = 0.01
        scene.cycles.adaptive_min_samples = 16
    except Exception:
        pass
    # Denoise — OptiX denoiser if we have it, fallback to OpenImageDenoise
    try:
        scene.cycles.use_denoising = True
        scene.cycles.denoiser = "OPENIMAGEDENOISE"
    except Exception:
        pass

    device_type = _enable_cycles_gpu()
    if device_type != "CPU":
        scene.cycles.device = "GPU"
        print(f"[cycles] using GPU device type: {device_type}")
    else:
        scene.cycles.device = "CPU"
        print("[cycles] using CPU")

    # Color management — Standard view, modest exposure
    try:
        scene.view_settings.view_transform = "Standard"
        scene.view_settings.look = "None"
        scene.view_settings.exposure = 0.0
        scene.view_settings.gamma = 1.0
    except Exception:
        pass

    # Freestyle — silhouette + crease + border edge lines, black ink.
    # This is the key fix: even on orthographic views where Cavity-only
    # would blur planar feature lines, Freestyle traces every face
    # boundary above the crease angle.
    scene.render.use_freestyle = True
    try:
        scene.render.line_thickness_mode = "ABSOLUTE"
        scene.render.line_thickness = 1.6  # ~1.6 px lines
    except Exception:
        pass

    vl = bpy.context.view_layer
    vl.use_freestyle = True
    fs = vl.freestyle_settings
    fs.crease_angle = math.radians(140)  # mark edges with angle <140deg as creases
    fs.use_smoothness = True
    fs.use_culling = True

    # Configure (or create) the default line set + its line style.
    lineset = fs.linesets.active if fs.linesets else fs.linesets.new("LineSet")
    lineset.select_silhouette = True
    lineset.select_border = True
    lineset.select_crease = True
    lineset.select_edge_mark = False
    lineset.select_contour = True
    lineset.select_external_contour = True
    # Don't draw lines on suggested-contour (too noisy on round parts)
    try:
        lineset.select_suggestive_contour = False
        lineset.select_ridge_valley = False
    except Exception:
        pass

    ls = lineset.linestyle
    ls.color = (0.0, 0.0, 0.0)
    ls.alpha = 1.0
    ls.thickness = 1.6
    try:
        ls.thickness_position = "CENTER"
    except Exception:
        pass


def make_body_material(name="OrbiterBody", color_rgba=None):
    """Matte Principled BSDF for a single body. Defaults to anchor green.

    Roughness 0.6 keeps it readable on flat panels without going chalky.
    A small Specular IOR Level lift (0.4 -> default 0.5 was too shiny on
    iso, 0.4 dials it back) keeps reflections discreet.

    color_rgba is a linear-space RGBA tuple (see hex_to_linear_rgba).
    """
    if color_rgba is None:
        color_rgba = BODY_COLOR
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = color_rgba
    try:
        bsdf.inputs["Roughness"].default_value = 0.6
    except Exception:
        pass
    try:
        bsdf.inputs["Metallic"].default_value = 0.0
    except Exception:
        pass
    # Principled in 4.x/5.x uses "Specular IOR Level" (was "Specular" pre-4).
    for spec_key in ("Specular IOR Level", "Specular"):
        if spec_key in bsdf.inputs:
            try:
                bsdf.inputs[spec_key].default_value = 0.4
            except Exception:
                pass
            break
    out = nodes.new("ShaderNodeOutputMaterial")
    links.new(bsdf.outputs[0], out.inputs[0])
    return mat


# Cache of palette-key -> Material so we re-use one material per colour
# across many meshes (e.g. both BasePulley:1 and BasePulley:2 share brass).
_PALETTE_MAT_CACHE = {}


def get_palette_material(palette_key):
    """Return (creating if needed) the Material for a palette key.

    Materials get wiped by clear_scene() between renders, so we look up
    by name in bpy.data.materials each call and rebuild on cache miss.
    """
    mat_name = f"Orbiter_{palette_key}"
    existing = bpy.data.materials.get(mat_name)
    if existing is not None:
        _PALETTE_MAT_CACHE[palette_key] = existing
        return existing
    hex_color = PALETTE[palette_key]
    rgba = hex_to_linear_rgba(hex_color)
    mat = make_body_material(name=mat_name, color_rgba=rgba)
    _PALETTE_MAT_CACHE[palette_key] = mat
    return mat


def assign_palette_materials(mesh_objs, source_label=""):
    """Walk meshes, pick a palette colour per object name, assign material.

    Prints a one-line summary per object so the mapping is auditable in
    the render log.
    """
    print(f"[palette] assigning per-part materials ({source_label}):")
    for o in mesh_objs:
        pal_key, hex_color = palette_color_for(o.name)
        mat = get_palette_material(pal_key)
        o.data.materials.clear()
        o.data.materials.append(mat)
        print(f"  - {o.name!r:40s} -> {pal_key} ({hex_color})")


def apply_material_recursive(obj, mat):
    if obj.type == "MESH":
        obj.data.materials.clear()
        obj.data.materials.append(mat)
    for child in obj.children:
        apply_material_recursive(child, mat)


def collect_meshes(roots):
    """Walk a list of root objects and return all descendant mesh objects."""
    meshes = []
    stack = list(roots)
    while stack:
        o = stack.pop()
        if o.type == "MESH":
            meshes.append(o)
        stack.extend(list(o.children))
    return meshes


def compute_world_bbox(mesh_objs):
    """Return (min_v, max_v, center, size_max, size_tuple) in world space."""
    if not mesh_objs:
        return (
            Vector((-1, -1, -1)),
            Vector((1, 1, 1)),
            Vector((0, 0, 0)),
            2.0,
            (2.0, 2.0, 2.0),
        )
    min_v = Vector((math.inf, math.inf, math.inf))
    max_v = Vector((-math.inf, -math.inf, -math.inf))
    for o in mesh_objs:
        for corner in o.bound_box:
            wc = o.matrix_world @ Vector(corner)
            for i in range(3):
                if wc[i] < min_v[i]:
                    min_v[i] = wc[i]
                if wc[i] > max_v[i]:
                    max_v[i] = wc[i]
    center = (min_v + max_v) * 0.5
    size = max_v - min_v
    size_max = max(size.x, size.y, size.z)
    return min_v, max_v, center, size_max, (size.x, size.y, size.z)


def add_lights(center, size_max, view="iso"):
    """Soft 3-point lighting that follows the camera view.

    The lights are placed relative to the *view direction* (the unit
    vector pointing from the model toward the camera). This way the
    front/side/top/iso renders all get sensible key/fill/rim coverage
    instead of one view ending up backlit just because the lights were
    keyed to world space. Area-light energy scales with area, so we tune
    relative to size_max.
    """
    base = max(size_max * size_max, 0.01)

    # cam_dir = unit vector from center toward the camera, per view.
    # up_dir   = world "up" used to derive the key's overhead position.
    if view == "front":          # cam at -Y, up = +Z
        cam_dir = Vector((0, -1, 0))
        up_dir  = Vector((0, 0, 1))
    elif view == "side":         # cam at +X, up = +Z
        cam_dir = Vector((1, 0, 0))
        up_dir  = Vector((0, 0, 1))
    elif view == "top":          # cam at +Z, up arbitrary horizontal
        cam_dir = Vector((0, 0, 1))
        up_dir  = Vector((0, -1, 0))
    else:                        # iso 3/4 view
        cam_dir = Vector((0.7, -0.7, 0.55)).normalized()
        up_dir  = Vector((0, 0, 1))

    # right_dir is orthogonal to both cam_dir and up_dir
    right_dir = cam_dir.cross(up_dir).normalized()
    # corrected up perpendicular to cam_dir
    up_perp = right_dir.cross(cam_dir).normalized()

    def place(name, dir_vec, dist_mult, size_mult, energy_mult, color, look_target):
        loc = center + dir_vec * (size_max * dist_mult)
        bpy.ops.object.light_add(type="AREA", location=tuple(loc))
        light = bpy.context.active_object
        light.name = name
        light.data.size = size_max * size_mult
        light.data.energy = base * energy_mult
        light.data.color = color
        # Aim the light at the model center
        track_empty = bpy.data.objects.new(f"{name}Target", None)
        track_empty.location = look_target
        bpy.context.collection.objects.link(track_empty)
        c = light.constraints.new(type="TRACK_TO")
        c.target = track_empty
        c.track_axis = "TRACK_NEGATIVE_Z"
        c.up_axis = "UP_Y"
        return light

    # Key: upper-front (camera-side, above)
    key_dir = (cam_dir * 0.8 + up_perp * 1.2).normalized()
    place("KeyLight", key_dir, 2.0, 1.6, 320.0, (1.0, 0.98, 0.95), center)

    # Fill: opposite side of camera, lower, softer
    fill_dir = (cam_dir * 0.5 - right_dir * 1.0 + up_perp * 0.3).normalized()
    place("FillLight", fill_dir, 2.2, 2.5, 180.0, (0.92, 0.95, 1.0), center)

    # Rim: from behind the model (opposite of camera), elevated
    rim_dir = (-cam_dir * 0.7 + up_perp * 0.8 + right_dir * 0.5).normalized()
    place("RimLight", rim_dir, 2.0, 1.2, 250.0, (1.0, 0.95, 0.88), center)


def place_camera(view, center, size_max, size_xyz=None):
    """Position an orthographic camera for view: 'front'|'side'|'top'|'iso'.

    Orthographic framing is deterministic for product shots: we set
    ortho_scale to the larger of the model's two on-screen extents times
    a margin factor and aim the camera at the bbox center.
    """
    if size_xyz is None:
        size_xyz = (size_max, size_max, size_max)
    sx, sy, sz = size_xyz

    cam_data = bpy.data.cameras.new("RenderCam")
    cam_obj = bpy.data.objects.new("RenderCam", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj

    cam_data.type = "ORTHO"
    cam_data.clip_start = 0.0001
    cam_data.clip_end = size_max * 100

    margin = 1.15  # ~15% padding around model
    aspect = RES_X / RES_Y

    dist = size_max * 4.0

    if view == "front":
        screen_w = sx
        screen_h = sz
        cam_obj.location = (center.x, center.y - dist, center.z)
        cam_obj.rotation_euler = (math.radians(90), 0, 0)
    elif view == "side":
        screen_w = sy
        screen_h = sz
        cam_obj.location = (center.x + dist, center.y, center.z)
        cam_obj.rotation_euler = (math.radians(90), 0, math.radians(90))
    elif view == "top":
        screen_w = sx
        screen_h = sy
        cam_obj.location = (center.x, center.y, center.z + dist)
        cam_obj.rotation_euler = (0, 0, 0)
    else:  # iso 3/4 view
        screen_w = math.sqrt(sx * sx + sy * sy)
        screen_h = sz + 0.4 * math.sqrt(sx * sx + sy * sy)
        d = dist
        cam_obj.location = (
            center.x + d * 0.7,
            center.y - d * 0.7,
            center.z + d * 0.55,
        )
        empty = bpy.data.objects.new("CamTarget", None)
        empty.location = center
        bpy.context.collection.objects.link(empty)
        c = cam_obj.constraints.new(type="TRACK_TO")
        c.target = empty
        c.track_axis = "TRACK_NEGATIVE_Z"
        c.up_axis = "UP_Y"

    needed_for_width = screen_w * margin
    needed_for_height = screen_h * margin * aspect
    cam_data.ortho_scale = max(needed_for_width, needed_for_height)

    return cam_obj


def remove_cameras_lights():
    """Remove only the render cam/lights/track-target we added.

    We intentionally do NOT touch EMPTY objects in general because GLB
    imports use empties as parent transform carriers for the model
    hierarchy; deleting them would collapse the model to the origin.
    """
    render_obj_names = {
        "KeyLight", "FillLight", "RimLight",
        "KeyLightTarget", "FillLightTarget", "RimLightTarget",
        "RenderCam", "CamTarget",
    }
    for o in list(bpy.data.objects):
        if o.name in render_obj_names or o.type in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(o, do_unlink=True)


def render_to(filepath):
    bpy.context.scene.render.filepath = filepath
    bpy.ops.render.render(write_still=True)
    print(f"[render] wrote {filepath}")


# -------------------- assembly renders --------------------
def render_assembly():
    print(f"[assembly] importing {GLB_PATH}")
    clear_scene()
    setup_world()
    setup_render(transparent_bg=False)

    bpy.ops.import_scene.gltf(filepath=GLB_PATH)

    all_meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    if not all_meshes:
        print("[assembly] no meshes imported from GLB")
        return []

    # Assign palette colours per part so adjacent components are
    # distinguishable. Falls back to the anchor forest_green for any
    # mesh whose name does not match an entry in PART_COLOR_MAP.
    assign_palette_materials(all_meshes, source_label="GLB assembly")

    _, _, center, size_max, size_xyz = compute_world_bbox(all_meshes)
    print(
        f"[assembly] center={tuple(round(v,3) for v in center)}, "
        f"size_xyz={tuple(round(v,3) for v in size_xyz)}, size_max={size_max:.3f}"
    )

    written = []
    for view in ("front", "side", "top", "iso"):
        remove_cameras_lights()
        add_lights(center, size_max, view=view)
        place_camera(view, center, size_max, size_xyz)
        out = f"{SHOTS_DIR}/assembly_{view}.png"
        render_to(out)
        written.append(out)

    return written


# -------------------- per-part renders --------------------
def render_parts():
    written = []
    for stl_name, out_name in PART_STLS:
        stl_path = f"{PROJECT_ROOT}/{stl_name}"
        if not os.path.exists(stl_path):
            print(f"[part] missing {stl_path}, skip")
            continue
        clear_scene()
        setup_world()
        setup_render(transparent_bg=False)

        imported_ok = False
        try:
            bpy.ops.wm.stl_import(filepath=stl_path)
            imported_ok = True
        except Exception as e:
            print(f"[part] wm.stl_import failed for {stl_name}: {e}")
            try:
                bpy.ops.import_mesh.stl(filepath=stl_path)
                imported_ok = True
            except Exception as e2:
                print(f"[part] import_mesh.stl also failed: {e2}")
        if not imported_ok:
            continue

        meshes = [o for o in bpy.data.objects if o.type == "MESH"]
        if not meshes:
            print(f"[part] {stl_name} produced no meshes")
            continue

        # The STL importer derives an object name from the file stem
        # (e.g. 'AxisFrame' or 'GT2 Pulley'). Use that to pick the same
        # palette colour the assembly view uses for this part.
        # Force the mesh-object's name to the STL stem so the matcher
        # has a clean key even if Blender appended a counter.
        stl_stem = os.path.splitext(stl_name)[0]
        for idx, m in enumerate(meshes):
            m.name = stl_stem if idx == 0 else f"{stl_stem}.{idx:03d}"
        assign_palette_materials(meshes, source_label=f"STL {stl_name}")

        _, _, center, size_max, size_xyz = compute_world_bbox(meshes)
        print(
            f"[part] {stl_name} center={tuple(round(v,3) for v in center)}, "
            f"size_max={size_max:.3f}"
        )

        remove_cameras_lights()
        add_lights(center, size_max, view="iso")
        place_camera("iso", center, size_max, size_xyz)
        out = f"{SHOTS_DIR}/{out_name}"
        render_to(out)
        written.append(out)
    return written


# -------------------- main --------------------
def main():
    print(f"[main] project={PROJECT_ROOT}")
    print(f"[main] shots dir={SHOTS_DIR}")
    print(f"[main] asm dir={ASM_DIR}")
    print(f"[main] resolution {RES_X}x{RES_Y}, samples={SAMPLES}")
    asm_files = render_assembly()
    part_files = render_parts()
    print("[main] done")
    print(f"[main] assembly files ({len(asm_files)}):")
    for f in asm_files:
        print(f"  - {f}")
    print(f"[main] part files ({len(part_files)}):")
    for f in part_files:
        print(f"  - {f}")


main()
