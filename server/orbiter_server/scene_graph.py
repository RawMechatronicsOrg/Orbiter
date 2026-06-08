"""Scene-graph builder — turns `ModelState` into an abstract node tree.

This is the ONLY server module with Orbiter domain knowledge about the 3D
scene; the frontend renderer is domain-agnostic and just draws whatever nodes
arrive. A node is a plain JSON-serialisable dict (see `_node`); the node
taxonomy mirrors Viser's (frame / grid / mesh / line_segments / point_cloud /
image_plane / camera_frustum / label).

The rig is drawn in the static world frame (visual az = 0); only the platform
disc (and scan-preview points) rotate with the live azimuth.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

from geom.machine_model import Cylinder, compute_rig
from geom.pose import GeomParams, camera_pose_at
from geom.rig_model import load_rig, to_three_name
from geom.scan_path import plan_scan_path
from geom.transforms import (
    quat_from_unit_vectors,
    quat_mul as _quat_mul,
    quat_to_matrix as _quat_to_matrix,
    roll_about_lens_quat as _roll_about_lens_quat,
)
from orbiter_model import ModelState

log = logging.getLogger("orbiter.scene_graph")

Node = dict[str, Any]

# Quaternion for a +Y-native cylinder stood up along +Z (Euler [pi/2, 0, 0]).
_Q_Y_TO_Z = (math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5))


def _node(
    node_id: str,
    node_type: str,
    props: dict[str, Any],
    *,
    parent: str | None = None,
    position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    quaternion: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
    visible: bool = True,
    pickable: bool = False,
) -> Node:
    """Build one scene node. `transform` is local (relative to `parent`)."""
    return {
        "id": node_id,
        "parent": parent,
        "type": node_type,
        "transform": {
            "position": list(position),
            "quaternion": list(quaternion),
            "scale": list(scale),
        },
        "visible": visible,
        "pickable": pickable,
        "props": props,
    }


def _rot_z(deg: float) -> tuple[float, float, float, float]:
    """Quaternion for a rotation about +Z by `deg`."""
    half = math.radians(deg) / 2.0
    return (0.0, 0.0, math.sin(half), math.cos(half))


def _to_platform_local(
    pos: tuple[float, float, float],
    az_deg: float,
    math_anchor: tuple[float, float, float],
    subj_centre: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Re-frame a stored OBJECT/math-frame camera position
    (`camera_pose_at(az, el).camera_xyz_mm`) into the `platform_spin` LOCAL
    frame, so that under platform_spin it renders EXACTLY where the LIVE camera
    sits at that azimuth (and then rotates with the disc).

    The live-camera path maps math→world as `math_anchor + Rz90·p` (scene
    build, the `cam_world` line) while platform_spin applies `Rz(−az)`. Solving
    for the local position that lands on the live camera at the capture az:
        local = Rz(az)·(math_anchor − subj_centre) + Rz90·p
    (numerically verified to coincide with the live camera to ~0 mm). The
    matching orientation is `_quat_mul(_rot_z(90), stored_quat)`.
    """
    a = math.radians(az_deg)
    ca, sa = math.cos(a), math.sin(a)
    dx = math_anchor[0] - subj_centre[0]
    dy = math_anchor[1] - subj_centre[1]
    dz = math_anchor[2] - subj_centre[2]
    # Rz(az)·(dx,dy,dz)  +  Rz90·pos   (Rz90: (x,y,z) → (−y, x, z))
    return (
        ca * dx - sa * dy - pos[1],
        sa * dx + ca * dy + pos[0],
        dz + pos[2],
    )


def _rot_x(deg: float) -> tuple[float, float, float, float]:
    """Quaternion for a rotation about +X by `deg`."""
    half = math.radians(deg) / 2.0
    return (math.sin(half), 0.0, 0.0, math.cos(half))


def _slug(name: str) -> str:
    """`BasePulley:2` -> `basepulley_2` — colon isn't friendly in a node id."""
    return name.replace(":", "_").lower()


def _mesh(
    node_id: str,
    primitive: dict[str, Any],
    color: str,
    *,
    parent: str | None = None,
    position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    quaternion: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
    mat_type: str = "standard",
    pickable: bool = False,
    extra_mat: dict[str, Any] | None = None,
) -> Node:
    material: dict[str, Any] = {"type": mat_type, "color": color}
    if extra_mat:
        material.update(extra_mat)
    return _node(
        node_id, "mesh",
        {"primitive": primitive, "material": material},
        parent=parent, position=position, quaternion=quaternion, pickable=pickable,
    )


def _geom_params(model: ModelState) -> GeomParams:
    return GeomParams(
        extrinsic=None,
        arm_radius=max(model.arm_radius_mm, 1.0),
        camera_offset=model.camera_offset_mm,
        # Camera optical axis is assumed to be along the arm direction
        # (arm ⊥ phone screen plane). The phone IMU drives EL alignment
        # now; an extra camera_tilt would fight that calibration.
        camera_tilt=0.0,
        camera_pan=model.camera_pan_deg,
        turntable_axis=getattr(model, "turntable_axis", None),
    )


def _cylinder_node(
    node_id: str, cyl: Cylinder, radius: float, color: str,
    *, parent: str | None = None,
) -> Node:
    return _mesh(
        node_id,
        {"kind": "cylinder", "radius": radius, "height": cyl.length, "segments": 16},
        color,
        position=cyl.mid,
        quaternion=cyl.quat,
        parent=parent,
    )


#: Default arm radius (mm) when the operator hasn't set one yet — keeps the
#: camera body + frustum visible from a fresh boot so the operator sees the
#: rig immediately.
_DEFAULT_ARM_RADIUS_MM = 120.0

#: Yaw of the math camera model (deg about world +Z, CCW from above) so the
#: procedural arm/camera line up with the CAD's el-pulley + arm direction.
#: The math arm direction at az=0 is +X; the CAD arm lives along world +Y -
#: 90 deg CCW takes math +X onto world +Y. Applied at the `rig_math` anchor
#: (NOT at the rig parent), so only the procedural camera viz rotates - the
#: static CAD bodies and captures stay put.
_MATH_YAW_CCW_DEG = 90.0


def _effective_arm_radius(model: ModelState) -> float:
    """`model.arm_radius_mm` if positive, else the CAD-ish default."""
    return model.arm_radius_mm if model.arm_radius_mm > 0 else _DEFAULT_ARM_RADIUS_MM


def build_scene(model: ModelState) -> list[Node]:
    """Compute the full scene-node list from the current model state.

    Tree (logical groups under one `world` root for the Scene Explorer):

        world
        |- overlays      axes + grid
        |- rig           the machine itself
        |  |- rig_cad        the static GLB
        |  |- platform_spin  rotating subject parts + disc
        |  |- orbit_spin     rotating arm parts
        |  |- rig_arm        dynamic stem
        |  |- rig_camera     camera body
        |  |- live_frustum   current camera frustum
        |- capture_layer captures + loaded captures + scan preview
    """
    nodes: list[Node] = []

    # ── world overlays ─────────────────────────────────────────────────────
    nodes.append(_node("overlays", "frame", {}))
    nodes.append(_node("axes", "frame", {"axesLength": 40.0},
                       parent="overlays", visible=model.show_axes))
    nodes.append(_node(
        "grid", "grid",
        {"size": 320.0, "divisions": 16, "color": "#1e3a5a"},
        parent="overlays",
        position=(0.0, 0.0, 0.0),
    ))

    nodes.append(_node("rig", "frame", {}))
    nodes.append(_node("capture_layer", "frame", {}))

    # ── CAD rig — geometry queried from Orbiter.glb (server-side) ─────────
    rig_cad = load_rig()
    _CAD_URL = "/Orbiter.glb"
    _CAD_SCALE_MM = 1000.0

    _SUBJECT_PARTS = ("BasePlatformInsert:1", "BasePulley:1")
    _ORBIT_PARTS = ("BasePulley:2", "OrbitBarInsert:1", "OrbitArm:1")
    _ROTATING = list(_SUBJECT_PARTS) + list(_ORBIT_PARTS)
    nodes.append(_node(
        "rig_cad", "cad_model",
        {
            "url": _CAD_URL,
            "scale": _CAD_SCALE_MM,
            "convert_y_up": True,
            "hide_parts": [to_three_name(n) for n in _ROTATING],
        },
        parent="rig",
        position=(0.0, 0.0, 0.0),
    ))

    # ── subject-disk spin (model.az + operator offset around Z) ───────────
    subj_centre = rig_cad.part_centre("BasePulley:1") or (0.0, 0.0, 0.0)
    nodes.append(_node(
        "platform_spin", "frame", {},
        parent="rig",
        position=subj_centre,
        quaternion=_rot_z(
            -(model.az + float(getattr(model, "az_kinematic_offset_deg", 0.0)))
        ),
    ))
    for part_name in _SUBJECT_PARTS:
        nodes.append(_node(
            f"part_{_slug(part_name)}", "cad_part",
            {
                "url": _CAD_URL,
                "part_name": to_three_name(part_name),
                "scale": _CAD_SCALE_MM,
                "convert_y_up": True,
                "pivot_offset": list(subj_centre),
            },
            parent="platform_spin",
        ))

    # ── orbit arm spin (rotation about pulley's natural +X axis) ──────────
    orbit_centre = rig_cad.part_centre("BasePulley:2") or (0.0, 0.0, 0.0)
    math_anchor = (subj_centre[0], subj_centre[1], orbit_centre[2])
    tip_max_y = rig_cad.part_extreme_vertex("OrbitArm:1", axis=1, sign=+1)

    el_with_offset = model.el + float(getattr(model, "el_kinematic_offset_deg", 0.0))
    rig = compute_rig(
        el_deg=el_with_offset,
        arm_radius_mm=_effective_arm_radius(model),
        camera_offset_mm=model.camera_offset_mm,
        # Tilt is zeroed by design — see _geom_params above.
        camera_tilt_deg=0.0,
        camera_pan_deg=model.camera_pan_deg,
        extrinsic=_geom_params(model).extrinsic,
        turntable_axis=_geom_params(model).turntable_axis,
        machine_geometry=None,
    )

    cam_math = np.asarray(rig.camera_pos, dtype=float)
    cam_world = np.asarray(math_anchor, dtype=float) + np.array(
        [-cam_math[1], cam_math[0], cam_math[2]],
    )
    cam_yz = (cam_world[1] - orbit_centre[1], cam_world[2] - orbit_centre[2])
    cam_yz_norm = math.hypot(cam_yz[0], cam_yz[1])

    orbit_spin_theta_deg = float(model.el)
    stem_start_math: tuple[float, float, float] | None = None
    if tip_max_y is not None and cam_yz_norm > 1e-3:
        picked = (
            tip_max_y[0] - orbit_centre[0],
            tip_max_y[1] - orbit_centre[1],
            tip_max_y[2] - orbit_centre[2],
        )
        a0 = math.atan2(picked[2], picked[1])
        a1 = math.atan2(cam_yz[1], cam_yz[0])
        theta = math.degrees(a1 - a0)
        theta = ((theta + 180.0) % 360.0) - 180.0
        orbit_spin_theta_deg = theta

        c, s = math.cos(math.radians(theta)), math.sin(math.radians(theta))
        y_rot = picked[1] * c - picked[2] * s
        z_rot = picked[1] * s + picked[2] * c
        tip_world = (
            orbit_centre[0] + picked[0],
            orbit_centre[1] + y_rot,
            orbit_centre[2] + z_rot,
        )
        d = (tip_world[0] - math_anchor[0],
             tip_world[1] - math_anchor[1],
             tip_world[2] - math_anchor[2])
        stem_start_math = (d[1], -d[0], d[2])

    nodes.append(_node(
        "orbit_spin", "frame", {},
        parent="rig",
        position=orbit_centre,
        quaternion=_rot_x(orbit_spin_theta_deg),
    ))
    for part_name in _ORBIT_PARTS:
        nodes.append(_node(
            f"part_{_slug(part_name)}", "cad_part",
            {
                "url": _CAD_URL,
                "part_name": to_three_name(part_name),
                "scale": _CAD_SCALE_MM,
                "convert_y_up": True,
                "pivot_offset": list(orbit_centre),
            },
            parent="orbit_spin",
        ))

    # ── subject disk — top-centre of BasePlatformInsert ────────────────────
    insert_top = rig_cad.part_top_centre("BasePlatformInsert:1") or (0.0, 0.0, 0.0)
    disc_height = 15.0
    disc_local = (
        insert_top[0] - subj_centre[0],
        insert_top[1] - subj_centre[1],
        insert_top[2] - subj_centre[2] + disc_height / 2.0,
    )
    nodes.append(_node(
        "platform_disc", "disc_dial",
        {
            "radius": 140.0,
            "height": disc_height,
            "ground_color": "#0e0e12",
            "marks_color": "#f5f5f7",
            "label_color": "#ffffff",
            "tick_step_deg": 5,
            "major_step_deg": 30,
            "label_step_deg": 10,
            "label_major_px": 46,
            "label_minor_px": 22,
        },
        parent="platform_spin",
        position=disc_local,
    ))

    # `rig_math`: anchor for the procedural camera math (camera body,
    # frustum, cam_stem). Sits at the AZ-EL axes intersection with a 90 CCW
    # yaw so math +X (arm direction at az=el=0) lands on world +Y.
    nodes.append(_node(
        "rig_math", "frame", {},
        parent="rig",
        position=math_anchor,
        quaternion=_rot_z(_MATH_YAW_CCW_DEG),
    ))

    # Sub-group so arm visualisations cluster cleanly in the Scene Explorer.
    nodes.append(_node("rig_arm", "frame", {}, parent="rig_math"))

    try:
        nodes.extend(_arm_nodes(model, rig, stem_start_math))
    except Exception:  # noqa: BLE001
        log.exception("arm rendering failed — continuing without it")

    # Camera body — always rendered (uses the default arm radius when the
    # operator hasn't set one yet, so the rig isn't invisible from a fresh
    # boot). All children are local to rig_camera.
    nodes.append(_node(
        "rig_camera", "frame", {},
        parent="rig_math",
        position=rig.camera_pos, quaternion=rig.camera_quat, pickable=True,
    ))
    nodes.append(_mesh(
        "rig_cam_box",
        {"kind": "box", "width": 34.0, "height": 26.0, "depth": 22.0},
        "#475569",
        parent="rig_camera", position=(0.0, 0.0, 6.0),
    ))
    nodes.append(_mesh(
        "rig_cam_lens",
        {"kind": "cylinder", "radiusTop": 8.0, "radiusBottom": 10.0,
         "height": 14.0, "segments": 24},
        "#64748b",
        parent="rig_camera", position=(0.0, 0.0, -6.0), quaternion=_Q_Y_TO_Z,
    ))
    nodes.append(_mesh(
        "rig_cam_ring",
        {"kind": "torus", "radius": 8.0, "tube": 1.3, "segments": 24},
        "#e2e8f0",
        parent="rig_camera", position=(0.0, 0.0, -13.0), quaternion=_Q_Y_TO_Z,
    ))
    # Live camera frustum at the current pose, banked about the optical axis by
    # `phone_roll_deg` — now the TRUE optical-axis roll (atan2 form in
    # phone_sensor), a small ~stable angle that shows the slight bracket tilt
    # WITHOUT spinning as EL changes (the old code banked by device-+Y
    # elevation, which tracked EL). The base is sized to the camera's ACTUAL
    # frame aspect (portrait phone → tall frustum) to match how frames arrive.
    from camera_stream import stream as _cam_stream
    _frustum_props: dict[str, Any] = {
        "scale": 18.0, "color": "#f97316", "lineWidth": 1.6,
    }
    _aspect = _cam_stream.frame_aspect()
    if _aspect is not None:
        _frustum_props["aspect"] = _aspect
    live_quat = rig.camera_quat
    if model.phone_sensor_online and model.phone_roll_deg is not None:
        live_quat = _quat_mul(live_quat, _roll_about_lens_quat(model.phone_roll_deg))
    nodes.append(_node(
        "live_frustum", "camera_frustum",
        _frustum_props,
        parent="rig_math",
        position=rig.camera_pos, quaternion=live_quat,
    ))

    # ── scan-path preview (planned camera positions) ──────────────────────
    if model.scan_preview:
        nodes.extend(_scan_preview_nodes(model, math_anchor, subj_centre))

    # ── captured frames — show ONE scan at a time. While a saved scan is open
    #    for review (loaded_captures), hide the active recording's frustums so
    #    the two don't overlap ("two scans open"); show only the review. ──────
    if not model.loaded_captures:
        nodes.extend(_capture_nodes(model, math_anchor, subj_centre))

    # ── loaded scan — frustums + photo cards in the absolute world frame ──
    nodes.extend(_loaded_capture_nodes(model, math_anchor))

    # Anything still parent-less goes under `world` as a safety net.
    for n in nodes:
        if n["parent"] is None:
            n["parent"] = "world"
    return [_node("world", "frame", {}), *nodes]


def _arm_nodes(
    model: ModelState,
    rig,
    stem_start_math: tuple[float, float, float] | None = None,
) -> list[Node]:
    """Dynamic arm visualisation — single procedural beam (`cam_stem`)
    from the CAD `OrbitArm:1` tip to the camera optical centre."""
    out: list[Node] = []

    cam_pos = np.asarray(rig.camera_pos, dtype=float)
    if stem_start_math is not None:
        stem_start = np.asarray(stem_start_math, dtype=float)
    else:
        pivot = (
            np.asarray(rig.el_pivot, dtype=float)
            if rig.el_pivot is not None
            else np.zeros(3)
        )
        stem_start = np.asarray(rig.arm_end, dtype=float) + pivot

    stem_vec = cam_pos - stem_start
    stem_len = float(np.linalg.norm(stem_vec))
    if stem_len >= 1.0:
        stem_dir = stem_vec / stem_len
        cam_stem = Cylinder(
            mid=tuple(stem_start + stem_vec * 0.5),
            quat=quat_from_unit_vectors(
                (0.0, 1.0, 0.0),
                (float(stem_dir[0]), float(stem_dir[1]), float(stem_dir[2])),
            ),
            length=stem_len,
        )
        out.append(_cylinder_node(
            "cam_stem", cam_stem, 2.5, "#cbd5e1", parent="rig_arm",
        ))

    return out


#: Distance (units) from a capture frustum's apex to its base along the local
#: optical axis (-Z). Shared by the frustum (`scale`) and the photo card so the
#: card lands on the frustum base.
_CAPTURE_FRUSTUM_SCALE = 14.0


def _frustum_with_card(
    *,
    fid: str,
    cid: str,
    apex_pos: tuple[float, float, float],
    stored_quat,
    sw,
    sh,
    url: str | None,
    parent: str,
    color: str,
) -> list[Node]:
    """Emit a wireframe camera frustum + (optional) textured photo card for one
    capture. `apex_pos` is the apex in the PARENT frame; `stored_quat` is the
    capture's object-frame three.js quaternion (we apply the +90° math→world
    yaw the live camera uses). Aspect from stored_width/height (portrait photo
    → tall frustum). Shared by live captures (platform frame) and loaded-review
    captures (scene root), so both render identically."""
    frustum_quat = _quat_mul(_rot_z(90.0), tuple(stored_quat))
    aspect = (float(sw) / float(sh)) if (sw and sh and sw > 0 and sh > 0) else None
    fprops: dict[str, Any] = {
        "scale": _CAPTURE_FRUSTUM_SCALE, "color": color, "lineWidth": 1.2,
    }
    if aspect is not None:
        fprops["aspect"] = aspect
    out = [_node(
        fid, "camera_frustum", fprops,
        parent=parent, position=apex_pos, quaternion=frustum_quat, pickable=True,
    )]
    if url:
        a = aspect if aspect is not None else 1.4
        card_w, card_h = (12.0, 12.0 / a) if a >= 1.0 else (12.0 * a, 12.0)
        # Slide the card to the frustum base: apex + R(quat)·(0, 0, -scale).
        fwd = _quat_to_matrix(frustum_quat) @ np.array(
            [0.0, 0.0, -_CAPTURE_FRUSTUM_SCALE],
        )
        card_pos = (
            apex_pos[0] + float(fwd[0]),
            apex_pos[1] + float(fwd[1]),
            apex_pos[2] + float(fwd[2]),
        )
        out.append(_node(
            cid, "image_plane",
            {"url": url, "width": card_w, "height": card_h},
            parent=parent, position=card_pos, quaternion=frustum_quat,
            pickable=True,
        ))
    return out


def _loaded_capture_nodes(
    model: ModelState,
    math_anchor: tuple[float, float, float],
) -> list[Node]:
    """Frustums + photo cards for a stored scan opened for review. They sit at
    the scene root (NOT under the rotating platform), but use the SAME math→world
    map as the live camera — `math_anchor + Rz90·p` plus the +90° yaw on the
    quaternion — so orientation matches and thumbnails show. Fixes the old
    wrong-rotation / no-thumbnail review render."""
    out: list[Node] = []
    for i, cap in enumerate(model.loaded_captures):
        xyz = cap.get("camera_xyz_mm") or {}
        px, py, pz = xyz.get("x", 0.0), xyz.get("y", 0.0), xyz.get("z", 0.0)
        # math→world: math_anchor + Rz90·p  (Rz90: (x,y,z) → (−y, x, z)) — the
        # exact map build_scene applies to the live camera position.
        apex_pos = (math_anchor[0] - py, math_anchor[1] + px, math_anchor[2] + pz)
        cap_index = cap.get("index", i)
        out.extend(_frustum_with_card(
            fid=f"loaded_{cap_index}",
            cid=f"loaded_card_{cap_index}",
            apex_pos=apex_pos,
            stored_quat=cap.get("camera_quat") or [0.0, 0.0, 0.0, 1.0],
            sw=cap.get("stored_width"),
            sh=cap.get("stored_height"),
            url=cap.get("thumb_tiny_url") or cap.get("thumb_url"),
            parent="capture_layer",
            color="#fbbf24",
        ))
    return out


def _capture_nodes(
    model: ModelState,
    math_anchor: tuple[float, float, float],
    subj_centre: tuple[float, float, float],
) -> list[Node]:
    """A wireframe frustum + a textured photo card per captured frame.

    Captures live in the platform frame so they swirl with the disc as the
    live azimuth changes. The stored object-frame pose is re-framed into
    platform_spin-local so each frustum lands exactly on the LIVE camera at its
    capture azimuth (see `_to_platform_local`) — no more 90°/offset desync from
    the live camera.

    For each capture we emit:
      * `capture_{index}`      — the wireframe frustum (apex at the camera).
      * `capture_card_{index}` — an `image_plane` textured with the capture's
        thumbnail, placed at the frustum BASE and facing the same way, so the
        photo is visible at the frustum.
    The two share the SAME index so the UI can map either id back to the
    capture.
    """
    out: list[Node] = []
    for i, cap in enumerate(model.captures):
        xyz = cap.get("camera_xyz_mm") or {}
        pos = (xyz.get("x", 0.0), xyz.get("y", 0.0), xyz.get("z", 0.0))
        az0 = float(cap.get("az_deg", 0.0))
        cap_index = cap.get("index", i)
        # Re-frame into platform_spin-local so the frustum lands on the LIVE
        # camera at its capture azimuth (see `_to_platform_local`).
        out.extend(_frustum_with_card(
            fid=f"capture_{cap_index}",
            cid=f"capture_card_{cap_index}",
            apex_pos=_to_platform_local(pos, az0, math_anchor, subj_centre),
            stored_quat=cap.get("camera_quat") or [0.0, 0.0, 0.0, 1.0],
            sw=cap.get("stored_width"),
            sh=cap.get("stored_height"),
            url=cap.get("thumb_tiny_url") or cap.get("thumb_url"),
            parent="platform_spin",
            color="#22d3ee",
        ))
    return out


def _scan_preview_nodes(
    model: ModelState,
    math_anchor: tuple[float, float, float],
    subj_centre: tuple[float, float, float],
) -> list[Node]:
    """A point-cloud preview of the active MotionPlanner path — the discrete
    ring-grid sampled at each planned camera position. The cloud sits under
    the rotating platform so it swirls with the live AZ."""
    mp = model.motion_plan or {}
    d = mp.get("discrete") or {}
    path = plan_scan_path(
        float(d.get("el_start_deg", 0.0)),
        float(d.get("el_max_deg", 60.0)),
        int(d.get("el_steps", 4)),
        float(d.get("az_step_deg", 20.0)),
    )
    if not path:
        return []
    geom = _geom_params(model)
    positions: list[float] = []
    for pt in path:
        pose = camera_pose_at(pt.az_deg, pt.el_deg, geom)
        positions.extend(
            _to_platform_local(
                pose.camera_xyz_mm, pt.az_deg, math_anchor, subj_centre,
            )
        )
    # Node type is `point_cloud` (the UI registry name); a `points` type would
    # render as an empty group and the preview would be invisible.
    return [_node(
        "scan_preview", "point_cloud",
        {"positions": positions, "color": "#a78bfa", "pointSize": 5.0},
        parent="platform_spin",
    )]


def diff(prev: dict[str, Node], nodes: list[Node]) -> dict[str, Any]:
    """Compute a scene_update payload (added / updated / removed)."""
    cur = {n["id"]: n for n in nodes}
    added = [n for nid, n in cur.items() if nid not in prev]
    removed = [nid for nid in prev if nid not in cur]
    updated: list[dict[str, Any]] = []
    for nid, n in cur.items():
        old = prev.get(nid)
        if old is None or old == n:
            continue
        patch: dict[str, Any] = {"id": nid}
        for key in ("parent", "type", "transform", "props", "visible", "pickable"):
            if old.get(key) != n.get(key):
                patch[key] = n[key]
        updated.append(patch)
    return {"added": added, "updated": updated, "removed": removed}
