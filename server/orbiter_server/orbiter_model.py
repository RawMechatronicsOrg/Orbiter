"""Authoritative in-memory model state — the server owns it (Viser-pattern).

`ModelState` holds everything the frontend zustand stores used to hold:
  * live machine state (az/el/motion + firmware encoder zero) — fed by `esp_proxy`,
  * rig build parameters (arm/camera/EL/AZ kinematic offsets) persisted to
    `data/orbiter_state.json`,
  * scan parameters + render preferences.

Mutations go through `update(**patch)`, which persists the config-like fields
and fans the patch out to subscribers (the WebSocket hub).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Callable

from config import settings

log = logging.getLogger("orbiter.model")

# Fields that survive a restart (the former localStorage keys). Live machine
# fields are NOT persisted — they are re-read from the device on startup.
PERSISTED_FIELDS: frozenset[str] = frozenset({
    "arm_radius_mm", "camera_offset_mm", "base_height_mm",
    "camera_tilt_deg", "camera_pan_deg", "turntable_axis",
    "el_kinematic_offset_deg", "az_kinematic_offset_deg",
    "camera_preset", "camera_url", "esp_ip", "esp_autodiscover",
    # camera intrinsics + ChArUco board params (used by calibration.py)
    "camera_fx", "camera_fy", "camera_cx", "camera_cy", "camera_distortion",
    "charuco_squares_x", "charuco_squares_y",
    "charuco_square_length_mm", "charuco_marker_length_mm", "aruco_dict_id",
    # Set True once a ChArUco geometry calibration has been applied; the UI
    # keys a "not calibrated" warning off it.
    "calibrated", "calib_extrinsic", "calib_board_world",
    "motion_plan",
    "show_axes", "scan_preview", "hide_back_facing", "mirror_photo_on_frustum",
})


def _default_motion_plan() -> dict[str, Any]:
    """Defaults match `models.MotionPlan` — kept inline so this module
    doesn't import models.py (avoids a circular import via commands.py)."""
    return {
        "mode": "discrete",
        "discrete": {
            "el_start_deg": 0.0,
            "el_max_deg": 60.0,
            "el_steps": 4,
            "az_step_deg": 20.0,
            "actions": ["photo"],
        },
    }


@dataclass
class ModelState:
    # ── live machine state (from esp_proxy; not persisted) ─────────────────
    esp_online: bool = False
    az: float = 0.0
    el: float = 0.0
    # Commanded pose for an in-flight /move (None when idle). Shown on the UI navball.
    move_target_az: float | None = None
    move_target_el: float | None = None
    motion_state: str = "unknown"      # idle | moving | spinning | error | unknown
    motors_on: bool = False
    # Raw /state.runner snapshot: {id, status, kind, result}.
    runner: dict[str, Any] | None = None
    # Firmware-reported encoder zero offsets (az/el raw degrees).
    encoder_zero: dict[str, float] = field(default_factory=dict)
    # Phone IMU (from IP Webcam /sensors.json). Both pitches are angles of
    # a phone axis above the world horizon — same quaternion, different
    # reference axis:
    #   * phone_pitch_deg   — device-Y tilt, shifted by a ~90° mount offset
    #     into the rig EL frame (see phone_sensor._PITCH_MOUNT_OFFSET_DEG).
    #     For the navball, where it is compared against `el`.
    #   * phone_lens_pitch_deg — device -Z (back-camera optical axis), raw.
    #     Used by `align_el_to_phone` so rig.el == 0 ⇒ lens horizontal.
    # Roll is the bank around the lens axis. All None while the sensor
    # stream is silent or the camera URL isn't set.
    phone_pitch_deg: float | None = None
    phone_lens_pitch_deg: float | None = None
    phone_roll_deg: float | None = None
    phone_sensor_online: bool = False

    # ── rig build params (persisted) ───────────────────────────────────────
    arm_radius_mm: float = 0.0
    camera_offset_mm: float = 80.0
    base_height_mm: float = 45.0
    camera_tilt_deg: float = 0.0
    camera_pan_deg: float = 0.0
    # AZ rotation-axis eccentricity in the platform XY plane (mm) when the
    # turntable axis is not through the object-frame origin. Fed to the
    # calibration hand-eye A matrix and the compute_camera_pose_x 6-DOF path.
    # None = axis at the origin (the common case). Operator-measured; set it
    # by editing orbiter_state.json. (Persisted as a [cx, cy] list in JSON.)
    turntable_axis: tuple[float, float] | None = None
    camera_preset: str = "native"
    camera_url: str = ""
    esp_ip: str = ""
    # When True, the server runs an mDNS browser for `_orbiter._tcp.local.`
    # and auto-fills `esp_ip` whenever the firmware announces itself. Can be
    # toggled from the UI for fully-manual control.
    esp_autodiscover: bool = True
    # ── camera intrinsics (used by calibration.py for ChArUco hand-eye) ──
    # Defaults are a reasonable guess for a 1920×1080 IP Webcam stream at
    # ~50° HFOV. Replace with values from a proper intrinsics calibration
    # if metric accuracy matters.
    camera_fx: float = 1500.0
    camera_fy: float = 1500.0
    camera_cx: float = 960.0
    camera_cy: float = 540.0
    # OpenCV distortion: (k1, k2, p1, p2, k3). Zero = no distortion model.
    camera_distortion: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0])
    # ── ChArUco calibration board geometry ──
    # Defaults: A4-printable 5×7 board, 30 mm squares, 15 mm markers,
    # DICT_4X4_50. Regenerate at https://calib.io/pages/camera-calibration-pattern-generator
    charuco_squares_x: int = 5
    charuco_squares_y: int = 7
    charuco_square_length_mm: float = 30.0
    charuco_marker_length_mm: float = 15.0
    # ArUco dictionary id (cv2.aruco.DICT_* int constant; 0 = DICT_4X4_50, the
    # default printed board). Persisted so the board spec is fully data-driven
    # — together with the square/marker sizes it is the ONLY user input the
    # photo-based calibration needs (intrinsics and axis are solved from photos).
    aruco_dict_id: int = 0
    # True once a ChArUco geometry calibration has been applied (set by
    # calibration.apply_result). Persisted so the UI can show a "not
    # calibrated" warning on a fresh rig that has never been solved.
    calibrated: bool = False
    # Calibration internals persisted for the post-calibration "Test accuracy"
    # check: the hand-eye X (camera↔arm) and the board-in-world reference pose
    # Z_ref, each as {"rvec": [...], "t": [...]}. None until a calibration runs.
    calib_extrinsic: dict[str, Any] | None = None
    calib_board_world: dict[str, Any] | None = None
    # Last "Test accuracy" result — a human string for the UI (runtime only).
    calib_test_msg: str = ""
    # Operator-tunable EL correction (deg) added to `model.el` in every
    # pose/kinematic calculation. Lets the operator dial out a constant EL
    # encoder zero bias without re-running the firmware encoder-zero op
    # mid-session.
    el_kinematic_offset_deg: float = 0.0
    # Same idea for AZ — added to `model.az` everywhere it drives geometry
    # (platform_spin in scene_graph). A miscalibrated AZ zero rotates the
    # whole accumulated cloud by a constant.
    az_kinematic_offset_deg: float = 0.0

    # ── scan plan + progress ───────────────────────────────────────────────
    # The MotionPlanner — the active scan-loop plan (discrete sweep).
    # See `models.MotionPlan`; persisted with the other config-like fields.
    motion_plan: dict[str, Any] = field(default_factory=_default_motion_plan)
    scan_running: bool = False
    scan_progress: int = 0
    scan_total: int = 0
    current_scan_id: str | None = None
    # True while the active scan is an automated scan-loop run; False for a
    # manual shot-by-shot session.
    machine_captured: bool = False
    # ── active scan: unsaved-changes tracking (not persisted) ──────────────
    # The manifest.json is written by the explicit Save command or the
    # debounced autosave; `scan_dirty` flags changes not yet on disk.
    scan_notes: str = ""
    scan_dirty: bool = False
    scan_saved_at: str | None = None
    # Captures of the active/just-finished scan — JSON-able Capture dicts
    # (pose + thumbnail URLs). Rendered as frustums by the scene builder.
    captures: list[dict[str, Any]] = field(default_factory=list)
    # A previously-stored scan opened for review. Loaded captures are in their
    # own absolute world frame — rendered at the scene root, NOT under the
    # rotating platform (the live/loaded split).
    loaded_captures: list[dict[str, Any]] = field(default_factory=list)
    loaded_scan_id: str | None = None
    # Scan summaries for the browser panel — refreshed on demand.
    scans: list[dict[str, Any]] = field(default_factory=list)

    # ── render preferences (persisted) ─────────────────────────────────────
    show_axes: bool = True
    scan_preview: bool = False
    hide_back_facing: bool = False
    mirror_photo_on_frustum: bool = False

    # ── runtime-only (not a dataclass field for serialisation) ─────────────

    def __post_init__(self) -> None:
        # Serialises multi-step command sequences (scan loop).
        self.lock = asyncio.Lock()
        self._subscribers: list[Callable[[dict[str, Any]], None]] = []

    # ── serialisation ──────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Full state as a plain dict (for /debug/model and the WS `model` msg)."""
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def _persist_path(self) -> Path:
        return settings.storage_dir / "orbiter_state.json"

    def save(self) -> None:
        """Atomically write the persisted subset to disk."""
        path = self._persist_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        snapshot = {k: getattr(self, k) for k in PERSISTED_FIELDS}
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
        os.replace(tmp, path)

    @classmethod
    def load(cls) -> "ModelState":
        """Construct from the persisted file if present, else defaults."""
        state = cls()
        path = state._persist_path()
        if not path.exists():
            return state
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("could not read %s: %s — using defaults", path, exc)
            return state
        for key, value in data.items():
            if key in PERSISTED_FIELDS:
                setattr(state, key, value)
        return state

    # ── mutation + change notification ─────────────────────────────────────

    def subscribe(self, callback: Callable[[dict[str, Any]], None]) -> None:
        """Register a callback invoked with the patch dict on every update."""
        self._subscribers.append(callback)

    def update(self, **patch: Any) -> None:
        """Apply a patch, persist if a config field changed, notify subscribers."""
        changed: dict[str, Any] = {}
        for key, value in patch.items():
            if not hasattr(self, key):
                log.warning("update: unknown field %r ignored", key)
                continue
            if getattr(self, key) != value:
                setattr(self, key, value)
                changed[key] = value
        if not changed:
            return
        if any(k in PERSISTED_FIELDS for k in changed):
            try:
                self.save()
            except OSError as exc:
                log.error("failed to persist model state: %s", exc)
        for cb in self._subscribers:
            try:
                cb(changed)
            except Exception:  # noqa: BLE001 — a bad subscriber must not break updates
                log.exception("model subscriber raised")


# Process-wide singleton. esp_proxy, the WS hub and command handlers all
# import and share this instance.
model = ModelState.load()
