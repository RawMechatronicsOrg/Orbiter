"""Named-command dispatch — the single channel for mutating server state.

Every UI action becomes a command here. Handlers validate args, then call
`esp_proxy` (hardware) or mutate `orbiter_model` (state). Model mutations
auto-broadcast scene/model updates via the WS hub, so handlers do not push
anything themselves.

Out of scope for v0.1: the parent project's photogrammetry job orchestration.
The surviving command surface drives the two-axis turntable, captures photos
(manual or motion-planned), and runs ChArUco hand-eye geometry calibration.
"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable

import scan_task
from esp_proxy import esp
from orbiter_model import model

log = logging.getLogger("orbiter.cmd")


class CommandError(RuntimeError):
    """A command was unknown or its arguments were invalid."""


# ── geometry / render-preference commands ──────────────────────────────────

_GEOMETRY_KEYS = (
    "arm_radius_mm", "camera_offset_mm", "base_height_mm",
    "camera_tilt_deg", "camera_pan_deg",
    "el_kinematic_offset_deg", "az_kinematic_offset_deg",
)
_RENDER_KEYS = ("show_axes", "scan_preview", "hide_back_facing",
                "mirror_photo_on_frustum")


async def _cmd_set_machine_config(args: dict[str, Any]) -> dict[str, Any]:
    """Update the rig build parameters (arm/camera/EL/AZ kinematic offsets)."""
    patch = {k: float(args[k]) for k in _GEOMETRY_KEYS if k in args}
    if not patch:
        raise CommandError("set_machine_config: no geometry fields given")
    model.update(**patch)
    return patch


async def _cmd_set_render_pref(args: dict[str, Any]) -> dict[str, Any]:
    patch = {k: bool(args[k]) for k in _RENDER_KEYS if k in args}
    if not patch:
        raise CommandError("set_render_pref: no render fields given")
    model.update(**patch)
    return patch


async def _cmd_set_motion_plan(args: dict[str, Any]) -> dict[str, Any]:
    """Replace the MotionPlanner (the active scan-loop plan).

    Accepts either `{motion_plan: {...}}` or a bare plan dict. The payload is
    validated through `models.MotionPlan` (defaults fill missing fields, so a
    partial plan is fine — e.g. just the discrete sub-plan); the full
    normalised plan is stored on `model.motion_plan` and broadcast over WS.
    """
    from models import MotionPlan

    payload = args.get("motion_plan") if isinstance(args.get("motion_plan"), dict) else args
    try:
        plan = MotionPlan.model_validate(payload)
    except Exception as exc:
        raise CommandError(f"set_motion_plan: invalid plan: {exc}") from exc
    serialised = plan.model_dump(mode="json")
    model.update(motion_plan=serialised)
    return serialised


# Per-axis closed-loop top step rate (Hz) bounds for the operator speed knobs.
# Tighter than the firmware sanity clamp (CL_HZ_MIN..CL_HZ_OVERRIDE_MAX) and
# matched to the UI slider ranges. AZ is the light axis (fast); EL carries the
# whole arm, so it's capped lower to keep calibration/scan sweeps from
# over-shooting (the reason the EL default was historically conservative).
_MOVE_HZ_BOUNDS: dict[str, tuple[int, int]] = {
    "az": (100, 4000),
    "el": (50, 1000),
}


async def _cmd_set_move_speed(args: dict[str, Any]) -> dict[str, Any]:
    """Set the per-axis closed-loop top step rate — the Machine-config speed
    knobs. Payload (either or both key): ``{az_hz_max: int, el_hz_max: int}``
    in Hz. Each value is clamped to its per-axis safe band and persisted on the
    model; the next /move carries it to the firmware (see esp_proxy.move)."""
    patch: dict[str, Any] = {}
    try:
        if "az_hz_max" in args:
            lo, hi = _MOVE_HZ_BOUNDS["az"]
            patch["move_hz_max_az"] = max(lo, min(hi, int(args["az_hz_max"])))
        if "el_hz_max" in args:
            lo, hi = _MOVE_HZ_BOUNDS["el"]
            patch["move_hz_max_el"] = max(lo, min(hi, int(args["el_hz_max"])))
    except (TypeError, ValueError) as exc:
        raise CommandError(f"set_move_speed: bad value: {exc}") from exc
    if not patch:
        raise CommandError("set_move_speed: provide az_hz_max and/or el_hz_max")
    model.update(**patch)
    return patch


async def _cmd_set_camera_url(args: dict[str, Any]) -> dict[str, Any]:
    """Update the live still-image URL (IP Webcam etc.) at runtime."""
    url = args.get("url")
    if url is None:
        raise CommandError("set_camera_url: missing 'url'")
    url = str(url).strip()
    model.update(camera_url=url)
    return {"camera_url": url}


#: ChArUco board geometry fields the UI can set, with the coercion to apply.
#: These are already persisted model fields (see PERSISTED_FIELDS) consumed by
#: calibration.py's `_board_spec_from_model`.
_BOARD_PARAM_COERCE: dict[str, Callable[[Any], Any]] = {
    "charuco_squares_x": lambda v: int(v),
    "charuco_squares_y": lambda v: int(v),
    "charuco_square_length_mm": lambda v: float(v),
    "charuco_marker_length_mm": lambda v: float(v),
    "aruco_dict_id": lambda v: int(v),
}


async def _cmd_set_board_params(args: dict[str, Any]) -> dict[str, Any]:
    """Update the ChArUco calibration board spec used by calibration.py.

    Payload (all optional — only keys present are applied):
      `charuco_squares_x`, `charuco_squares_y` (int),
      `charuco_square_length_mm`, `charuco_marker_length_mm` (float),
      `aruco_dict_id` (int — a cv2.aruco.DICT_* constant).
    Types are coerced; a value that won't coerce raises CommandError.
    """
    patch: dict[str, Any] = {}
    for key, coerce in _BOARD_PARAM_COERCE.items():
        if key in args:
            try:
                patch[key] = coerce(args[key])
            except (TypeError, ValueError) as exc:
                raise CommandError(f"set_board_params: bad {key!r}: {exc}") from exc
    if patch:
        model.update(**patch)
    return {"ok": True}


# ── binocular pair (camserver) ─────────────────────────────────────────────
#
# ORIENTATION CONTRACT — the whole point of this config, so read it before
# changing anything here.
#
# An eye's orientation is **flip first, then rotate**: `flip_h` / `flip_v` act
# in the sensor's own frame, and `quarter_turns_cw` is applied to the already
# flipped image. The Stereo tab renders exactly that order with
#
#     transform: rotate(<90*n>deg) scaleX(<±1>) scaleY(<±1>)
#
# (CSS composes right-to-left, so that reads flip-then-rotate). Any future
# server-side path — folding these into the `cv2.remap` map alongside
# undistort and stereo rectification — MUST use the same order. If the preview
# and the CV path disagree, the operator lines the rig up against a picture
# that does not match what the solver sees, and it surfaces later as an
# inexplicable calibration error rather than as an obvious mirror.

#: Per-eye orientation fields and their coercions.
_EYE_COERCE: dict[str, Callable[[Any], Any]] = {
    "camera_id": lambda v: str(v).strip(),
    "quarter_turns_cw": lambda v: int(v) % 4,
    "flip_h": lambda v: bool(v),
    "flip_v": lambda v: bool(v),
    "intrinsics": lambda v: _coerce_intrinsics(v),
    "readout": lambda v: _coerce_readout(v),
}


def _coerce_readout(v: Any) -> dict[str, Any] | None:
    """One eye's rolling-shutter readout time, or None to clear it.

    `seconds` is the time the sensor takes to read one whole frame, signed:
    positive when row 0 is read first. It belongs to one sensor mode, so
    `width`/`height` travel with it like they do on the intrinsics. The
    native app measures it from the board in motion and uses it to take
    every scanned point into the board's frame at its own row's instant.
    """
    if v is None:
        return None
    if not isinstance(v, dict):
        raise CommandError("readout must be an object or null")
    try:
        out: dict[str, Any] = {
            "seconds": float(v["seconds"]),
            "width": int(v["width"]), "height": int(v["height"]),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise CommandError(f"readout needs seconds, width, height: {exc}") from exc
    if not (0.0 < abs(out["seconds"]) < 0.1):
        raise CommandError("readout: seconds must be a non-zero frame readout time")
    if out["width"] <= 0 or out["height"] <= 0:
        raise CommandError("readout: width and height must be positive")
    for key in ("sigma_s", "skew_px", "rms_px"):
        if key in v and v[key] is not None:
            out[key] = float(v[key])
    if "views" in v and v["views"] is not None:
        out["views"] = int(v["views"])
    return out


def _coerce_intrinsics(v: Any) -> dict[str, Any] | None:
    """One eye's solved intrinsics, or None to clear them.

    `width`/`height` are REQUIRED and are not decoration: a camera matrix is
    only valid for the frame size it was solved at. camserver can be
    reconfigured to another resolution between runs, and applying 1280x720
    intrinsics to a 1920x1080 frame yields poses that look plausible and are
    wrong by the ratio of the two. Consumers compare these against the live
    frame and refuse a mismatch.
    """
    if v is None:
        return None
    if not isinstance(v, dict):
        raise CommandError("intrinsics must be an object or null")
    try:
        dist = [float(x) for x in v.get("dist", [])]
        out: dict[str, Any] = {
            "fx": float(v["fx"]), "fy": float(v["fy"]),
            "cx": float(v["cx"]), "cy": float(v["cy"]),
            "dist": dist,
            "width": int(v["width"]), "height": int(v["height"]),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise CommandError(
            f"intrinsics needs numeric fx, fy, cx, cy, width, height: {exc}"
        ) from exc
    if out["width"] <= 0 or out["height"] <= 0:
        raise CommandError("intrinsics width/height must be positive")
    if out["fx"] <= 0 or out["fy"] <= 0:
        raise CommandError("intrinsics fx/fy must be positive")
    # Provenance — how good the solve was and how many views it used. Carried
    # so a later reader can judge the numbers instead of trusting them blindly.
    for key, cast in (("rms_px", float), ("views", int), ("solved_at", str)):
        if key in v and v[key] is not None:
            try:
                out[key] = cast(v[key])
            except (TypeError, ValueError) as exc:
                raise CommandError(f"intrinsics: bad {key!r}: {exc}") from exc
    return out

#: Rig-level fields (everything that is not one of the two eyes).
_RIG_COERCE: dict[str, Callable[[Any], Any]] = {
    "host": lambda v: str(v).strip().rstrip("/"),
    "token": lambda v: str(v).strip(),
    "baseline_mm": lambda v: float(v),
    "extrinsics": lambda v: _coerce_extrinsics(v),
    "laser_plane": lambda v: _coerce_laser_plane(v),
}


def _coerce_laser_plane(v: Any) -> dict[str, Any] | None:
    """The laser sheet as `n . X = d` in the left camera's frame, mm.

    Meaningful only while the cameras and the laser stay rigidly coupled: the
    plane is fixed in the cameras' frame, not in the world. `width`/`height`
    travel with it for the same reason as on the intrinsics — the frame it is
    expressed in is defined by a camera matrix valid at one resolution.
    """
    if v is None:
        return None
    if not isinstance(v, dict):
        raise CommandError("laser_plane must be an object or null")
    try:
        n = [float(x) for x in v["n"]]
        out: dict[str, Any] = {
            "n": n, "d": float(v["d"]),
            "width": int(v["width"]), "height": int(v["height"]),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise CommandError(f"laser_plane needs n (3), d, width, height: {exc}") from exc
    if len(n) != 3:
        raise CommandError("laser_plane: n must have 3 components")
    if sum(x * x for x in n) < 1e-12:
        raise CommandError("laser_plane: n must not be the zero vector")
    if out["width"] <= 0 or out["height"] <= 0:
        raise CommandError("laser_plane width/height must be positive")
    for key, cast in (("rms_mm", float), ("points", int), ("frames", int)):
        if key in v and v[key] is not None:
            try:
                out[key] = cast(v[key])
            except (TypeError, ValueError) as exc:
                raise CommandError(f"laser_plane: bad {key!r}: {exc}") from exc
    return out


def _coerce_extrinsics(v: Any) -> dict[str, Any] | None:
    """The right eye's pose in the left eye's frame, or None to clear.

    `R` is 3x3 and `T` is in MILLIMETRES. `width`/`height` are required for the
    same reason as on intrinsics: this geometry was solved alongside a specific
    pair of camera matrices, which are only valid at one frame size, so it must
    be refused rather than reused when the resolution changes.
    """
    if v is None:
        return None
    if not isinstance(v, dict):
        raise CommandError("extrinsics must be an object or null")
    try:
        R = [[float(x) for x in row] for row in v["R"]]
        T = [float(x) for x in v["T"]]
        out: dict[str, Any] = {
            "R": R, "T": T,
            "width": int(v["width"]), "height": int(v["height"]),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise CommandError(f"extrinsics needs R (3x3), T (3), width, height: {exc}") from exc
    if len(R) != 3 or any(len(row) != 3 for row in R) or len(T) != 3:
        raise CommandError("extrinsics: R must be 3x3 and T must have 3 elements")
    if out["width"] <= 0 or out["height"] <= 0:
        raise CommandError("extrinsics width/height must be positive")
    for key, cast in (("rms_px", float), ("views", int), ("baseline_mm", float)):
        if key in v and v[key] is not None:
            try:
                out[key] = cast(v[key])
            except (TypeError, ValueError) as exc:
                raise CommandError(f"extrinsics: bad {key!r}: {exc}") from exc
    return out


def _merge_eye(current: Any, patch: Any, side: str) -> dict[str, Any]:
    """Apply the present keys of `patch` onto one eye's current config."""
    merged = dict(current) if isinstance(current, dict) else {}
    if not isinstance(patch, dict):
        raise CommandError(f"set_stereo_rig: {side!r} must be an object")
    for key, coerce in _EYE_COERCE.items():
        if key in patch:
            try:
                merged[key] = coerce(patch[key])
            except (TypeError, ValueError) as exc:
                raise CommandError(
                    f"set_stereo_rig: bad {side}.{key!r}: {exc}"
                ) from exc
    return merged


async def _cmd_set_stereo_rig(args: dict[str, Any]) -> dict[str, Any]:
    """Update the binocular pair config — the per-run baseline.

    Payload (all optional — only keys present are applied):
      `host`, `token`, `baseline_mm`, `extrinsics`, `laser_plane`,
      `left` / `right`: `{camera_id, quarter_turns_cw, flip_h, flip_v,
      intrinsics, readout}`, where `intrinsics` is `{fx, fy, cx, cy, dist,
      width, height}` (plus optional `rms_px`, `views`, `solved_at`) or null
      to clear, and `readout` is `{seconds, width, height}` (plus optional
      `sigma_s`, `skew_px`, `rms_px`, `views`) or null to clear.

    Which upstream camera is the left eye cannot be derived: camserver's ids
    follow /dev/videoN enumeration and say nothing about physical placement.
    The operator asserts it here and it persists across restarts.
    """
    rig = dict(model.stereo_rig or {})

    for key, coerce in _RIG_COERCE.items():
        if key in args:
            try:
                rig[key] = coerce(args[key])
            except (TypeError, ValueError) as exc:
                raise CommandError(f"set_stereo_rig: bad {key!r}: {exc}") from exc

    if "baseline_mm" in args and rig["baseline_mm"] <= 0:
        raise CommandError("set_stereo_rig: baseline_mm must be positive")

    for side in ("left", "right"):
        if side in args:
            rig[side] = _merge_eye(rig.get(side), args[side], side)

    # Both eyes pointing at one camera is silently wrong rather than visibly
    # broken: the tab shows two live panes that happen to be identical, and
    # every downstream stereo result is nonsense. Refuse it.
    left_id = str((rig.get("left") or {}).get("camera_id") or "")
    right_id = str((rig.get("right") or {}).get("camera_id") or "")
    if left_id and left_id == right_id:
        raise CommandError(
            f"set_stereo_rig: left and right are both {left_id!r} — "
            "the two eyes must be different cameras"
        )

    model.update(stereo_rig=rig)
    return rig


async def _cmd_calibrate_geometry(args: dict[str, Any]) -> dict[str, Any]:
    """Run a ChArUco hand-eye sweep and return derived rig geometry.

    Body fields (all optional):
      * `apply` (bool, default `True`) — write the derived geometry into
        `model` after solving. Set to `False` for a dry run.
      * `preset` (`"fast"` | `"normal"` | `"full"`, default `"fast"`) — accuracy
        preset selecting the sweep pose set (see `calibration.poses_for_preset`).
        Higher accuracy = denser ring + more elevations = longer sweep.

    Errors propagate as `CommandError` (UI surfaces them).
    """
    import calibration

    if not model.encoder_zero_initialized:
        raise CommandError(
            "encoder zero not initialized — the sweep would move to absolute "
            "poses while the encoders may read arbitrary angles. Wait for the "
            "first-boot auto-zero (needs ESP + phone IMU online) or zero the "
            "encoders manually in Machine config first."
        )
    do_apply = bool(args.get("apply", True))
    preset = str(args.get("preset", "fast"))
    poses = calibration.poses_for_preset(preset)
    try:
        result = await calibration.run_calibration(poses=poses)
    except RuntimeError as exc:
        raise CommandError(f"calibrate_geometry: {exc}") from exc
    if do_apply:
        calibration.apply_result(result)
    out = calibration.result_dict(result)
    out["applied"] = do_apply
    out["preset"] = preset
    return out


async def _cmd_set_endpoints(args: dict[str, Any]) -> dict[str, Any]:
    """Update endpoint-class settings in one go: `esp_ip`, `camera_url`,
    `esp_autodiscover`. Missing fields are left untouched.

    When `esp_autodiscover` changes, the mDNS browser is started or stopped
    to match — the user gets the new mode immediately without a restart.
    """
    import discovery

    patch: dict[str, Any] = {}
    if "esp_ip" in args:
        patch["esp_ip"] = str(args["esp_ip"]).strip()
    if "camera_url" in args:
        patch["camera_url"] = str(args["camera_url"]).strip()
    if "esp_autodiscover" in args:
        patch["esp_autodiscover"] = bool(args["esp_autodiscover"])
    if not patch:
        raise CommandError("set_endpoints: no fields given")
    model.update(**patch)
    if "esp_autodiscover" in patch:
        await discovery.sync_to_model()
    return patch


async def _cmd_set_active_session(args: dict[str, Any]) -> dict[str, Any]:
    """Switch which previously-saved scan is loaded as the active session.

    Accepts `{scan_id: "..."}` or `{scan_id: null}` to clear. When a scan_id
    is given, its manifest is read and the captures are pushed onto
    `model.loaded_captures` for the 3D viewer; the live `captures` are
    preserved (the active recording is separate from the loaded review).
    """
    import asyncio

    import storage

    scan_id = args.get("scan_id")
    if scan_id is None:
        model.update(loaded_captures=[], loaded_scan_id=None)
        return {"loaded_scan_id": None}

    scan_id = str(scan_id)
    manifest = await asyncio.to_thread(storage.read_manifest, scan_id)
    captures = [c.model_dump(mode="json") for c in manifest.captures]
    model.update(loaded_captures=captures, loaded_scan_id=scan_id)
    return {"loaded_scan_id": scan_id, "captures": len(captures)}


async def _cmd_save_scan_notes(args: dict[str, Any]) -> dict[str, Any]:
    """Patch the active scan's notes field. Marks the scan dirty so the
    autosave loop (or the explicit Save) flushes it to disk.
    ``scan_notes_edited`` makes the manifest write take notes from the model
    — otherwise the on-disk value (REST catalog edits) is kept."""
    notes = str(args.get("notes", ""))
    model.update(scan_notes=notes, scan_notes_edited=True, scan_dirty=True)
    return {"scan_notes": notes}


# ── hardware commands (via the ESP proxy) ──────────────────────────────────

async def _cmd_move(args: dict[str, Any]) -> dict[str, Any]:
    az = args.get("az")
    el = args.get("el")
    return await esp.move_and_await(
        azimuth_deg=None if az is None else float(az),
        elevation_deg=None if el is None else float(el),
    )


async def _cmd_jog(args: dict[str, Any]) -> dict[str, Any]:
    """Bump one axis by a small delta. `{axis: 'az'|'el', delta_deg: float}`."""
    axis = str(args.get("axis", "")).lower()
    if axis not in ("az", "el"):
        raise CommandError("jog: axis must be 'az' or 'el'")
    delta = float(args.get("delta_deg", 0.0))
    if axis == "az":
        return await esp.move_and_await(azimuth_deg=model.az + delta)
    return await esp.move_and_await(elevation_deg=model.el + delta)


async def _cmd_motors(args: dict[str, Any]) -> dict[str, Any]:
    """Enable/disable the stepper drivers — `{enabled: bool}`. UI's
    Motors ON/OFF toggle hits this."""
    if "enabled" not in args:
        raise CommandError("motors: 'enabled' (bool) required")
    return await esp.motors(bool(args["enabled"]))


async def _cmd_calibrate_encoder(args: dict[str, Any]) -> dict[str, Any]:
    """Set the firmware encoder zero.

    Tells the ESP firmware to interpret the current encoder reading (or a
    supplied raw degree value) as zero for the named axis. The same primitive
    the operator hits after physically aligning the rig — so it also counts
    as the rig's first-boot zero initialization.
    """
    import encoder_init

    result = await esp.calibrate(
        axis=str(args.get("axis", "both")),
        mode=str(args.get("mode", "current")),
        az_raw_deg=args.get("az_raw_deg"),
        el_raw_deg=args.get("el_raw_deg"),
    )
    encoder_init.mark_initialized("manual encoder zero")
    return result


async def _cmd_reboot_firmware(_args: dict[str, Any]) -> dict[str, Any]:
    """Restart the ESP32 firmware. The device drops offline for a few seconds;
    the poll loop flips `esp_online` back when it returns."""
    return await esp.reboot()


# ── scan commands (async server-side tasks) ────────────────────────────────

async def _cmd_take_shot(_args: dict[str, Any]) -> dict[str, Any]:
    return await scan_task.take_shot()


async def _cmd_delete_capture(args: dict[str, Any]) -> dict[str, Any]:
    """Delete a single capture from the ACTIVE scan session.

    Payload: ``{capture_id: str}`` (preferred) or ``{index: int}`` (fallback,
    matched against the capture's ``index`` field). The mirror of
    ``take_shot`` — it unwinds one capture everywhere take_shot put one:

      1. drop the entry from ``model.captures`` (the live list the scene
         builder turns into frustums + photo cards),
      2. delete the pool files (original + all thumb tiers + meta) via
         ``storage.delete_capture_media``,
      3. rewrite the active manifest from the now-shrunk ``model.captures``
         so an explicit Save / the autosave can't resurrect it — and so the
         shot is gone from disk even before the next save fires,
      4. mark the scan dirty.

    Mutating ``model.captures`` auto-broadcasts a ``model_patch`` AND a scene
    diff (ws_hub rebuilds + diffs build_scene), so the removed ``capture_{i}``
    / ``capture_card_{i}`` nodes vanish from the 3D scene and the UI list with
    no extra push from here.

    Held under ``scan_task.scan_lock()`` — the same lock ``take_shot`` and the
    scan loop hold — so a concurrent capture's read-modify-write of
    ``model.captures`` can't interleave and lose this edit. Refused outright
    while an automated scan loop is running (it owns ``model.captures`` and is
    actively appending; deleting mid-sweep is incoherent), mirroring how
    ``new_scan`` / ``recreate_scan`` refuse.

    Unknown capture_id/index → CommandError, matching how the other handlers
    signal bad args (the UI surfaces it).
    """
    import asyncio

    import storage

    capture_id = args.get("capture_id")
    index = args.get("index")
    if capture_id is None and index is None:
        raise CommandError("delete_capture: 'capture_id' (or 'index') required")
    if model.scan_running:
        raise CommandError("cannot delete a capture while a scan is running")

    async with scan_task.scan_lock():
        # Re-check under the lock: start_scan holds this same lock for the
        # whole sweep, so if a scan began between the guard above and here we'd
        # have blocked until it finished — by which point its captures replaced
        # the list we were asked about. Refuse cleanly rather than act on it.
        if model.scan_running:
            raise CommandError("cannot delete a capture while a scan is running")
        captures = list(model.captures)

        def _matches(c: dict[str, Any]) -> bool:
            if capture_id is not None and c.get("capture_id") == capture_id:
                return True
            # Fall back to index only when no capture_id was supplied, so a
            # mismatched (id, index) pair never deletes the wrong photo.
            if capture_id is None and index is not None and c.get("index") == index:
                return True
            return False

        victim = next((c for c in captures if _matches(c)), None)
        if victim is None:
            raise CommandError(
                f"delete_capture: no capture matching "
                f"capture_id={capture_id!r} index={index!r}"
            )

        cap_id = victim.get("capture_id")
        remaining = [c for c in captures if c is not victim]

        # Mutate the live list FIRST so the broadcast (frustum/card removal)
        # reflects the deletion even if the disk ops below partially fail.
        model.update(captures=remaining)

        # Delete the pool files. Tolerant of already-missing files; a stuck
        # file logs but does not fail the command — the capture is already
        # gone from the model + manifest, which is what the operator sees.
        if cap_id:
            try:
                removed = await asyncio.to_thread(
                    storage.delete_capture_media, str(cap_id)
                )
                if not removed:
                    log.info("delete_capture: no pool files for %s", cap_id)
            except OSError as exc:
                log.warning("delete_capture: pool cleanup for %s failed: %s",
                            cap_id, exc)

        # Rewrite the active manifest from the shrunk model.captures so the
        # deletion is durable immediately (explicit Save / autosave already
        # rebuild captures from model.captures — this just front-runs them).
        # Best-effort: if there's no active scan or its manifest isn't on disk
        # yet, marking the scan dirty is enough — the next save writes the
        # already-correct list.
        if model.current_scan_id:
            try:
                await scan_task.persist_active_manifest_now()
            except FileNotFoundError:
                # Manifest not written yet (brand-new manual session) — the
                # dirty flag below ensures the next save carries the deletion.
                pass
            except Exception:  # noqa: BLE001 — never let a write error crash the cmd
                log.exception("delete_capture: manifest rewrite failed")

        model.update(scan_dirty=True)

    return {"ok": True, "deleted": cap_id, "remaining": len(remaining)}


async def _cmd_save_scan(_args: dict[str, Any]) -> dict[str, Any]:
    """Write the active scan's manifest to disk now (the explicit Save)."""
    return await scan_task.save_active_scan()


async def _cmd_new_scan(args: dict[str, Any]) -> dict[str, Any]:
    """Start a fresh active scan (the 'New' button)."""
    if model.scan_running:
        raise CommandError("cannot start a new scan while a scan is running")
    return await scan_task.new_active_scan(
        machine_captured=bool(args.get("machine_captured", False)),
    )


async def _cmd_recreate_scan(_args: dict[str, Any]) -> dict[str, Any]:
    """Save the current scan, then start a fresh one ('Recreate & Save')."""
    if model.scan_running:
        raise CommandError("cannot recreate while a scan is running")
    return await scan_task.recreate_active_scan()


async def _cmd_start_scan(args: dict[str, Any]) -> dict[str, Any]:
    """Run the MotionPlanner sweep ('Start scan').

    Optionally applies the ``motion_plan`` carried in the payload FIRST (so the
    UI can configure + start in a single click without a set/start race — both
    happen in this one ordered handler), then launches the discrete loop.
    """
    if model.scan_running:
        raise CommandError("a scan is already running")
    mp = args.get("motion_plan")
    if mp is not None:
        from models import MotionPlan
        try:
            model.update(
                motion_plan=MotionPlan.model_validate(mp).model_dump(mode="json"),
            )
        except Exception as exc:  # noqa: BLE001
            raise CommandError(f"invalid motion_plan: {exc}") from exc
    try:
        return await scan_task.start_scan()
    except RuntimeError as exc:  # surface the human-readable reason to the UI
        raise CommandError(str(exc)) from exc


async def _cmd_stop_scan(_args: dict[str, Any]) -> dict[str, Any]:
    """Request the running scan to abort at the next iteration ('Stop scan')."""
    return scan_task.stop_scan()


async def _cmd_test_calibration_accuracy(_args: dict[str, Any]) -> dict[str, Any]:
    """Capture at the CURRENT pose, detect the board, and compare the optical
    board-in-world pose to the calibrated reference — reports the delta
    (rotation° / translation mm). Result also lands in `model.calib_test_msg`
    and the log panel. Needs a prior calibration (for the reference)."""
    import calibration
    return await calibration.test_accuracy()


# ── dispatch table ─────────────────────────────────────────────────────────

_COMMANDS: dict[str, Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]] = {
    # hardware
    "move": _cmd_move,
    "jog": _cmd_jog,
    "motors": _cmd_motors,
    "calibrate_encoder": _cmd_calibrate_encoder,
    "reboot_firmware": _cmd_reboot_firmware,
    # scan-loop / capture
    "take_shot": _cmd_take_shot,
    "delete_capture": _cmd_delete_capture,
    "save_scan": _cmd_save_scan,
    "new_scan": _cmd_new_scan,
    "recreate_scan": _cmd_recreate_scan,
    "start_scan": _cmd_start_scan,
    "stop_scan": _cmd_stop_scan,
    "save_scan_notes": _cmd_save_scan_notes,
    "set_active_session": _cmd_set_active_session,
    # config / render
    "set_motion_plan": _cmd_set_motion_plan,
    "set_move_speed": _cmd_set_move_speed,
    "set_machine_config": _cmd_set_machine_config,
    "set_render_pref": _cmd_set_render_pref,
    "set_camera_url": _cmd_set_camera_url,
    "set_board_params": _cmd_set_board_params,
    "set_endpoints": _cmd_set_endpoints,
    "set_stereo_rig": _cmd_set_stereo_rig,
    "calibrate_geometry": _cmd_calibrate_geometry,
    "test_calibration_accuracy": _cmd_test_calibration_accuracy,
}

# UI commands.ts still uses the older names from the parent project. Map
# them to the current handlers so the UI stops spamming `unknown command`
# every click — until commands.ts gets a contract-aligned rewrite.
_COMMANDS.update({
    "calibrate":  _COMMANDS["calibrate_encoder"],
    "reboot_esp": _COMMANDS["reboot_firmware"],
    "set_geometry": _COMMANDS["set_machine_config"],
})


def known_commands() -> list[str]:
    return sorted(_COMMANDS)


async def dispatch(name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Run a named command. Raises CommandError for an unknown name; handler
    exceptions (bad args, ESP failures) propagate to the caller."""
    handler = _COMMANDS.get(name)
    if handler is None:
        raise CommandError(f"unknown command: {name!r}")
    log.info("command %s %s", name, args)
    return await handler(args)
