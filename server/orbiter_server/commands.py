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
    autosave loop (or the explicit Save) flushes it to disk."""
    notes = str(args.get("notes", ""))
    model.update(scan_notes=notes, scan_dirty=True)
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


async def _cmd_align_el_to_phone(_args: dict[str, Any]) -> dict[str, Any]:
    """Re-zero the EL encoder so the rig's reported elevation tracks the
    camera's actual orientation 1:1 in the operator-natural [0°, 90°]
    range.

    Mapping (this mount has lens ⊥ arm):
      lens_pitch = -90°  ⇔  arm horizontal, lens pointing straight down
                            ⇔  rig.el = 0°
      lens_pitch =   0°  ⇔  arm vertical, lens horizontal
                            ⇔  rig.el = 90°

    So the alignment target is `lens_pitch + 90` and after calibration
    rig.el reads as the operator expects.

    Requires the phone IMU to be online.
    """
    if (
        not model.phone_sensor_online
        or model.phone_lens_pitch_deg is None
    ):
        raise CommandError(
            "phone IMU not online — set the camera URL in endpoints and "
            "make sure the IP Webcam app is exposing accelerometer data"
        )
    # Force a /state refresh — the periodic poll is dormant while the WS
    # pose stream is fresh, so model.encoder_zero may be empty since
    # startup. fetch_state() pulls /state from the firmware AND updates
    # the model — after this returns, the model is the source of truth.
    await esp.fetch_state()
    cur_el = float(model.el)
    lens_pitch = float(model.phone_lens_pitch_deg)
    target_el = lens_pitch + 90.0
    prev_zero = float(model.encoder_zero.get("el_zero_raw_deg", 0.0))
    # displayed = raw - zero. We want displayed_new = target_el when raw
    # is unchanged, so zero_new = zero_old + (displayed_old - target_el).
    new_zero = prev_zero + (cur_el - target_el)
    result = await esp.calibrate(
        axis="el", mode="explicit", el_raw_deg=new_zero,
    )
    log.info(
        "align_el_to_phone: el=%.2f lens=%.2f target=%.2f → el_zero %.3f → %.3f",
        cur_el, lens_pitch, target_el, prev_zero, new_zero,
    )
    return {
        "previous_el_zero_raw_deg": prev_zero,
        "new_el_zero_raw_deg": new_zero,
        "delta_applied_deg": cur_el - target_el,
        "phone_lens_pitch_deg": lens_pitch,
        "target_el_deg": target_el,
        "rig_el_before": cur_el,
        **result,
    }


async def _cmd_calibrate_encoder(args: dict[str, Any]) -> dict[str, Any]:
    """Set the firmware encoder zero.

    Tells the ESP firmware to interpret the current encoder reading (or a
    supplied raw degree value) as zero for the named axis. The same primitive
    the operator hits after physically aligning the rig.
    """
    return await esp.calibrate(
        axis=str(args.get("axis", "both")),
        mode=str(args.get("mode", "current")),
        az_raw_deg=args.get("az_raw_deg"),
        el_raw_deg=args.get("el_raw_deg"),
    )


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
    "align_el_to_phone": _cmd_align_el_to_phone,
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
    "set_machine_config": _cmd_set_machine_config,
    "set_render_pref": _cmd_set_render_pref,
    "set_camera_url": _cmd_set_camera_url,
    "set_board_params": _cmd_set_board_params,
    "set_endpoints": _cmd_set_endpoints,
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
