"""Server-side scan orchestration — the MotionPlanner loop.

The active scan-loop plan lives on `model.motion_plan` (see `models.MotionPlan`).
`start_scan` reads it and runs the discrete sweep: elevation rings × azimuth
step. At each ring point the loop runs every checked `action` in order
(`photo`, `photo_flash`). One stored capture per action.

Photo bytes always hit the `captures/` pool immediately; only the lightweight
manifest.json gets the dirty / save / autosave treatment (UI shows it as
"unsaved changes").
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

import camera_io
import storage
from geom.pose import GeomParams, camera_pose_at
from geom.scan_path import plan_scan_path
from geom.transforms import quat_mul, roll_about_lens_quat
from models import (
    Capture,
    CaptureMeta,
    CreateScanReq,
    DiscretePlan,
    EncoderZero,
    Geometry,
    Manifest,
    MotionPlan,
    ScanParams,
    ScanPathPoint,
    ScanSummary,
    Vec3,
)
from orbiter_model import model

log = logging.getLogger("orbiter.scan")

_scan_lock = asyncio.Lock()
_abort = asyncio.Event()
_scan_task: asyncio.Task[None] | None = None


def scan_lock() -> asyncio.Lock:
    """The lock serialising scan-loop / capture operations (start_scan,
    take_shot). Exposed so other modules that mutate the active scan's
    captures — e.g. the delete_capture command — can serialise against an
    in-flight capture instead of racing its read-modify-write of
    ``model.captures``."""
    return _scan_lock

#: Settle delay after toggling the torch before exposing — gives the phone
#: camera's auto-exposure a beat so flash frames are not stale.
_ACTION_SETTLE_S = 0.2


def _geom_params() -> GeomParams:
    return GeomParams(
        extrinsic=None,
        arm_radius=max(model.arm_radius_mm, 1.0),
        camera_offset=model.camera_offset_mm,
        # Tilt zeroed — see scene_graph._geom_params for the rationale.
        camera_tilt=0.0,
        camera_pan=model.camera_pan_deg,
        turntable_axis=getattr(model, "turntable_axis", None),
    )


def _vec3(t: tuple[float, float, float]) -> Vec3:
    return Vec3(x=t[0], y=t[1], z=t[2])


# ── single capture ─────────────────────────────────────────────────────────


async def _capture_at(
    scan_id: str,
    index: int,
    planned_az: float | None,
    planned_el: float | None,
    action: str | None = None,
) -> dict[str, Any]:
    """Take one photo at the current rig pose and store it. Returns the
    JSON-able Capture dict."""
    geom = _geom_params()
    az_act, el_act = model.az, model.el
    pose = camera_pose_at(az_act, el_act, geom)
    raw = await camera_io.fetch_photo(el_deg=el_act)

    # Bank the stored camera orientation by the live bracket roll — the slight
    # ~5° tilt of the non-rigid mount about the optical axis — so the capture
    # reflects the camera's TRUE attitude. Uses the SAME helper as the live
    # frustum bank, so a captured frustum lands on the live camera exactly
    # (position + orientation). Honest for SfM too. Skipped if the IMU is offline.
    cam_quat = pose.camera_quat
    roll = model.phone_roll_deg
    if cam_quat is not None and model.phone_sensor_online and roll is not None:
        cam_quat = quat_mul(cam_quat, roll_about_lens_quat(roll))

    meta = CaptureMeta(
        index=index,
        az_deg=az_act,
        el_deg=el_act,
        camera_xyz_mm=_vec3(pose.camera_xyz_mm),
        look_at_xyz_mm=_vec3(pose.look_at_xyz_mm),
        optical_axis_unit=_vec3(pose.optical_axis_unit),
        camera_quat=list(cam_quat) if cam_quat else None,
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        planned_az_deg=planned_az,
        planned_el_deg=planned_el,
        camera_preset=model.camera_preset,
        action=action,
    )
    cap_id, _orig, _thumb, _meta_path, width, height = await asyncio.to_thread(
        storage.save_capture_with_thumb, scan_id, meta, raw,
    )
    capture = Capture(
        **meta.model_dump(),
        capture_id=cap_id,
        thumb_url=f"/captures/{cap_id}/thumb",
        full_url=f"/captures/{cap_id}/full",
        meta_url=f"/captures/{cap_id}/meta",
        stored_width=width,
        stored_height=height,
    )
    return capture.model_dump(mode="json")


# ── path + plan helpers ────────────────────────────────────────────────────


def _build_active_path() -> list[ScanPathPoint]:
    """The path to record on the active machine scan's manifest. Converts the
    `geom.scan_path` dataclass to the Pydantic `models.ScanPathPoint` so the
    manifest serialises cleanly."""
    plan = MotionPlan.model_validate(model.motion_plan)
    d = plan.discrete
    raw = plan_scan_path(d.el_start_deg, d.el_max_deg, d.el_steps, d.az_step_deg)
    return [ScanPathPoint(index=p.index, az_deg=p.az_deg, el_deg=p.el_deg) for p in raw]


def _scan_total_for_plan(path: list[ScanPathPoint]) -> int:
    """Expected total capture count for the active plan: points × actions."""
    plan = MotionPlan.model_validate(model.motion_plan)
    n_actions = max(1, len(plan.discrete.actions))
    return len(path) * n_actions


def _create_scan(path: list[ScanPathPoint], machine_captured: bool = False) -> str:
    """Create a scan manifest on disk from the current model state.

    For machine scans we snapshot the MotionPlanner into `motion_plan` and,
    for backward compatibility with the older `params` field, also fill the
    discrete sub-plan's params there.
    """
    enc = model.encoder_zero or {}
    params: ScanParams | None = None
    motion_plan_dict: dict[str, Any] | None = None
    if machine_captured:
        try:
            plan = MotionPlan.model_validate(model.motion_plan)
        except Exception:  # noqa: BLE001 — fall back to defaults if persisted state is bad
            plan = MotionPlan()
        motion_plan_dict = plan.model_dump(mode="json")
        d = plan.discrete
        params = ScanParams(
            el_start_deg=d.el_start_deg,
            el_max_deg=d.el_max_deg,
            el_steps=d.el_steps,
            az_step_deg=d.az_step_deg,
        )
    req = CreateScanReq(
        geometry=Geometry(
            arm_radius_mm=model.arm_radius_mm,
            camera_offset_mm=model.camera_offset_mm,
            base_height_mm=model.base_height_mm,
        ),
        encoder_zero=EncoderZero(
            az_zero_raw_deg=float(enc.get("az_zero_raw_deg", 0.0)),
            el_zero_raw_deg=float(enc.get("el_zero_raw_deg", 0.0)),
        ),
        params=params,
        machine_captured=machine_captured,
        path=path,
        motion_plan=motion_plan_dict,
    )
    return storage.create_scan(req).scan_id


# ── active scan: persistence (explicit Save + debounced autosave) ──────────
#
# Photo bytes are always written to the captures/ pool immediately. Only the
# lightweight manifest.json — the scan *document* — gets the dirty/save
# treatment: it is written by save_active_scan() (explicit Save) or the
# debounced autosave loop, so the UI can warn about unsaved changes.

_AUTOSAVE_DEBOUNCE_S = 4.0
# Created inside the running loop by start_autosave() — an Event built at
# import time would bind to the wrong loop under pytest's per-test loops.
_dirty_event: asyncio.Event | None = None
_autosave_task: asyncio.Task[None] | None = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _build_manifest() -> Manifest:
    """The active scan's manifest: the on-disk base (created / path / params /
    geometry / motion_plan — fixed at scan creation) with the live captures
    and notes merged in."""
    base = storage.read_manifest(model.current_scan_id)
    return base.model_copy(update={
        "machine_captured": model.machine_captured,
        "captures": [Capture.model_validate(c) for c in model.captures],
        "notes": model.scan_notes,
    })


def _write_active_manifest() -> None:
    storage.write_manifest(_build_manifest())


async def persist_active_manifest_now() -> None:
    """Rewrite the active scan's manifest from the current ``model.captures``,
    off the event loop. A bare write (unlike ``save_active_scan`` it does not
    touch the ``scan_dirty`` / ``scan_saved_at`` flags) — used by the
    delete-capture command so a removed shot is durable on disk immediately,
    not just after the next Save/autosave.

    Raises ``FileNotFoundError`` if no manifest exists yet for the active scan
    (e.g. a brand-new manual session before its first persist); the caller
    decides whether that's fatal."""
    await asyncio.to_thread(_write_active_manifest)


def _mark_dirty() -> None:
    """Flag the active scan as having unsaved changes and wake the autosave."""
    model.update(scan_dirty=True)
    if _dirty_event is not None:
        _dirty_event.set()


def publish_scan_list() -> None:
    """Refresh ``model.scans`` (the Library's ONLY data source) from the
    on-disk scans. Nothing else populates it, so without this saved scans
    never appear in the Library even though they ARE on disk. Cheap — one
    ``list_scans`` — call at startup and after any scan-list mutation
    (save / new / delete / archive)."""
    summaries = [
        ScanSummary(
            scan_id=m.scan_id,
            created=m.created,
            captures_count=len(m.captures),
            archived=m.archived,
            archived_at=m.archived_at,
        ).model_dump(mode="json")
        for m in storage.list_scans()
    ]
    model.update(scans=summaries)


async def save_active_scan() -> dict[str, Any]:
    """Write the active scan's manifest to disk now (the explicit Save)."""
    if not model.current_scan_id:
        return {"saved": False, "reason": "no active scan"}
    scan_id = model.current_scan_id
    await asyncio.to_thread(_write_active_manifest)
    model.update(scan_dirty=False, scan_saved_at=_now())
    publish_scan_list()   # refresh the Library list (new/updated count + saved_at)
    return {"saved": True, "scan_id": scan_id}


async def new_active_scan(machine_captured: bool = False) -> dict[str, Any]:
    """Start a fresh active scan — a new manifest on disk, the model reset.

    Does NOT save the previous active scan; the caller (recreate) or the UI
    is responsible for that. Captures and edits then accumulate into this
    scan until the next new/recreate.
    """
    if machine_captured:
        path = _build_active_path()
        total = _scan_total_for_plan(path)
    else:
        path = []
        total = 0
    scan_id = await asyncio.to_thread(_create_scan, path, machine_captured)
    model.update(
        current_scan_id=scan_id,
        machine_captured=machine_captured,
        captures=[],
        scan_notes="",
        scan_progress=0,
        scan_total=total,
        scan_dirty=False,
        scan_saved_at=_now(),
        # Drop any loaded-for-review scan so its yellow frustums don't linger
        # over a fresh recording.
        loaded_captures=[],
        loaded_scan_id=None,
    )
    log.info("new active scan %s (machine_captured=%s, total=%d)",
             scan_id, machine_captured, total)
    publish_scan_list()   # the new scan appears in the Library immediately
    return {"scan_id": scan_id, "machine_captured": machine_captured, "total": total}


async def recreate_active_scan() -> dict[str, Any]:
    """Save the current scan, then start a fresh one ('Recreate & Save')."""
    saved = await save_active_scan()
    started = await new_active_scan(machine_captured=False)
    return {"saved": saved, "started": started}


async def _autosave_loop(dirty: asyncio.Event) -> None:
    """Debounced autosave: once the active scan goes dirty, wait for a quiet
    `_AUTOSAVE_DEBOUNCE_S` window (so a burst of captures coalesces into one
    write) then persist the manifest."""
    while True:
        await dirty.wait()
        while True:
            dirty.clear()
            try:
                await asyncio.wait_for(dirty.wait(), _AUTOSAVE_DEBOUNCE_S)
            except asyncio.TimeoutError:
                break  # quiet for the debounce window → persist now
        if model.scan_dirty and model.current_scan_id:
            try:
                await save_active_scan()
                log.info("autosaved scan %s", model.current_scan_id)
            except Exception:  # noqa: BLE001
                log.exception("autosave failed")


def start_autosave() -> None:
    """Launch the autosave loop. Called once from the app lifespan — the
    dirty Event is created here so it binds to the app's event loop."""
    global _autosave_task, _dirty_event
    if _autosave_task is not None and not _autosave_task.done():
        return
    _dirty_event = asyncio.Event()
    _autosave_task = asyncio.create_task(_autosave_loop(_dirty_event))


async def stop_autosave() -> None:
    """Cancel the autosave loop — called from the app lifespan shutdown."""
    global _autosave_task
    if _autosave_task is not None:
        _autosave_task.cancel()
        try:
            await _autosave_task
        except asyncio.CancelledError:
            pass
        _autosave_task = None


# ── scan execution (discrete sweep) ─────────────────────────────────────────


async def _capture_with_action(
    scan_id: str,
    index: int,
    planned_az: float | None,
    planned_el: float | None,
    action: str,
) -> dict[str, Any]:
    """Run one MotionPlanner action: prep hardware (torch), settle, capture,
    tear down. Hardware tear-down runs in `finally` so the rig returns to a
    clean state even on capture failure."""
    want_flash = action == "photo_flash"
    try:
        if want_flash:
            await camera_io.set_torch(True)
            await asyncio.sleep(_ACTION_SETTLE_S)
        return await _capture_at(scan_id, index, planned_az, planned_el, action=action)
    finally:
        if want_flash:
            try:
                await camera_io.set_torch(False)
            except Exception:  # noqa: BLE001
                log.exception("failed to turn torch off after capture")


async def _run_discrete(scan_id: str, plan: DiscretePlan) -> None:
    """Discrete loop: ring-grid points × per-point actions."""
    path = plan_scan_path(
        plan.el_start_deg, plan.el_max_deg, plan.el_steps, plan.az_step_deg,
    )
    actions: list[str] = list(plan.actions) or ["photo"]
    captures: list[dict[str, Any]] = []
    log.info("discrete scan %s started — %d points × %d actions",
             scan_id, len(path), len(actions))
    for point in path:
        if _abort.is_set():
            log.info("scan %s aborted at point %d/%d",
                     scan_id, point.index, len(path))
            break
        await esp_move(point.az_deg, point.el_deg)
        for action in actions:
            if _abort.is_set():
                break
            cap = await _capture_with_action(
                scan_id, len(captures), point.az_deg, point.el_deg, action,
            )
            captures.append(cap)
            model.update(
                scan_progress=len(captures),
                captures=list(captures),
            )
            _mark_dirty()


async def _run_motion_plan(scan_id: str, plan: MotionPlan) -> None:
    """Outer scan-runner: hold the scan lock, run the discrete sweep, ensure a
    final manifest save + hardware idle state on exit."""
    async with _scan_lock:
        model.update(scan_running=True)
        log.info("scan %s started", scan_id)
        try:
            await _run_discrete(scan_id, plan.discrete)
        except Exception:  # noqa: BLE001
            log.exception("scan %s failed", scan_id)
        finally:
            model.update(scan_running=False)
            # Belt-and-braces: make sure the torch is off when the loop returns.
            try:
                await camera_io.set_torch(False)
            except Exception:  # noqa: BLE001
                log.debug("torch-off teardown failed", exc_info=True)
            await save_active_scan()
            log.info("scan %s finished", scan_id)


async def start_scan() -> dict[str, Any]:
    """Validate the MotionPlanner, open a fresh active scan, launch the loop.
    Raises with a human-readable message if anything's off."""
    global _scan_task
    if _scan_lock.locked():
        raise RuntimeError("a scan is already running")
    if model.arm_radius_mm <= 0:
        raise RuntimeError("set rig geometry (arm radius) before scanning")

    try:
        plan = MotionPlan.model_validate(model.motion_plan)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"motion_plan is invalid: {exc}") from exc

    if not plan.discrete.actions:
        raise RuntimeError("no capture actions selected in the discrete plan")
    check = plan_scan_path(
        plan.discrete.el_start_deg, plan.discrete.el_max_deg,
        plan.discrete.el_steps, plan.discrete.az_step_deg,
    )
    if not check:
        raise RuntimeError("discrete plan is empty — check params")

    info = await new_active_scan(machine_captured=True)
    scan_id = info["scan_id"]
    _abort.clear()
    _scan_task = asyncio.create_task(_run_motion_plan(scan_id, plan))
    return {"scan_id": scan_id, "total": info["total"], "mode": plan.mode}


async def esp_move(az: float, el: float) -> None:
    """Move the rig — imported lazily so this module has no esp_proxy cycle."""
    from esp_proxy import esp

    await esp.move_and_await(azimuth_deg=az, elevation_deg=el)


def stop_scan() -> dict[str, Any]:
    """Request the running scan to abort at the next iteration."""
    _abort.set()
    return {"stopping": _scan_lock.locked()}


async def take_shot() -> dict[str, Any]:
    """Capture one photo at the current pose. Opens a manual active scan if
    none exists. Refuses while an automated scan is running."""
    if _scan_lock.locked():
        raise RuntimeError("cannot take a manual shot during a scan")
    async with _scan_lock:
        if not model.current_scan_id:
            await new_active_scan(machine_captured=False)
        scan_id = model.current_scan_id
        captures = list(model.captures)
        capture = await _capture_at(scan_id, len(captures), None, None)
        captures.append(capture)
        model.update(captures=captures)
        _mark_dirty()
        return {"scan_id": scan_id, "capture_id": capture["capture_id"]}
