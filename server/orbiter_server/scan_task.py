"""Server-side scan orchestration — the MotionPlanner loop.

The active scan-loop plan lives on `model.motion_plan` (see `models.MotionPlan`).
`start_scan` reads it and runs the discrete sweep: elevation rings × azimuth
step. At each ring point the loop runs every checked `action` in order
(`photo`, `photo_flash`). One stored capture per action.

Photo bytes always hit the `captures/` pool immediately; only the lightweight
manifest.json gets the dirty / save / autosave treatment (UI shows it as
"unsaved changes").

A capture is two steps: `_take` exposes the photo at the current pose and
`_store` gets it into the pool. The sweep moves the rig on as soon as `_take`
returns and runs `_store` in the background (`_StoreQueue`). With a
CameraStreamer capture node `_take` returns once the frame is exposed and
`_store` pulls the JPEG off the phone, so the transfer overlaps the next move;
with IP Webcam `_take` already includes the transfer.
"""

from __future__ import annotations

import asyncio
import bisect
import logging
import threading
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, TypeVar

import storage
from camera_adapter import get_preset
from config import settings
from geom.pose import GeomParams, camera_pose_at
from geom.rig import MountTransform
from geom.scan_path import plan_scan_path
from geom.transforms import (
    quat_mul,
    roll_about_lens_quat,
    stored_pixel_quat,
)
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
import camera_io
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

# Transient-failure policy: a phone/ESP hiccup must
# not kill a 30-minute sweep. Each move/capture gets `_OP_ATTEMPTS` tries
# with growing backoff; a point that still fails is RECORDED (manifest
# `failed_points`) and skipped. Only `_MAX_CONSECUTIVE_FAILED_POINTS` points
# failing in a row — the rig genuinely down, retrying the rest is pointless
# wear — aborts the scan.
_OP_ATTEMPTS = 3
_OP_BACKOFF_S = (1.0, 3.0)
_MAX_CONSECUTIVE_FAILED_POINTS = 3
#: Photos taken but not yet in the pool. Each one holds the phone's storage
#: (capture node) or memory (IP Webcam); beyond this the sweep waits.
_MAX_PENDING_STORES = 4

_T = TypeVar("_T")


async def _with_retries(op_name: str, fn: Callable[[], Awaitable[_T]]) -> _T:
    """Run ``fn`` with the transient-failure policy; raises the LAST error
    after `_OP_ATTEMPTS` tries."""
    for attempt in range(1, _OP_ATTEMPTS + 1):
        try:
            return await fn()
        except Exception as exc:  # noqa: BLE001 — every failure mode retries
            if attempt == _OP_ATTEMPTS:
                raise
            delay = _OP_BACKOFF_S[min(attempt - 1, len(_OP_BACKOFF_S) - 1)]
            log.warning("%s failed (attempt %d/%d): %s — retrying in %.0fs",
                        op_name, attempt, _OP_ATTEMPTS, exc, delay)
            await asyncio.sleep(delay)
    raise AssertionError("unreachable")


def _geom_params() -> GeomParams:
    # Calibrated mount X + rocker when available — capture poses (and the
    # SfM priors built from them) must come from the same calibrated chain
    # the solver verified, not the manual fallback model.
    return GeomParams(
        extrinsic=MountTransform.from_dict(getattr(model, "calib_extrinsic", None)),
        arm_radius=max(model.arm_radius_mm, 1.0),
        camera_offset=model.camera_offset_mm,
        # Tilt zeroed — see scene_graph._geom_params for the rationale.
        camera_tilt=0.0,
        camera_pan=model.camera_pan_deg,
        turntable_axis=getattr(model, "turntable_axis", None),
        rocker=MountTransform.from_dict(getattr(model, "rocker_correction", None)),
        az_harm=(tuple(model.az_encoder_correction)
                 if settings.az_harmonic_enabled
                 and getattr(model, "az_encoder_correction", None) else None),
        el_scale=getattr(model, "el_encoder_scale", None),
    )


def _vec3(t: tuple[float, float, float]) -> Vec3:
    return Vec3(x=t[0], y=t[1], z=t[2])


# ── single capture ─────────────────────────────────────────────────────────


@dataclass
class _Taken:
    """A photo exposed at one pose: ``raw`` when it is already here (IP
    Webcam), ``shot_id`` while it is still on the capture node."""

    meta: CaptureMeta
    raw: bytes | None = None
    shot_id: str | None = None


async def _take(
    scan_id: str,
    index: int,
    planned_az: float | None,
    planned_el: float | None,
    action: str | None = None,
    shot_id: str | None = None,
) -> _Taken:
    """Expose one photo at the current rig pose. On a capture node this
    returns as soon as the frame is exposed — the rig may move on — and the
    JPEG stays on the phone for `_store`."""
    geom = _geom_params()
    az_act, el_act = model.az, model.el
    pose = camera_pose_at(az_act, el_act, geom)
    # The orientation the pixels will be stored in: a capture node's follows
    # its sensor, IP Webcam's the operator's preset (`camera_io.capture_preset`).
    preset = get_preset(await camera_io.capture_preset())
    raw: bytes | None = None
    if await camera_io.is_capture_node():
        shot_id = shot_id or camera_io.new_shot_id(f"{scan_id}-{index:04d}")
        await camera_io.trigger_shot(shot_id, {
            "scan_id": scan_id, "index": index, "action": action,
            "az_deg": az_act, "el_deg": el_act,
        })
    else:
        shot_id = None
        raw = await camera_io.fetch_photo(el_deg=el_act)

    # Bank the stored camera orientation by the live bracket roll — ONLY on
    # the manual-model path. The calibrated mount X already carries the
    # phone's real roll (it is part of the solved orientation); banking on
    # top of it would rotate the stored pose ~2× off the truth (the same
    # double-counting that put the live frustum 90° off the camera body).
    cam_quat = pose.camera_quat
    roll = model.phone_roll_deg
    if (geom.extrinsic is None and cam_quat is not None
            and model.phone_sensor_online and roll is not None):
        cam_quat = quat_mul(cam_quat, roll_about_lens_quat(roll))

    # The solved pose lives in the SENSOR frame, but original.jpg is saved
    # AFTER the preset's quarter-turn rotation (e.g. sm22 -> 90 deg CW).
    # Re-express the quaternion in the stored-pixel frame so the manifest
    # matches the pixels — frustum renderers and the SfM priors exporter
    # both rely on that invariant.
    if cam_quat is not None:
        cam_quat = stored_pixel_quat(cam_quat, preset.extra_cw_quarter_turns)

    meta = CaptureMeta(
        index=index,
        az_deg=az_act,
        el_deg=el_act,
        camera_xyz_mm=_vec3(pose.camera_xyz_mm),
        look_at_xyz_mm=_vec3(pose.look_at_xyz_mm),
        optical_axis_unit=_vec3(pose.optical_axis_unit),
        camera_quat=list(cam_quat) if cam_quat else None,
        quat_frame="stored" if cam_quat else None,
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        planned_az_deg=planned_az,
        planned_el_deg=planned_el,
        camera_preset=preset.name,
        action=action,
    )
    return _Taken(meta=meta, raw=raw, shot_id=shot_id)


async def _store(scan_id: str, taken: _Taken) -> dict[str, Any]:
    """Get a taken photo into the captures pool — pulled off the phone first
    when it is still there — and return the JSON-able Capture dict. The
    phone's copy is released only once the pool holds it. Safe to retry."""
    meta, raw = taken.meta, taken.raw
    if raw is None:
        raw, record = await camera_io.fetch_shot(taken.shot_id)
        meta = meta.model_copy(update={"camera_shot": camera_io.shot_summary(record)})
    cap_id, _orig, _thumb, _meta_path, width, height = await asyncio.to_thread(
        storage.save_capture_with_thumb, scan_id, meta, raw,
    )
    if taken.shot_id is not None:
        await camera_io.release_shot(taken.shot_id)
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


async def _capture_at(
    scan_id: str,
    index: int,
    planned_az: float | None,
    planned_el: float | None,
    action: str | None = None,
) -> dict[str, Any]:
    """Take one photo at the current rig pose and store it. Returns the
    JSON-able Capture dict."""
    return await _store(scan_id, await _take(scan_id, index, planned_az, planned_el, action))


def _shot_attempts(base_id: str, take: Callable[[str], Awaitable[_T]]) -> Callable[[], Awaitable[_T]]:
    """Adapts a shot for `_with_retries`: retries reuse the shot id — the phone
    never shoots a known id twice, so a lost answer cannot double a frame —
    until the phone reports the shot itself failed; then the next attempt
    needs a fresh id."""
    failed = 0

    async def attempt() -> _T:
        nonlocal failed
        try:
            return await take(base_id if failed == 0 else f"{base_id}-r{failed}")
        except camera_io.ShotError as exc:
            if exc.definitive:
                failed += 1
            raise

    return attempt


class _StoreQueue:
    """Stores taken photos in the background while the sweep moves on, at
    most `_MAX_PENDING_STORES` pending. Transfers run one at a time: one
    already fills the WiFi link, and parallel ones only delay the next shot's
    small request behind them (bench: 800 ms instead of 240 ms on a slow
    link). Captures land in ``captures`` in index order."""

    def __init__(self, scan_id: str, captures: list[dict[str, Any]]) -> None:
        self._scan_id = scan_id
        self._captures = captures
        self._tasks: set[asyncio.Task[None]] = set()
        self._transfer = asyncio.Lock()

    async def add(self, point: ScanPathPoint, action: str, taken: _Taken) -> None:
        while len(self._tasks) >= _MAX_PENDING_STORES:
            await asyncio.wait(self._tasks, return_when=asyncio.FIRST_COMPLETED)
        task = asyncio.create_task(self._store(point, action, taken))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def drain(self) -> None:
        if self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)

    async def _store(self, point: ScanPathPoint, action: str, taken: _Taken) -> None:
        try:
            async with self._transfer:
                cap = await _with_retries(
                    f"store {action} @ point {point.index}",
                    lambda: _store(self._scan_id, taken),
                )
        except Exception as exc:  # noqa: BLE001 — the point fails, the sweep goes on
            stage = f"{action} transfer" + (f" (shot {taken.shot_id} left on the phone)"
                                            if taken.shot_id else "")
            _record_failed_point(point, stage, exc)
            await _persist_capture_now()
            return
        bisect.insort(self._captures, cap, key=lambda c: c["index"])
        model.update(scan_progress=len(self._captures), captures=list(self._captures))
        # Write-through: the capture is durable the moment it exists.
        await _persist_capture_now()


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
    and failed points merged in.

    Fields this session did not touch stay as they are on disk — notes/tags
    can be edited concurrently through the REST catalog routes, and a
    wholesale overwrite from memory is a lost-update bug. ``notes`` is taken
    from the model only when it was actually edited in the Scaner UI this
    session."""
    base = storage.read_manifest(model.current_scan_id)
    update: dict[str, Any] = {
        "machine_captured": model.machine_captured,
        "captures": [Capture.model_validate(c) for c in model.captures],
        "failed_points": list(model.scan_failed_points),
    }
    if model.scan_notes_edited:
        update["notes"] = model.scan_notes
    return base.model_copy(update=update)


#: One manifest write at a time. Background stores, the sweep and the autosave
#: each write from their own worker thread; two renames over the same
#: manifest.json race (on a network share the loser gets "Access is denied").
_manifest_write = threading.Lock()


def _write_active_manifest() -> None:
    with _manifest_write:
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


async def _persist_capture_now() -> None:
    """Write-through: make the just-appended capture durable immediately.

    The debounce-only manifest left a ≤4 s crash window in which pool bytes
    existed but their manifest records did not (the orphaned-captures class
    of bugs). A manifest rewrite is a few-ms `to_thread`
    write at our rates — do it per capture; the debounced autosave stays for
    notes typing. On failure fall back to marking dirty (the autosave loop
    retries)."""
    try:
        await asyncio.to_thread(_write_active_manifest)
        model.update(scan_dirty=False, scan_saved_at=_now())
    except Exception:  # noqa: BLE001 — never let persistence kill the sweep
        log.exception("write-through manifest save failed — autosave will retry")
        _mark_dirty()


def publish_scan_list() -> None:
    """Refresh ``model.scans`` (the Library's ONLY data source) from the
    on-disk scans. Nothing else populates it, so without this saved scans
    never appear in the Library even though they ARE on disk. Cheap — one
    ``list_scans`` call — at startup and after any scan-list mutation
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
        scan_notes_edited=False,
        scan_failed_points=[],
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
    `_AUTOSAVE_DEBOUNCE_S` window (so a burst of edits coalesces into one
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
    """Cancel the autosave loop and FLUSH a dirty active scan — called from
    the app lifespan shutdown. Without the flush even a graceful Ctrl-C
    inside the debounce window lost the last captures' manifest records."""
    global _autosave_task
    if _autosave_task is not None:
        _autosave_task.cancel()
        try:
            await _autosave_task
        except asyncio.CancelledError:
            pass
        _autosave_task = None
    if model.scan_dirty and model.current_scan_id:
        try:
            await save_active_scan()
            log.info("shutdown flush: saved dirty scan %s", model.current_scan_id)
        except Exception:  # noqa: BLE001 — shutdown must not raise
            log.exception("shutdown flush failed for scan %s",
                          model.current_scan_id)


# ── scan execution (discrete sweep) ─────────────────────────────────────────


async def _take_with_action(
    scan_id: str,
    index: int,
    planned_az: float | None,
    planned_el: float | None,
    action: str,
    shot_id: str,
) -> _Taken:
    """Run one MotionPlanner action: prep hardware (torch), settle, take,
    tear down. Hardware tear-down runs in `finally` so the rig returns to a
    clean state even on capture failure."""
    want_flash = action == "photo_flash"
    try:
        if want_flash:
            await camera_io.set_torch(True)
            await asyncio.sleep(_ACTION_SETTLE_S)
        return await _take(scan_id, index, planned_az, planned_el, action=action, shot_id=shot_id)
    finally:
        if want_flash:
            try:
                await camera_io.set_torch(False)
            except Exception:  # noqa: BLE001
                log.exception("failed to turn torch off after capture")


def _record_failed_point(
    point: ScanPathPoint, stage: str, exc: Exception,
) -> None:
    """Append one skipped point to the model's failed-points list (persisted
    as manifest ``failed_points`` on the next write)."""
    entry = {
        "index": point.index,
        "az_deg": point.az_deg,
        "el_deg": point.el_deg,
        "stage": stage,
        "error": str(exc)[:300] or type(exc).__name__,
        "timestamp": _now(),
    }
    model.update(scan_failed_points=[*model.scan_failed_points, entry])
    _ui_log("E", f"scan point {point.index} (az={point.az_deg:.1f} el={point.el_deg:.1f}) "
                 f"FAILED at {stage} after retries: {entry['error']} "
                 "— skipping the point, sweep continues")


def _ui_log(level: str, msg: str) -> None:
    """Mirror a scan line to the server log AND the UI LogPanel — a skipped
    point or an aborted sweep must be seen while it happens. Best-effort: a
    no-op without a running hub (unit tests). `level` ∈ {'I','W','E'}."""
    {"W": log.warning, "E": log.error}.get(level, log.info)(msg)
    try:
        from ws_hub import hub
        hub.emit_log({"level": level, "source": "api", "tag": "scan", "msg": msg})
    except Exception:  # noqa: BLE001
        pass


async def _run_discrete(scan_id: str, plan: DiscretePlan) -> None:
    """Discrete loop: ring-grid points × per-point actions.

    Transient camera/ESP failures retry (`_with_retries`); a point that
    still fails is recorded in `failed_points` and SKIPPED — one hiccup must
    not discard the rest of a 30-minute sweep. Only
    `_MAX_CONSECUTIVE_FAILED_POINTS` failures in a row abort the scan."""
    path = plan_scan_path(
        plan.el_start_deg, plan.el_max_deg, plan.el_steps, plan.az_step_deg,
    )
    actions: list[str] = list(plan.actions) or ["photo"]
    captures: list[dict[str, Any]] = []
    stores = _StoreQueue(scan_id, captures)
    log.info("discrete scan %s started — %d points × %d actions",
             scan_id, len(path), len(actions))
    try:
        await _sweep(scan_id, path, actions, stores)
    finally:
        # Photos already taken still get stored, also after an abort or a
        # failure — they are on the phone or in memory, not yet in the pool.
        await stores.drain()


async def _sweep(
    scan_id: str,
    path: list[ScanPathPoint],
    actions: list[str],
    stores: _StoreQueue,
) -> None:
    """The move → take loop of `_run_discrete`; stores run in ``stores``."""
    next_index = 0
    consecutive_failed = 0
    for point in path:
        if _abort.is_set():
            log.info("scan %s aborted at point %d/%d",
                     scan_id, point.index, len(path))
            break

        try:
            await _with_retries(
                f"move to az={point.az_deg:.1f} el={point.el_deg:.1f}",
                lambda p=point: esp_move(p.az_deg, p.el_deg),
            )
        except Exception as exc:  # noqa: BLE001 — point failure, not scan failure
            _record_failed_point(point, "move", exc)
            await _persist_capture_now()
            consecutive_failed += 1
            if consecutive_failed >= _MAX_CONSECUTIVE_FAILED_POINTS:
                raise RuntimeError(
                    f"{consecutive_failed} consecutive points failed — "
                    "rig looks down, aborting the scan"
                ) from exc
            continue

        point_ok = True
        for action in actions:
            if _abort.is_set():
                break
            index = next_index
            try:
                taken = await _with_retries(
                    f"capture {action} @ point {point.index}",
                    _shot_attempts(
                        f"{scan_id}-{index:04d}",
                        lambda shot_id, p=point, a=action, i=index:
                            _take_with_action(scan_id, i, p.az_deg, p.el_deg, a, shot_id),
                    ),
                )
            except Exception as exc:  # noqa: BLE001 — skip this point's action
                _record_failed_point(point, action, exc)
                await _persist_capture_now()
                point_ok = False
                break
            next_index += 1
            # The rig moves on now; the photo reaches the pool in the background.
            await stores.add(point, action, taken)

        if point_ok:
            consecutive_failed = 0
        else:
            consecutive_failed += 1
            if consecutive_failed >= _MAX_CONSECUTIVE_FAILED_POINTS:
                raise RuntimeError(
                    f"{consecutive_failed} consecutive points failed — "
                    "rig looks down, aborting the scan"
                )


async def _run_motion_plan(scan_id: str, plan: MotionPlan) -> None:
    """Outer scan-runner: hold the scan lock, run the discrete sweep, ensure a
    final manifest save + hardware idle state on exit."""
    async with _scan_lock:
        model.update(scan_running=True)
        log.info("scan %s started", scan_id)
        try:
            await _run_discrete(scan_id, plan.discrete)
        except Exception as exc:  # noqa: BLE001
            log.exception("scan %s failed", scan_id)
            _ui_log("E", f"scan {scan_id} aborted: {exc}")
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

    if camera_io.camera_base() is None:
        raise RuntimeError("camera_url is not configured — set the camera URL in "
                           "Machine config → Connections")
    if await camera_io.is_capture_node():
        try:
            await camera_io.require_locked_node()
        except camera_io.ShotError as exc:
            raise RuntimeError(str(exc)) from exc

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
        # Write-through: durable immediately (same policy as the scan loop).
        await _persist_capture_now()
        return {"scan_id": scan_id, "capture_id": capture["capture_id"]}
