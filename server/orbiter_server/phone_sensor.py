"""Phone IMU poller — reads orientation from the IP Webcam app on the
camera phone and pushes the derived tilt into `model.phone_pitch_deg` /
`model.phone_roll_deg` so the UI can show where the lens is actually
pointing, independent of the rig's encoder pose.

We poll `{camera_url}/sensors.json` at a few Hz. The endpoint returns a
rolling buffer; we use the last sample only.

Primary source: **rot_vector** — Android's OS-fused orientation as a unit
quaternion (gyro + accel + mag). Far steadier than raw accel because the
fusion filters out linear acceleration. Format from IP Webcam:
`data = [[ts_ms, [qx, qy, qz, qw, accuracy]], …]` where
(qx, qy, qz, qw) rotates *device-frame vectors into world frame*.

Pitch above horizon — the phone is mounted so its long edge (device +Y,
toward the top of the phone) runs along the OrbitArm. So the "where the
camera points" direction for EL purposes is device +Y, not device -Z
(which is the physical lens normal, perpendicular to the arm in this
mount). After rotating device +Y into the world:
    arm_world.z = R·(0,1,0) → 2·qy·qz + 2·qw·qx
    pitch_deg = asin(arm_world.z) · 180/π

Roll about the arm axis (signed angle of world-up in device XZ plane,
since the arm is along device-Y so the perpendicular plane is XZ):
    world_up_in_device = Rᵀ·(0,0,1) → (2xz+2wy, …, 1-2x²-2y²)
    roll_deg = atan2(2xz + 2wy, 1 - 2x² - 2y²) · 180/π

Fallback: if rot_vector is missing (older Android, sensor disabled in the
IP Webcam app), fall back to raw accel — less accurate but still useful.

When the camera URL is empty or every fetch fails, the model fields flip
to None and `phone_sensor_online` to False — the UI hides the marker.

Re-target when `model.camera_url` changes (UI edit) — same subscriber
pattern as `EspProxy`.
"""

from __future__ import annotations

import asyncio
import logging
import math
from typing import Any
from urllib.parse import urljoin

import httpx

from orbiter_model import model

log = logging.getLogger("orbiter.phone_sensor")

#: Sensor poll cadence. IP Webcam's /sensors.json is cheap (small JSON), but
#: we don't need it faster than the UI refresh rate.
_POLL_INTERVAL_S = 0.25
#: How long a fetch may take before we give up and mark offline.
_FETCH_TIMEOUT_S = 1.5
#: If we go this long with no successful fetch, flip phone_sensor_online to
#: False so the UI hides the marker (the value is stale).
_OFFLINE_AFTER_S = 4.0
#: Mount offset (deg) subtracted from the raw device-Y pitch so the navball's
#: phone marker reads in the rig EL frame (lens roughly horizontal ⇒ ~0°). The
#: phone sits rotated ~90° relative to this formula's nominal axes (its measured
#: roll is ~90°), so the raw pitch is ~90° off the EL convention; subtracting it
#: makes the marker track `el`. Empirically set against the reference rig.
#: NOTE: only `phone_pitch_deg` (the navball readout) gets this offset;
#: `phone_lens_pitch_deg` (used by `align_el_to_phone`) is left untouched.
_PITCH_MOUNT_OFFSET_DEG = 90.0


class PhoneSensor:
    def __init__(self) -> None:
        self._task: asyncio.Task[None] | None = None
        self._stopping = asyncio.Event()
        self._url: str = ""

    # ── lifecycle ───────────────────────────────────────────────────────────

    async def start(self) -> None:
        self._url = model.camera_url or ""
        self._stopping.clear()
        self._task = asyncio.create_task(self._poll_loop())
        model.subscribe(self._on_model_update)
        log.info("phone-sensor poller started (camera_url=%r)", self._url)

    async def stop(self) -> None:
        self._stopping.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        log.info("phone-sensor poller stopped")

    def _on_model_update(self, patch: dict[str, Any]) -> None:
        if "camera_url" not in patch:
            return
        new_url = str(patch["camera_url"] or "")
        if new_url == self._url:
            return
        log.info("camera_url changed → %r", new_url)
        self._url = new_url
        # Wake the loop so the next iteration sees the new URL immediately.
        # The loop already re-reads self._url each tick, so there's nothing
        # else to cancel/restart — just reset the online flag.
        if not new_url:
            model.update(phone_sensor_online=False,
                         phone_pitch_deg=None, phone_roll_deg=None)

    # ── poll ────────────────────────────────────────────────────────────────

    async def _poll_loop(self) -> None:
        last_ok = 0.0
        # Single client reused across ticks for keep-alive efficiency.
        async with httpx.AsyncClient(timeout=_FETCH_TIMEOUT_S) as client:
            while not self._stopping.is_set():
                await asyncio.sleep(_POLL_INTERVAL_S)
                url_base = self._url
                if not url_base:
                    continue
                try:
                    pitch, lens_pitch, roll = await self._fetch_tilt(
                        client, url_base,
                    )
                except Exception as exc:  # noqa: BLE001
                    if model.phone_sensor_online:
                        log.warning("phone sensor fetch failed: %s", exc)
                    elapsed = asyncio.get_running_loop().time() - last_ok
                    if elapsed > _OFFLINE_AFTER_S and (
                        model.phone_sensor_online
                        or model.phone_pitch_deg is not None
                    ):
                        model.update(phone_sensor_online=False,
                                     phone_pitch_deg=None,
                                     phone_lens_pitch_deg=None,
                                     phone_roll_deg=None)
                    continue
                last_ok = asyncio.get_running_loop().time()
                model.update(
                    phone_sensor_online=True,
                    phone_pitch_deg=pitch,
                    phone_lens_pitch_deg=lens_pitch,
                    phone_roll_deg=roll,
                )

    async def _fetch_tilt(
        self, client: httpx.AsyncClient, url_base: str,
    ) -> tuple[float, float, float]:
        """Return (pitch_deg, lens_pitch_deg, roll_deg) from the latest
        sensor sample — rot_vector if available, accel as fallback.

        Phone is in **landscape** mount: arm runs along the short edge
        (device ±X) and tilting the rig EL rotates the phone about its
        device-Y axis. So:

          * pitch     — full-range tilt angle about device-Y (atan2 form,
                        continuous through ±90°, no gimbal lock here).
                        Goes to the navball.
          * lens_pitch — elevation of the back-camera optical axis (device
                         -Z) above the world horizon. Used by
                         `align_el_to_phone` so rig.el == 0 ⇒ lens
                         horizontal.
          * roll      — bank about the OPTICAL axis (device −Z); the slight
                        bracket tilt. Drives the live frustum bank.
        """
        url = urljoin(url_base.rstrip("/") + "/", "sensors.json")
        resp = await client.get(url)
        resp.raise_for_status()
        data = resp.json()

        rv = (data.get("rot_vector") or {}).get("data") or []
        if rv:
            _ts, sample = rv[-1]
            # IP Webcam layout: [qx, qy, qz, qw, accuracy]. The first 4 are
            # a unit quaternion rotating device-frame → world-frame.
            if isinstance(sample, (list, tuple)) and len(sample) >= 4:
                qx, qy, qz, qw = (float(sample[i]) for i in range(4))
                qmag2 = qx * qx + qy * qy + qz * qz + qw * qw
                if 0.95 <= qmag2 <= 1.05:
                    # pitch = atan2 form that tracks the tilt rotation
                    # about device-Y across the full range; shifted into the
                    # rig EL frame by the mount offset (see _PITCH_MOUNT_OFFSET_DEG).
                    pitch = math.degrees(math.atan2(
                        2 * qx * qz + 2 * qw * qy,
                        1 - 2 * qx * qx - 2 * qy * qy,
                    )) - _PITCH_MOUNT_OFFSET_DEG
                    # lens_pitch: world-z of R·(0,0,-1) → 2x² + 2y² - 1.
                    lens = max(-1.0, min(1.0, 2 * qx * qx + 2 * qy * qy - 1))
                    lens_pitch = math.degrees(math.asin(lens))
                    # roll: TRUE bank about the OPTICAL axis (device −Z) — the
                    # slight tilt from a non-rigid bracket. atan2 of world-up's
                    # components along camera right (device +X) vs up (+Y) =
                    # atan2(R[2][0], R[2][1]). Small & STABLE across EL (unlike
                    # device-+Y elevation, which the old code used and which
                    # tracked EL → spun the frustum).
                    roll = math.degrees(math.atan2(
                        2 * qx * qz - 2 * qw * qy,
                        2 * qy * qz + 2 * qw * qx,
                    ))
                    return pitch, lens_pitch, roll

        # Fallback: raw accel. accel/|A| = world-up in device frame
        # (third row of R). pitch = atan2(ax, az) tracks Y-rotation;
        # lens_pitch uses az; roll uses ay.
        accel = (data.get("accel") or {}).get("data") or []
        if not accel:
            raise RuntimeError("no rot_vector and no accel samples")
        _ts, sample = accel[-1]
        if not isinstance(sample, (list, tuple)) or len(sample) < 3:
            raise RuntimeError(f"malformed accel sample: {sample!r}")
        ax, ay, az = float(sample[0]), float(sample[1]), float(sample[2])
        mag = math.sqrt(ax * ax + ay * ay + az * az)
        if mag < 1.0:
            raise RuntimeError(f"accel magnitude too small: {mag:.3f}")
        pitch = math.degrees(math.atan2(ax, az)) - _PITCH_MOUNT_OFFSET_DEG
        lens_pitch = math.degrees(math.asin(max(-1.0, min(1.0, -az / mag))))
        roll = math.degrees(math.atan2(ax, ay))  # bank about the optical axis
        return pitch, lens_pitch, roll


# Process-wide singleton — same pattern as the other lifespan-managed bits.
phone_sensor = PhoneSensor()
