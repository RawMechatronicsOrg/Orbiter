# Firmware API (REST + WebSocket)

The ESP32 exposes a small HTTP API. Bodies are `application/json`.

This API treats the device as a **generic 2-axis IP actuator** with
absolute angle feedback. It doesn't know it's a turntable. You can drop
this firmware on any 2-axis mechanism (gimbal, antenna rotator, ...) and
the server will still talk to it.

---

## `GET /health`

Liveness probe. Always responds immediately.

**200**
```json
{
  "status": "ok",
  "device": "Orbiter",
  "chip": "ESP32-D0WD-V3",
  "revision": 3,
  "cores": 2,
  "mac": "d4:e9:f4:88:f3:e8",
  "free_heap_bytes": 214320,
  "uptime_ms": 83400,
  "idf_version": "v6.0.1"
}
```

---

## `GET /state`

Current angles (live encoder reads), motor enable, coarse motion state,
per-axis spin flags, and persisted calibration.

Angles are in **user space** (raw encoder reading minus the calibration
zero). Azimuth wraps to `[0°, 360°)`. Elevation is signed.

**200**
```json
{
  "state": "idle",
  "motors_enabled": true,
  "spinning_az": false,
  "spinning_el": false,
  "azimuth":   { "angle_deg": 123.45 },
  "elevation": { "angle_deg":  -3.10 },
  "calibration": {
    "az_zero_raw_deg":   0.000,
    "el_zero_raw_deg": 138.296
  }
}
```

| `state` | Meaning |
|---------|---------|
| `idle` | Nothing in progress |
| `moving` | A `/move` request is in flight |
| `spinning` | At least one axis is in continuous spin |
| `error` | Motion error |

---

## `POST /move`

Drive one or both axes to target angles. **Asynchronous** — the firmware
queues the move and answers `202` at once with a `task_id`; the move runs on
the motion runner while the HTTP server stays free for `/state` and the
`/ws/log` stream.

**Request**
```json
{
  "azimuth_deg":   90.0,
  "elevation_deg": 45.0,
  "timeout_ms":  10000
}
```

| Field | Type | Required | Default | Notes |
|-------|------|----------|---------|-------|
| `azimuth_deg` | float | no | unchanged | Any value. Firmware picks the shortest path on the circle. |
| `elevation_deg` | float | no | unchanged | Must be in `[−25, +90]`. Out of range → 400. |
| `timeout_ms` | int | no | 15000 | The task ends with status `timeout` if the target is not reached in time. |
| `az_hz_max` | int | no | 0 | Per-move AZ top step rate (Hz). `0`/absent = compiled default; clamped to a safe band in firmware. The UI speed knobs flow in here. |
| `el_hz_max` | int | no | 0 | Per-move EL top step rate (Hz). Same semantics as `az_hz_max`. |

Omitting an axis leaves it at its current position.

**202** — accepted
```json
{ "status": "accepted", "task_id": 17, "kind": "move" }
```

The outcome arrives as a `task` frame on [`/ws/log`](#get-wslog-websocket)
and in `GET /state` → `runner`, keyed by `task_id`:

```json
{ "kind": "task", "task_id": 17, "status": "done",
  "result": { "azimuth_deg": 90.0, "elevation_deg": 45.0, "duration_ms": 1840 } }
```

`status` is `done`, `timeout` (not reached within `timeout_ms`) or `error`.

**409** — another task is still running
```json
{ "status": "busy" }
```

**400** — bad request: neither axis given, or `elevation_deg` outside the
travel (`{"status":"error","message":"elevation_deg out of range -25..+90"}`).

### Arrival tolerance

| Axis | Tolerance | Source |
|------|-----------|--------|
| AZ | ± 1.0° | One AS5600 step ≈ 0.088°; debounced over 2 readings. |
| EL | ± 1.0° | One AS5600 step ≈ 0.088°; debounced over 2 readings. |

---

## `POST /spin`

Continuous rotation of one axis, open loop, at a driver step rate. Useful
for video sweeps or warm-up.

> **No range check on EL.** Unlike `/move`, a spin is not stopped at the
> elevation limits: on the v2 mechanics the arm meets the cross rail at
> −28.8°. Spin elevation only while watching it. (The Orbiter server stops
> an EL spin it started a few degrees before the limits, from the encoder
> reading.)

**Request**
```json
{ "axis": "az", "dir": "cw",  "step_hz": 2100 }
{ "axis": "el", "dir": "ccw", "step_hz": 350 }
```

| Field | Type | Required | Default | Notes |
|-------|------|----------|---------|-------|
| `axis` | `"az"` \| `"el"` | yes | — | |
| `dir` | `"cw"` \| `"ccw"` | no | `cw` | |
| `step_hz` | int | no | 525 | Driver step rate (1/16 microsteps). `0` or above 10000 falls back to 525. At the v2 gearing 1 °/s is about 62 Hz on AZ and 89 Hz on EL. |

**200**
```json
{ "status": "ok", "axis": "az", "dir": "cw", "step_hz": 2100 }
```

**409** — that axis is already spinning, or a move is running.

The pose frames on `/ws/log` carry `sp_az` / `sp_el` while an axis spins.

## `POST /spin/stop`

```json
{ "axis": "az" }      // or "el", "both"
```

**200** `{ "status": "ok" }`

---

## `POST /calibrate`

Update encoder zero offsets. Persisted to NVS, survives reboot.

**Request**
```json
{ "axis": "el", "mode": "current" }
{ "axis": "az", "mode": "explicit", "az_raw_deg": 0.0 }
{ "axis": "both", "mode": "reset" }
```

| Mode | Effect |
|------|--------|
| `current` | Capture the current raw encoder reading as the new zero. |
| `explicit` | Set zero to the value in `az_raw_deg` / `el_raw_deg`. |
| `reset` | Restore factory defaults (`az = 0.0`, `el = 138.296`). |

**200**
```json
{
  "status": "ok",
  "axis": "el",
  "mode": "current",
  "az_zero_raw_deg":   0.000,
  "el_zero_raw_deg": 142.117
}
```

> `POST /zero` is kept as a backward-compatibility alias for
> `POST /calibrate {"mode": "current"}`.

---

## `POST /reboot`

Restart the firmware. Acks **200** first; the actual reboot fires ~500 ms
later so the response reaches you.

**200**
```json
{ "status": "ok", "message": "rebooting" }
```

---

## `POST /ota`

Update the firmware over Wi-Fi. The request body is the **raw app image**
(`build/orbiter.bin`), not JSON; the firmware streams it into the inactive
OTA slot.

```bash
curl --data-binary @build/orbiter.bin http://<esp-ip>/ota
```

Before the image is accepted the firmware checks that the axes are idle,
that the body is not empty, and that it fits the slot. While the image is
written the motors are disabled. On success the slot becomes the boot
partition and the device reboots about 500 ms after the response.

A new image confirms itself once its HTTP server is up. If that does not
happen within 120 s, the bootloader rolls back to the previous image. The
stored calibration (NVS) is not touched by an update.

**200** — image written and verified; rebooting
```json
{ "status": "ok", "message": "updated, rebooting" }
```

If the reboot timer cannot be created, the message is
`"updated, reboot to run it"` — send `POST /reboot`.

Errors have the form `{ "status": "error", "message": "..." }`:

| Code | When |
|------|------|
| 400 | The image is not an Orbiter firmware image (it belongs to another project), the body is empty, or the written image failed verification (checksum / hash). |
| 409 | An axis is moving or spinning — stop it first. |
| 413 | The image is larger than the OTA slot. |
| 500 | Any other failure: no OTA slot (flash the OTA partition table over USB once), out of memory, receive timeout, or a flash write / boot-partition error. |

> **First flash over USB.** A board with the old single-app partition table
> has no OTA slot. Flash it once with `idf.py -p COM3 flash`; some ESP32
> dev-board clones need **BOOT** held while `esptool` prints `Connecting...`
> (see [FIRMWARE.md](FIRMWARE.md#updating-over-wi-fi)). After that, use
> `/ota`.

---

## `GET /ws/log` (WebSocket)

Streaming feed of curated firmware events. Each text frame is one JSON
object:

```json
{ "seq": 142, "ts_ms": 83400, "lvl": "I", "tag": "motion", "msg": "move done in 1840 ms — AZ=90.00° EL=45.00° OK" }
```

| Field | Description |
|-------|-------------|
| `seq` | Monotonic counter since boot. Use for client-side ordering. |
| `ts_ms` | Milliseconds since boot. |
| `lvl` | `"I"` / `"W"` / `"E"`. |
| `tag` | Module tag — `motion`, `http`, `encoder`. |
| `msg` | Pre-formatted message. |

Up to 4 concurrent clients. Only operational events are emitted — this is
**not** a mirror of `ESP_LOGx` (that would risk recursion through Wi-Fi /
HTTP logs).
