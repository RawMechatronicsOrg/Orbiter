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

Drive one or both axes to target angles. **Blocking** — the response is
held open until both axes are within tolerance, or `timeout_ms` elapses.

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
| `timeout_ms` | int | no | 15000 | Returns `408` if not reached in time. |

Omitting an axis leaves it at its current position.

**200** — target reached
```json
{
  "status": "ok",
  "azimuth_deg":   90.0,
  "elevation_deg": 45.0,
  "duration_ms":   1840
}
```

**408** — timeout; motors keep going
```json
{
  "status": "timeout",
  "azimuth_deg":   87.3,
  "elevation_deg": 44.1,
  "duration_ms":   10000
}
```

**409** — another move is already in progress
```json
{ "status": "busy" }
```

**400** — bad request (e.g. EL out of range).

### Arrival tolerance

| Axis | Tolerance | Source |
|------|-----------|--------|
| AZ | ± 1.0° | One AS5600 step ≈ 0.088°; debounced over 2 readings. |
| EL | ± 1.0° | One AS5048A step ≈ 0.022°; debounced over 2 readings. |

---

## `POST /spin`

Continuous rotation. Useful for video sweeps or warm-up.

**Request**
```json
{ "axis": "az", "rate_deg_per_s": 30.0 }
{ "axis": "el", "rate_deg_per_s": 5.0 }
{ "axis": "az", "rate_deg_per_s": 0 }     // stop
```

**200**
```json
{ "status": "ok", "spinning_az": true, "spinning_el": false }
```

EL spin is clamped at the physical limits — the firmware stops the axis
automatically when it hits `−25°` or `+90°`.

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
