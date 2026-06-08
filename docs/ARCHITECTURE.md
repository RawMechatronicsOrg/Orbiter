# Architecture

A 3-tier setup. The frontend never talks to the ESP32 directly.

```
┌──────────────┐    WebSocket    ┌──────────────┐    HTTP/WS    ┌──────────────┐
│   Browser    │ ◄──────────────►│   Server     │ ◄────────────►│    ESP32     │
│  (Vite + R3F)│    /ws/scene    │  (FastAPI)   │    /move      │  (firmware)  │
└──────────────┘                 └──────────────┘    /state     └──────┬───────┘
                                       │                                │
                                       │              ┌────────────────►│
                                       │              │  encoders       │
                                       │   HTTP       │  steppers       │
                                       ▼              │                 │
                                ┌──────────────┐      │
                                │ IP Webcam    │      │
                                │ (Android)    │      │
                                └──────────────┘      │
                                                      │
                                ┌─────────────────────┘
                                │
                                ▼
                         JPEGs over HTTP
```

## Why the server is in the middle

The server is the source of truth for **scene state and geometry**:

- It holds the model state (current AZ/EL, motor enable, scan session,
  saved captures).
- It computes camera pose from `(az, el, arm_radius, base_height,
  camera_offset)`. The browser doesn't do trig.
- It proxies every ESP32 call (and the firmware log stream).
- It writes images and `manifest.json` to disk.

The frontend is a **thin viewer** over a scene-graph diff protocol on
`/ws/scene`. The server sends node-add / node-update / node-delete messages,
the frontend renders. Commands flow the other way as named operations
(`move`, `take_shot`, `save_scan`, ...).

This pattern is sometimes called "server-side scene graph" (the
[Viser](https://github.com/nerfstudio-project/viser) project popularised it).
We don't depend on Viser itself — we ship our own minimal implementation, but
the idea is the same: one piece of code owns the truth, the browser is a
display.

**The reason:** off-by-unit bugs in 3D math compound very fast when two ends
of a websocket disagree on whether they're in millimetres or metres. Keeping
the math on the server side, and never letting it leak to the browser, killed
a category of bugs.

## Coordinate system

```
        +Z (up)
         │
         │   camera at (az=0, el=0):
         │      (arm_radius, 0, base_height + camera_offset)
         │
         O ───────── +X
        / 
       /
     +Y
```

- **World frame:** right-handed. Origin at the centre of the turntable, on
  its top surface. `+Z` up, `+X` forward (towards the camera at `az=0`), `+Y`
  by right-hand rule.
- **Azimuth (AZ):** rotation of the platform around `+Z`. Float input, no
  end stops, wraps to `[0°, 360°)` for display.
- **Elevation (EL):** rotation of the camera arm. Signed, physical range
  `[−25°, +90°]`. `0°` = arm horizontal pointing along `+X`; `+90°` = arm
  vertical, camera looking straight down at the platform centre.
- **Camera pose:** at angles `(az, el)`, the camera sits at
  `R_z(az) · R_y(-el) · (arm_radius, 0, base_height + camera_offset)`
  with optical axis pointing back at the platform centre (you can override
  the look-at with a calibration if you do one).

`arm_radius`, `base_height`, `camera_offset` are **build parameters** you
measure on your physical machine and enter in the UI. They live in the
session and get persisted into each scan's `manifest.json` so the data is
self-contained.

## Data model — a scan session

A **scan session** is one JSON document on disk (`scans/<id>/manifest.json`):

```jsonc
{
  "id": "2026-05-29T14-22-08",
  "created": "2026-05-29T14:22:08Z",
  "updated": "2026-05-29T14:38:11Z",
  "machine_captured": false,        // true = automated sweep; false = manual session
  "build": {
    "arm_radius_mm": 220,
    "base_height_mm": 45,
    "camera_offset_mm": 80
  },
  "photos": [
    {
      "capture_id": "c_001",
      "az_deg": 0.0,
      "el_deg": 30.0,
      "pose_world": { /* 4x4 row-major */ },
      "timestamp": "2026-05-29T14:23:11Z",
      "file": "captures/c_001/photo.jpg"
    }
    /* ... */
  ],
  "notes": "...",
  "tags": []
}
```

Photo bytes go to a global `captures/` pool (immutable once written). The
manifest references them. This means deleting a session doesn't lose the
bytes; an explicit GC sweeps unreferenced captures.

## Subsystem boundaries

| Subsystem | Owns                                | Doesn't know about           |
|-----------|-------------------------------------|------------------------------|
| Firmware  | Steppers, encoders, closed-loop control | Photos, sessions, world frame |
| Server    | Scene graph, scan session, poses, photo files | Stepper microsteps, encoder bits |
| UI        | Rendering and user interaction      | Anything geometric           |

Each subsystem can be replaced independently. Swap the firmware for a
different MCU as long as the [API contract](API.md) holds. Swap the UI for a
CLI script — the server's `/ws/scene` is just one of several entry points.
