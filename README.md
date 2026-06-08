# Orbiter v0.1

A two-axis camera turntable for taking calibrated photo sets of small objects,
controlled over Wi-Fi from a browser. Think of it as a **bench-top hemisphere
camera rig** you can build on a weekend from parts that are easy to source.

This is **not a finished product**. It's a kit of working pieces — CAD,
firmware, server, web UI — packaged in one repo so other people can build
something similar without re-deriving everything from scratch. Pick what you
need, throw out the rest, adapt it to whatever you have on the shelf.

This README is the consolidated single-page reference. Per-subsystem deep
dives live in `docs/` and in each subsystem's own `README.md`; treat those as
authoritative when you need depth.

## Contents

1. [Project overview](#1-project-overview)
2. [Quick start](#2-quick-start)
3. [Architecture](#3-architecture)
4. [Coordinate system & data model](#4-coordinate-system--data-model)
5. [Bill of materials](#5-bill-of-materials)
6. [Wiring & pin map](#6-wiring--pin-map)
7. [Motion math](#7-motion-math)
8. [Printable parts & print profile](#8-printable-parts--print-profile)
9. [Assembly](#9-assembly)
10. [Firmware](#10-firmware)
11. [Server](#11-server)
12. [UI](#12-ui)
13. [Docker stack](#13-docker-stack)
14. [COLMAP integration](#14-colmap-integration)
15. [Troubleshooting](#15-troubleshooting)
16. [What's out of scope in v0.1](#16-whats-out-of-scope-in-v01)
17. [License](#17-license)
18. [Repository map](#18-repository-map)

---

## 1. Project overview

Orbiter is a bench-top hemisphere camera rig: an object sits on a rotating platform (azimuth axis), a camera arm sweeps over it (elevation axis), and a host server records each photo together with the (az, el) pose at the moment of capture. The output is a folder of photos with extrinsics, suitable for feeding straight into COLMAP as SfM priors.

v0.1 is a **lab kit**, not a finished product. Expect rough edges; the pieces work — we use them — but they were extracted from an active research project rather than polished for distribution. Issues and PRs welcome.

### What you get

| Layer       | What it does                                                                                  |
|-------------|-----------------------------------------------------------------------------------------------|
| **CAD**     | 3D-printable mechanical parts + full assembly (`.glb`, `.fbx`, `.obj`, individual `.stl`)     |
| **Firmware**| ESP32 sketch that exposes the table as a **generic 2-axis IP actuator** with encoder feedback |
| **Server**  | FastAPI service that owns scan-session state, drives the actuator, stores photos with poses   |
| **UI**      | Browser app (Vite + React + react-three-fiber) — live 3D view + photo library on a hemisphere |
| **COLMAP**  | Optional container + UI panel: feed your photo session into COLMAP with our poses as SfM priors |

### The "lab kit" framing

The hardware is whatever was within arm's reach during prototyping: A4988 drivers (cheap, forgiving, the silicon is everywhere), generic 1.2°/step NEMA-17 motors with a 1:10 GT2 belt reduction, AS5600 (I²C) on azimuth + AS5048A (SPI) on elevation (mixed because of migration history — two AS5600s if starting fresh), an old ESP32-D0WD-V3 dev board, and an Android phone running [IP Webcam](https://play.google.com/store/apps/details?id=com.pas.webcam) for the camera. Substitution points are called out in each section so you can swap parts.

### Three licenses, one repo

| What                                                          | License                          |
|---------------------------------------------------------------|----------------------------------|
| Source code (firmware, server, UI, scripts, configs)          | [Apache-2.0](LICENSE)            |
| Hardware designs (CAD, STL, FBX, GLB) in `cad/`               | [CERN-OHL-S-2.0](cad/LICENSE)    |
| Documentation in `docs/`                                      | [CC-BY-4.0](docs/LICENSE)        |

---

## 2. Quick start

The minimum-viable path from "I have the parts" to "I have a session of photos":

1. **Print the STLs** in `cad/parts/` per the [print profile](#8-printable-parts--print-profile).
2. **Wire up** the ESP32, drivers, motors, and encoders per [§6](#6-wiring--pin-map).
3. **Flash the firmware** ([§10.1](#101-build--flash)), set Wi-Fi credentials via `idf.py menuconfig`, note the IP that appears in the boot log.
4. **Calibrate the encoders** ([§10.3](#103-first-run-encoder-calibration)) — arm horizontal, platform facing forward, then `POST /calibrate {"axis":"el","mode":"current"}` and the same for AZ.
5. **Bring up the stack:**
   ```bash
   cd docker/
   cp .env.example .env             # set ESP32_IP and CAMERA_URL
   docker compose up
   ```
   Open <http://localhost:5173>.
6. **Point the UI at the phone** running IP Webcam — set the camera URL in the Camera config panel.
7. **Click "New session"** → click "Take shot" at each position you want → "Save". The manifest goes to disk with one pose per photo.

That's it. The session is now usable — browse it on the hemisphere in the Library tab, or export it for COLMAP ([§14](#14-colmap-integration)).

---

## 3. Architecture

Three tiers; the browser never talks to the ESP32 directly.

```
┌──────────────┐    WebSocket    ┌──────────────┐    HTTP/WS    ┌──────────────┐
│   Browser    │ ◄──────────────►│   Server     │ ◄────────────►│    ESP32     │
│  (Vite + R3F)│    /ws/scene    │  (FastAPI)   │  /move /state │  (firmware)  │
└──────────────┘                 └──────────────┘  /ws/log      └──────┬───────┘
                                       │                                │
                                       ▼                                ▼
                                IP Webcam (HTTP/JPEG)         encoders + steppers
```

**Server-side scene graph (Viser pattern).** The server is the source of truth for scene state and geometry. It holds the model (current AZ/EL, motors, scan session, captures), computes camera poses from `(az, el, arm_radius, base_height, camera_offset)`, proxies every firmware call, and writes images + manifests to disk. The browser is a thin viewer over a scene-graph diff protocol — the server pushes `scene_snapshot` then `scene_update` messages with added/updated/removed typed nodes; the browser maps `NodeType` → three.js objects and renders. Commands flow back as named operations (`move`, `take_shot`, `save_scan`, …).

**Why.** Off-by-unit / off-by-frame 3D math bugs compound very fast when two ends of a WebSocket disagree on whether they're in millimetres or metres, or which axis is up. Keeping the math on the server, and never letting it leak to the browser, killed that whole class of bugs. It's why the v0.1 codebase is small enough to extract and ship.

**Subsystem boundaries:**

| Subsystem | Owns                                            | Doesn't know about                  |
|-----------|-------------------------------------------------|-------------------------------------|
| Firmware  | Steppers, encoders, closed-loop control         | Photos, sessions, world frame       |
| Server    | Scene graph, scan session, poses, photo files   | Stepper microsteps, encoder bits    |
| UI        | Rendering and user interaction                  | Anything geometric                  |

The firmware presents itself as a **generic 2-axis IP actuator** — `POST /move {azimuth_deg, elevation_deg}`, `GET /state`. Drop the firmware on any 2-axis mechanism (gimbal, antenna rotator) and the server still talks to it.

---

## 4. Coordinate system & data model

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

- **World frame:** right-handed. Origin at the centre of the turntable, on its top surface. `+Z` up, `+X` forward (towards the camera at `az=0`), `+Y` by right-hand rule.
- **Azimuth (AZ):** platform rotation around `+Z`. Float input, no end stops, wraps to `[0°, 360°)` for display.
- **Elevation (EL):** arm rotation. Signed, physical range `[−25°, +90°]`. `0°` = arm horizontal pointing along `+X`; `+90°` = arm vertical, camera looking straight down.
- **Camera pose** at angles (az, el):

      world_pose = R_z(az) · R_y(-el) · (arm_radius, 0, base_height + camera_offset)

  with optical axis pointing back at the platform centre. A calibration can override the look-at.

`arm_radius`, `base_height`, `camera_offset` are **build parameters** you measure on your physical machine and enter in the UI. They are persisted into each scan's `manifest.json` so the data is self-contained.

**Scan manifest** at `scans/<id>/manifest.json`:

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
    // ...
  ],
  "notes": "...",
  "tags": []
}
```

Photo bytes go to a global **immutable** `captures/` pool. Manifests reference them by capture id. Deleting a scan does not delete the bytes — an explicit GC sweeps unreferenced captures.

---

## 5. Bill of materials

### 5.1 Electronics + motion

| Group              | Part                                    | What was used                                                                              | Sensible alternatives                                                                       |
|--------------------|-----------------------------------------|--------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|
| MCU                | ESP32 dev board                         | ESP32-D0WD-V3 rev 3.1, 4 MB flash                                                          | Any ESP32 with Wi-Fi (ESP32-WROOM, ESP32-S3 — pin map shifts).                              |
| Motor driver ×2    | A4988 stepper driver carrier            | Generic Pololu-style clone                                                                 | DRV8825 (same footprint, swap microstep table). TMC2208/2209 if you want quiet — same STEP/DIR/EN. |
| Stepper ×2         | NEMA-17                                 | 1.2°/step (300 step/rev), ~1 A                                                             | Standard 1.8°/step (200 step/rev) — adjust microstepping. Any NEMA-17 with enough torque for 1:10. |
| Encoder AZ         | AS5600 magnetic angle sensor breakout   | 12-bit, I²C @ 0x36                                                                         | AS5048B (I²C, 14-bit) drop-in for more resolution on AZ.                                    |
| Encoder EL         | AS5048A magnetic angle sensor breakout  | 14-bit, SPI Mode 1                                                                         | AS5600 if 0.088° resolution is enough.                                                      |
| Encoder magnet ×2  | Diametrically magnetised disc           | 6 mm × 2.5 mm N35 cylindrical                                                              | Anything diametric, ~5–8 mm × ~1.5–3 mm. Glue or press-fit into the back of the output pulley. |
| Belt ×2            | GT2 closed-loop                         | **`2M-348-6`** (2 mm pitch, 348 mm circumference, 6 mm width) — runs fine but is slightly long for this frame; a shorter belt would tension cleaner. | Any closed-loop GT2-compatible belt (2 mm pitch, 6 mm wide). Pick the next length down from 348 mm if you want less slack. |
| Pulley ×2          | GT2 10-tooth pinion + 100-tooth output  | 1:10 reduction                                                                             | Any pair giving 1:8 to 1:15.                                                                |
| Power supply       | 12 V, ≥ 2 A                             | Lab bench supply during dev                                                                | 12 V / 3 A "LED strip" brick for headless.                                                  |
| Step-down (logic)  | DC-DC buck (for headless)               | Feeds ESP32 `Vin` from the 12 V rail                                                       | MP1584 / LM2596 module.                                                                     |
| Camera             | Android phone + IP Webcam app           | Any source that exposes MJPEG/JPEG over HTTP                                               | Raspberry Pi + CSI cam + `mjpg-streamer`; USB webcam via `motion`.                          |

### 5.2 Mechanical + fasteners

| Group                    | What                                              | Qty       | Notes                                                                  |
|--------------------------|---------------------------------------------------|----------:|------------------------------------------------------------------------|
| Output shaft             | 5 mm round steel shaft cut to length              | 2         | One per axis.                                                          |
| Bearing                  | 608ZZ (5 mm bore × 22 mm OD × 7 mm) or 608-class  | 2         | Press-fit; one per output axis.                                        |
| Screws — frame to base   | M3 × 12 mm SHCS                                   | 8         |                                                                        |
| Screws — encoder mounts  | M3 × 8 mm                                         | 4         |                                                                        |
| Standoffs — encoders     | M3 nylon, 4 mm                                    | 4         | Sets the 1.5–2 mm gap between IC and magnet.                           |
| Screws — driver carriers | M2.5 × 6 mm                                       | as needed | A4988 boards to ESP32 board or perfboard.                              |
| Wiring sundries          | Dupont wires, screw terminals, perfboard          | —         |                                                                        |
| Camera mount             | Friction-grip phone holder                        | 1         | A car-mount holder works; mount at the **end of the arm** for longer parallax baseline. |

### 5.3 Frame (3D-printed)

PLA at 30 % infill is enough for a desk-bound rig. PETG / ABS if the lab is warm (PLA softens above ~55 °C). Per-part settings in [§8](#8-printable-parts--print-profile).

---

## 6. Wiring & pin map

GPIO numbers (not silkscreen) for the ESP32-D0WD-V3.

**Steppers (A4988):**

| Signal              | GPIO | Notes                                                                                                |
|---------------------|-----:|------------------------------------------------------------------------------------------------------|
| Stepper 1 STEP (AZ) |   25 |                                                                                                      |
| Stepper 1 DIR  (AZ) |   26 |                                                                                                      |
| Stepper 2 STEP (EL) |   27 |                                                                                                      |
| Stepper 2 DIR  (EL) |   33 | Moved off JTAG MTMS (GPIO 14), which floated to ~0.9 V at boot and clocked phantom steps. Avoid strapping / JTAG pins on your dev board. |
| ENABLE (shared)     |   32 | Active LOW. Wire to both drivers in parallel.                                                        |
| MS1 / MS2 / MS3     | 3.3V | 1/16 microstepping. Pull-up to VCC, no MCU control needed.                                           |

**Encoder AZ — AS5600 (I²C @ 400 kHz, address 0x36):**

| Signal | GPIO | Notes                                                |
|--------|-----:|------------------------------------------------------|
| SDA    |   21 | 4.7 kΩ pull-up to 3.3 V (often already on breakout). |
| SCL    |   22 | 4.7 kΩ pull-up to 3.3 V.                             |
| DIR    |  GND | Tie low for CCW = increasing angle. VCC to invert.   |

**Encoder EL — AS5048A (SPI Mode 1, manual chip select):**

| Signal | GPIO | Notes                                                       |
|--------|-----:|-------------------------------------------------------------|
| CLK    |   18 |                                                             |
| MISO   |   19 |                                                             |
| MOSI   |   23 | Chip reads commands over MOSI — don't omit.                 |
| CS     |   17 | Manual GPIO (we don't use the SPI peripheral's hardware CS). |

`GPIO 5` is free on this board (was CS for an older AS5048A on AZ).

**Wiring sketch:**

```
       12 V supply ─────┬──────┬──────────────────
                        │      │
                      A4988  A4988
                       AZ     EL
                  STEP─25     STEP─27
                   DIR─26      DIR─33
                  ENA─┴────────┴──32  (ESP32)
                        │      │
                    motor1   motor2

       AS5600  AS5048A
        SDA─21  CLK─18
        SCL─22  MISO─19
                MOSI─23
                CS  ─17

       USB ── ESP32 ── Wi-Fi to your network
```

ESP32 can be USB-powered for bench work. For headless deployment, feed `Vin` from the same 12 V rail through the step-down.

---

## 7. Motion math

| Parameter                       | Value     | Source                                            |
|---------------------------------|-----------|---------------------------------------------------|
| Motor step angle                | 1.2°      | Datasheet of the motor used (300 step/rev).       |
| Microstepping                   | 1/16      | A4988 with MS1/MS2/MS3 high.                      |
| Gear ratio                      | 1 : 10    | 10-tooth pinion / 100-tooth output pulley.        |
| Steps per output revolution     | 48 000    | 300 × 16 × 10.                                    |
| Degrees per microstep at output | 0.0075°   | 360 / 48 000.                                     |

If the motor is 1.8°/step (200 step/rev) with the same 1/16 microstep + 1:10 gearing, the math becomes 32 000 microsteps/rev or 0.01125°/microstep. Adjust in firmware `menuconfig`.

**Open-loop micro-step accuracy is a fiction at low currents.** The A4988 micro-step table assumes balanced phase currents; real ones drift. The firmware uses the magnetic encoders as the source of truth and runs a **P-loop in velocity mode** rather than counting steps. It will still drive open-loop if you skip the encoders — but the reported angle is then unreliable.

---

## 8. Printable parts & print profile

| Part                       | Qty | Material | Layer    | Infill | Supports             | Notes                                  |
|----------------------------|----:|----------|---------:|-------:|----------------------|----------------------------------------|
| `AxisFrame.stl`            |   1 | PLA      | 0.2 mm  | 30 %  | none                 | Arm pivot frame; flat on the bed.      |
| `OrbitFrame.stl`           |   1 | PLA      | 0.2 mm  | 30 %  | small (bearing seat) | Main vertical structure.               |
| `TableSupportFrame.stl`    |   1 | PLA      | 0.2 mm  | 25 %  | none                 | Base under the platform.               |
| `PlatformInsert.stl`       |   1 | PLA      | 0.2 mm  | 25 %  | none                 | Flat side down. Top surface.           |
| `BarHolderInsert.stl`      |   2 | PLA      | 0.2 mm  | 30 %  | none                 | Captures the camera arm.               |
| `GT2 Pulley.stl`           |   2 | PLA      | 0.15 mm | 40 %  | none                 | Slow down to keep teeth crisp.         |
| `Hall Sensor Mount.stl`    |   2 | PLA      | 0.2 mm  | 25 %  | none                 | Holds the encoder breakouts over the magnets. |

Source CAD ships as `cad/assembly/Orbiter.glb` (also `.fbx`, `.obj`); per-part STLs in `cad/parts/`; renders in `cad/screenshots/`. Model started in FreeCAD; the distribution format is the GLB. Edit individual STLs in Blender / Meshmixer for small tweaks; for structural changes (arm length, gear ratio) re-derive in a parametric tool and re-export.

---

## 9. Assembly

This is the order of operations. The CAD geometry is the source of truth — open `cad/assembly/Orbiter.glb` in Blender / your slicer to see how the pieces fit.

1. **Print everything.** Inspect for blobs / strings on the pulley teeth — they cost ~0.5° of backlash if you don't clean them up.
2. **Press-fit the bearings** into `OrbitFrame` and the base.
3. **Glue the diametric magnets** into the back of the 100-tooth output pulleys, centred as well as you can manage. Concentricity matters more than depth (within ~0.3 mm of the axis).
4. **Mount the encoder breakouts** on their printed seats with 4 mm nylon standoffs. Aim for a 1.5–2 mm gap between IC and magnet. If the AS5600 reports unstable angles, check concentricity and gap — its `MD/ML/MH` register reports magnet field status.
5. **Belt and pulleys** — install with the belt under modest tension. If teeth skip during slow moves, tighten; if motors stall, loosen.
6. **Wire it up** per [§6](#6-wiring--pin-map). Triple-check stepper coil pairs — getting one wrong reverses direction or causes high-current rattling.
7. **Flash the firmware** ([§10](#10-firmware)), set Wi-Fi, watch the boot log.
8. **Calibrate the encoders** with the arm horizontal and the platform facing forward ([§10.3](#103-first-run-encoder-calibration)).
9. **Start the server + UI** ([§13](#13-docker-stack)) and verify a manual move works end-to-end.

**Camera mounting.** The arm has slots for a generic phone holder (a friction-grip car-mount holder works fine). Mount the camera at the **end of the arm** — farther from the AZ pivot = longer parallax baseline for photogrammetry. Measure `arm_radius` (pivot → camera sensor along the arm), `base_height` (pivot above the platform surface), `camera_offset` (perpendicular distance from arm to camera) with callipers, and enter these in the UI. Eyeball numbers are fine for turntable-GIF use; measure carefully if you'll feed COLMAP as priors.

---

## 10. Firmware

**ESP-IDF v6** project at `firmware/`. ~1500 lines of C, builds on Windows / Linux / macOS.

### 10.1 Build & flash

```bash
cd firmware/
idf.py set-target esp32         # or esp32s3
idf.py menuconfig               # set Wi-Fi credentials + axis limits
idf.py -p COM3 flash monitor    # COM3 / /dev/ttyUSB0 / etc.
```

Headless / CI build (no host ESP-IDF install):

```bash
docker run --rm -v "$PWD:/project" -w /project \
    espressif/idf:release-v6.0 \
    idf.py build
```

A successful boot prints something like:

```
I (xxxx) wifi:connected → ip = 192.168.1.42
I (xxxx) encoder: AS5600 found at 0x36
I (xxxx) encoder: AS5048A SPI OK
I (xxxx) calibration loaded: az_zero=0.000° el_zero=138.296°
I (xxxx) http: server started on port 80
```

Write down the IP — the server needs it.

### 10.2 `menuconfig` settings

`idf.py menuconfig` → **Orbiter Configuration**:

| Symbol                  | Purpose                                  | Default       |
|-------------------------|------------------------------------------|---------------|
| `ORBITER_WIFI_SSID`     | 2.4 GHz SSID                             | — (required)  |
| `ORBITER_WIFI_PASSWORD` | Wi-Fi password                           | — (required)  |
| `ORBITER_WIFI_MAX_RETRY`| Connect retries before backoff           | 10            |
| `ORBITER_AZ_MAX_DEG`    | Soft upper bound on AZ (`0` = unbounded) | 0             |
| `ORBITER_EL_MIN_DEG`    | Lower limit on EL                        | -25           |
| `ORBITER_EL_MAX_DEG`    | Upper limit on EL                        | 90            |

Credentials are baked in at build time. `sdkconfig` is gitignored by default. For runtime provisioning (captive portal, BLE, SmartConfig) you'll need to add it — out of scope for v0.1.

### 10.3 First-run encoder calibration

Out of the box, encoder zeros are placeholders (`az_zero_raw = 0.0°`, `el_zero_raw = 138.296°`) — almost certainly wrong for your build. To fix:

1. Move the elevation arm to horizontal (spirit level helps). Arm points along `+X` (away from the user).
2. `curl -X POST http://<device-ip>/calibrate -H 'Content-Type: application/json' -d '{"axis":"el","mode":"current"}'`
3. Repeat for AZ with the platform facing forward.
4. `GET /state` should now report `el ≈ 0.0°`, `az ≈ 0.0°`.

Values are persisted to NVS and survive reboot.

### 10.4 HTTP API

| Method | Path            | Purpose                                                       |
|--------|-----------------|---------------------------------------------------------------|
| GET    | `/health`       | Liveness; device identity (chip, MAC, heap, uptime).          |
| GET    | `/state`        | Live angles, motor enable, motion state, calibration.         |
| POST   | `/move`         | Blocking drive to `azimuth_deg` / `elevation_deg` with `timeout_ms`. |
| POST   | `/spin`         | Continuous rotation: `axis` ∈ `az`/`el`, `rate_deg_per_s` (0 = stop). |
| POST   | `/spin/stop`    | Stop continuous rotation on both axes.                        |
| POST   | `/calibrate`    | Encoder zero offsets — `mode` ∈ `current` / `explicit` / `reset`. |
| POST   | `/zero`         | Back-compat alias for `/calibrate {"mode":"current"}`.        |
| POST   | `/motors`       | Enable / disable steppers.                                    |
| POST   | `/reboot`       | Restart firmware (200 first, reboot ~500 ms later).           |
| GET    | `/test/encoder` | Raw encoder reads — debugging.                                |
| POST   | `/test/jog`     | Open-loop jog for stepper bring-up.                           |
| WS     | `/ws/log`       | Streaming firmware events.                                    |

**Selected request bodies:**

```jsonc
// POST /move — omit an axis to leave it unchanged; AZ wraps, EL must be in [−25, +90]
{ "azimuth_deg": 90.0, "elevation_deg": 45.0, "timeout_ms": 10000 }

// POST /spin
{ "axis": "az", "rate_deg_per_s": 30.0 }     // start
{ "axis": "az", "rate_deg_per_s": 0   }      // stop

// POST /calibrate
{ "axis": "el",   "mode": "current" }
{ "axis": "az",   "mode": "explicit", "az_raw_deg": 0.0 }
{ "axis": "both", "mode": "reset" }
```

**Move responses:** `200` reached, `408` timeout (motors keep going), `409` another move in progress, `400` EL out of range.

**Arrival tolerance:** ± 1.0° on both axes (AS5600 step ≈ 0.088°, AS5048A step ≈ 0.022°, debounced over two readings).

**`/ws/log` frames:**

```json
{ "seq": 142, "ts_ms": 83400, "lvl": "I", "tag": "motion", "msg": "move done in 1840 ms — AZ=90.00° EL=45.00° OK" }
```

Up to 4 concurrent clients. Only operational events (`motion`, `http`, `encoder`) — **not** a mirror of `ESP_LOGx`.

### 10.5 What the firmware does not do

- No closed-loop trajectory tracking — point-to-point only.
- No homing routine — encoders are absolute.
- No on-device storage of photos; no on-device web UI.

Memory footprint at idle: ~720 KB flash, ~210 KB free heap. Comfortable on a stock ESP32 with 4 MB flash.

---

## 11. Server

FastAPI service at `server/orbiter_server/`. Owns scene state, proxies the ESP32, stores photos and manifests. Python ≥ 3.11.

### 11.1 Run

```bash
cd server/
pip install -e .
uvicorn orbiter_server.app:app --reload --port 8000
```

Or in Docker:

```bash
docker build -t orbiter-server .
docker run --rm -p 8000:8000 \
    -e ORBITER_ESP_IP=192.168.1.50 \
    -e ORBITER_CAMERA_URL=http://192.168.1.51:8080 \
    -v "$(pwd)/data:/data" \
    orbiter-server
```

### 11.2 HTTP / WS routes

| Method | Path                              | Notes                                                                 |
|--------|-----------------------------------|-----------------------------------------------------------------------|
| GET    | `/health`                         | Service heartbeat + resolved storage paths.                           |
| GET    | `/debug/model`                    | Read-only snapshot of the full `ModelState`.                          |
| GET    | `/config`                         | Persisted, config-like model fields.                                  |
| GET    | `/scans`                          | List stored scan summaries.                                           |
| POST   | `/scans`                          | Create a scan manifest.                                               |
| GET    | `/scans/{sid}`                    | Read a manifest.                                                      |
| DELETE | `/scans/{sid}`                    | Remove a scan manifest.                                               |
| POST   | `/scans/{sid}/archive`            | Mark archived (kept on disk).                                         |
| GET    | `/scans/{sid}/download`           | Zip of scan + photos.                                                 |
| POST   | `/scans/{sid}/sfm_priors`         | Write `sfm_priors.json` ([§14](#14-colmap-integration)).              |
| POST   | `/scans/{sid}/photos`             | Upload a photo for a scan.                                            |
| GET    | `/scans/{sid}/photos/{idx}/...`   | Thumbnails / full / meta.                                             |
| GET    | `/captures/{cid}/...`             | Capture pool (immutable).                                             |
| GET    | `/camera/stream.mjpeg`            | Live MJPEG preview re-multiplexed from the phone.                     |
| GET    | `/camera/stream/status`           | Camera connection status.                                             |
| WS     | `/ws/scene`                       | Scene + model diffs, command channel ([§12.3](#123-wire-protocol)).   |

### 11.3 Environment variables

| Variable                       | Default                | Purpose                                        |
|--------------------------------|------------------------|------------------------------------------------|
| `ORBITER_STORAGE_DIR`          | `./data`               | Root for `scans/` + `captures/`.               |
| `ORBITER_ESP_IP`               | `192.168.1.50`         | ESP32 firmware host.                           |
| `ORBITER_CAMERA_URL`           | empty                  | IP Webcam HTTP base URL (e.g. `http://phone-ip:8080`). |
| `ORBITER_PORT`                 | `8000`                 | Bound port.                                    |
| `ORBITER_CORS_ORIGINS`         | localhost:5173 / 5174  | Comma-separated allowed origins.               |
| `ORBITER_DEFAULT_CAMERA_PRESET`| `native`               | Pixel-rotation preset for new captures.        |

### 11.4 Data layout on disk

```
<ORBITER_STORAGE_DIR>/
  orbiter_state.json          # persisted ModelState subset
  scans/
    <scan_id>/
      manifest.json
      sfm_priors.json         # written by POST /sfm_priors
  captures/
    <capture_id>/
      original.jpg
      thumb.jpg               # medium tier
      thumb_small.jpg         # sidebar tier
      thumb_tiny.jpg          # in-scene texture tier
      meta.json
```

The capture pool is **immutable** — once `captures/<id>/original.jpg` exists, it doesn't change. Deleting a scan does not delete the bytes; a separate GC sweeps unreferenced captures.

---

## 12. UI

Vite + React + react-three-fiber + zustand + Radix Tabs. Two tabs: **Scaner** (live 3D, motion controls, motion planner, scan-session controls) and **Library** (saved scans, per-scan **Export SfM priors**, a placeholder **Run COLMAP** button reserved for v0.2).

### 12.1 Dev / build

```bash
cd ui/
npm install
npm run dev          # http://localhost:5173 (Vite proxies API to localhost:8000)
npx tsc --noEmit     # type-check without building
```

Production (Docker, two-stage `node:20-alpine` build → `nginx:alpine` serve on port 80):

```bash
docker build -t orbiter-v0.1-ui .
docker run --rm -p 8080:80 orbiter-v0.1-ui
```

### 12.2 Tabs

- **Scaner.** Single persistent WebSocket to `ws://server:8000/ws/scene`. Server pushes `scene_snapshot` + `scene_update` diffs (rendered via `nodeRegistry.ts`) and model state. Commands (`move`, `take_shot`, `start_scan`, …) go back over the same socket.
- **Library.** REST-only: `GET /scans`, `GET /config`, `POST /scans/{sid}/sfm_priors`, `DELETE /scans/{sid}`.

### 12.3 Wire protocol

Frame envelope: `{ t: string, seq: number, ts: number, data: T }`. Server → browser carries `scene_snapshot` and `scene_update` with `added` / `updated` / `removed` lists. Each node:

```ts
{
  id: string,
  parent: string | null,
  type: 'frame' | 'grid' | 'mesh' | 'line_segments' | 'point_cloud'
      | 'image_plane' | 'camera_frustum' | 'label'
      | 'cad_model' | 'cad_part' | 'disc_dial',
  transform: { position, quaternion, scale },
  visible: boolean,
  pickable: boolean,
  props: Record<string, unknown>  // type-specific, intentionally untyped
}
```

Patches carry only the changed fields. The browser does no trig — new rendering features mean a new node type in `protocol.ts` plus a server-side scene-graph builder, not new client-side math.

---

## 13. Docker stack

Compose lives at `docker/docker-compose.yml`. Three services, two profiles.

| Service | Build context | Image                | Host port |
|---------|---------------|----------------------|-----------|
| server  | `../server`   | `orbiter/server:dev` | 8000      |
| ui      | `../ui`       | `orbiter/ui:dev`     | 5173      |
| colmap  | `../colmap`   | `orbiter/colmap:dev` | —         |

### 13.1 First-time setup

```bash
cd docker/
cp .env.example .env             # set ESP32_IP, CAMERA_URL
docker compose up                # server + ui (colmap is profile-gated, see below)
```

Open <http://localhost:5173>.

### 13.2 COLMAP profile (opt-in)

```bash
docker compose --profile colmap run --rm colmap run_colmap_session.sh <sid>
docker compose --profile colmap run --rm colmap          # interactive shell at /data
```

### 13.3 Storage

Everything the server reads/writes lives under `./data` on the host (mounted as `/data` in both the `server` and `colmap` containers). `data/` is gitignored — wipe to start clean.

### 13.4 GPU passthrough

The compose stanza requests an NVIDIA GPU. Only effective with:

- **Linux native** — NVIDIA Container Toolkit installed.
- **Windows + WSL2** — Docker Desktop's WSL2 backend with NVIDIA drivers exposed.
- **macOS** — no passthrough; COLMAP runs CPU-only.

If unavailable, `run_colmap_session.sh` falls back to CPU SIFT automatically.

### 13.5 Camera URL inside containers

`localhost` from inside a container is the container, not your phone. Use the phone's **LAN IP** in `CAMERA_URL`.

---

## 14. COLMAP integration

A scan session — a folder of photos with extrinsics — is a near-perfect input for COLMAP: hand it the photos *and* the camera positions and it can skip pose estimation, going straight to triangulation and dense reconstruction.

### 14.1 Two ways to drive it

**Option A — containerised.**

```bash
cd docker/
docker compose --profile colmap run --rm colmap \
    run_colmap_session.sh <session-id>             # CPU SIFT (default)
docker compose --profile colmap run --rm colmap \
    run_colmap_session.sh <session-id> --gpu       # GPU SIFT
docker compose --profile colmap run --rm colmap \
    run_colmap_session.sh <session-id> --dry-run   # print command plan, no work
```

Output ends up in `<storage>/scans/<sid>/colmap/`. Progress is streamed back to the UI.

**Option B — hand-off.** UI → Library → session → **Export → SfM priors** writes `<storage>/scans/<sid>/sfm_priors.json`. Feed it to your own COLMAP install; the conversion logic is in `server/orbiter_server/sfm_export.py`.

### 14.2 `sfm_priors.json` schema

```jsonc
{
  "schema": "orbiter.sfm_priors.v1",
  "camera_intrinsics": {
    "model": "PINHOLE",
    "width":  1920, "height": 1080,
    "fx": 1500, "fy": 1500,
    "cx":  960, "cy":  540
  },
  "images": [
    {
      "file": "c_001/photo.jpg",
      "qw":  0.707, "qx": 0, "qy": 0.707, "qz": 0,   // Hamilton quaternion
      "tx":   220,  "ty": 0, "tz":  45               // translation in mm
    }
    // ...
  ]
}
```

Quaternion convention: **Hamilton** (w, x, y, z). Translations in **millimetres** in the world frame from [§4](#4-coordinate-system--data-model). The transform takes world points into camera space (COLMAP's convention).

### 14.3 Container pipeline

The wrapper executes seven steps, halting on the first error:

| # | Step                              | Notes                                                  |
|--:|-----------------------------------|--------------------------------------------------------|
| 1 | `sfm_priors_to_colmap.py`         | Convert priors JSON → COLMAP text model.               |
| 2 | `colmap feature_extractor`        | SIFT; `--gpu` enables `SiftExtraction.use_gpu`.        |
| 3 | `colmap exhaustive_matcher`       | Pairwise match.                                        |
| 4 | `colmap point_triangulator`       | Uses the prior sparse as input.                        |
| 5 | `colmap image_undistorter`        | Sparse → dense workspace.                              |
| 6 | `colmap patch_match_stereo`       | Per-image depth maps (slow on CPU; huge GPU speedup).  |
| 7 | `colmap stereo_fusion`            | Depth maps → fused `.ply` point cloud.                 |

Output layout under `<storage>/scans/<sid>/colmap/`: `sparse_priors/`, `database.db`, `sparse/0/`, `dense/{images,sparse,stereo,fused.ply}`.

### 14.4 How accurate are the priors?

With a calliper-measured arm and the standard AS5600 / AS5048A encoder pair:

| Quantity                | Typical    |
|-------------------------|------------|
| Per-shot rotation error | 0.5° – 1.5° |
| Per-shot position error | 2 – 10 mm  |

Not enough for "feature-free" reconstruction, but plenty as a warm start — COLMAP's bundle adjustment will polish them. Camera intrinsics are guessed from the IP Webcam stream by default; override in **Camera config** before exporting if you've calibrated separately.

---

## 15. Troubleshooting

| Symptom                                                  | Likely cause                                              | Fix                                                                              |
|----------------------------------------------------------|-----------------------------------------------------------|----------------------------------------------------------------------------------|
| Motors hum but don't move                                | Driver `EN` floating, or `STEP` not toggling.             | Probe with a scope. Check GPIO 32 wiring.                                        |
| Drivers get hot fast                                     | Current trimpot too high.                                 | Aim for ~70 % of rated motor current.                                            |
| Reported AZ jumps randomly                               | Magnet not concentric, or stray field nearby.             | Move encoder away from motors; check AS5600 `MD/ML/MH` register.                 |
| EL keeps spinning past `90°`                             | Encoder zero off, or magnet flipped.                      | Re-run calibration.                                                              |
| Belts squeal                                             | Too tight.                                                | Loosen tensioner.                                                                |
| Stepper skips at speed                                   | Too fast for current setting.                             | Reduce `rate_deg_per_s` or increase driver current.                              |
| `sfm_priors.json missing` (COLMAP)                       | Priors weren't exported.                                  | UI → Library → session → Export → SfM priors.                                    |
| `feature_extractor` aborts with `cudaError`              | `--gpu` set but no usable GPU.                            | Re-run without `--gpu`, or fix host GPU passthrough.                             |
| `point_triangulator` registers 0 images                  | Filenames in priors don't match disk.                     | Check `file` fields vs the layout under the session dir.                         |
| `patch_match_stereo` runs forever on CPU                 | Expected on CPU.                                          | Use `--gpu`, reduce image count, or accept the wait.                             |
| `stereo_fusion` near-empty `.ply`                        | Too few overlapping views, or noisy priors.               | Add more shots (small angular gaps); verify priors with `--dry-run`.             |
| Out-of-memory mid-PatchMatch                             | Docker Desktop's RAM cap.                                 | Bump RAM in Docker Desktop → Settings → Resources, or downsample.                |
| `permission denied` writing `/data/.../colmap`           | Host `data/` owned by a different UID; container is root. | `chown` the host dir, or wipe and let the container recreate.                    |
| Camera fetch fails from inside container                 | `CAMERA_URL` points at `localhost`.                       | Use the phone's LAN IP, not `localhost`.                                         |

For deeper firmware problems: the firmware log is verbose by design — watch it via `/ws/log` (the server forwards firmware frames as `log` messages over `/ws/scene` to the UI's log panel).

---

## 16. What's out of scope in v0.1

The full Orbiter project (in the parent repo) includes more — left out of v0.1 on purpose:

- **Laser-stripe scanner + live triangulator.** Needs a line laser and a separately calibrated camera; the parent project couples both tightly to its own optical stack.
- **Photogrammetry job orchestration.** Beyond the single-session COLMAP wrapper that ships here.

ChArUco hand-eye geometry calibration *is* shipped here — see [§5](#5-bill-of-materials) for the board and the **Machine config → Calibrate from board** flow in the UI. Per-shot residuals reach ~0.5°/few mm with a calliper-printed board.

The architecture leaves clean seams where these can plug back in — the server already speaks in terms of `Manifest`, `MotionPlan`, `Pose`. The UI's Library tab carries a disabled **Run COLMAP** button reserved for v0.2.

---

## 17. License

| What                                                    | License                          |
|---------------------------------------------------------|----------------------------------|
| Source code (firmware, server, UI, scripts, configs)    | [Apache-2.0](LICENSE)            |
| Hardware designs (CAD, STL, FBX, GLB) in `cad/`         | [CERN-OHL-S-2.0](cad/LICENSE)    |
| Documentation in `docs/`                                | [CC-BY-4.0](docs/LICENSE)        |

Files outside `cad/` and `docs/` are covered by the root `LICENSE` (Apache-2.0).

---

## 18. Repository map

```
Orbiter/
├── README.md               ← you are here
├── LICENSE                 ← Apache-2.0 (default for source code)
├── docs/                   ← per-subsystem deep dives + license
│   └── LICENSE             ← CC-BY-4.0 (docs)
├── cad/                    ← CAD (assembly + per-part STLs + screenshots)
│   └── LICENSE             ← CERN-OHL-S-2.0 (hardware)
├── firmware/               ← ESP-IDF project — the 2-axis IP actuator
├── server/                 ← FastAPI — scene state + ESP proxy + storage
├── ui/                     ← Vite + React + react-three-fiber — viewer & library
├── docker/                 ← docker-compose stack
└── colmap/                 ← COLMAP container + integration notes
```

| Top-level file / dir | One-line role                                                                          |
|----------------------|----------------------------------------------------------------------------------------|
| `README.md`          | This file — consolidated single-page reference.                                        |
| `firmware/`          | ESP-IDF v6 source for the 2-axis IP actuator.                                          |
| `server/`            | FastAPI app, `ModelState`, scene-graph builder, scan orchestration, COLMAP-priors exporter. |
| `ui/`                | Single-page React app over the scene-graph WebSocket.                                  |
| `cad/`               | `.glb`/`.fbx`/`.obj` assembly + per-part `.stl` files + rendered previews.             |
| `docker/`            | `docker-compose.yml` with three services and a `colmap` profile.                       |
| `colmap/`            | Thin container over `colmap/colmap:latest` + an SfM-priors → COLMAP-text converter.    |
| `docs/`              | Per-subsystem deep dives (architecture, hardware, firmware, API, assembly, COLMAP).    |
