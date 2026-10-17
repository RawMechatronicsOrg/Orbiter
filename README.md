# Orbiter v0.3

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

**New in v0.3**

- **`native/` — a desktop CV workbench** for a binocular camera pair
  ([`native/README.md`](native/README.md)): per-eye intrinsics and stereo
  calibration guided stage by stage, a laser-stripe scanner that tracks the
  ChArUco board and accumulates a coloured point cloud, a rig check, a photo
  session shot alongside the scan, and `orbiter-recon` — a textured mesh
  from the laser cloud and the photographs, or the COLMAP 4 dense chain fed
  with our poses and the stripe masked out.
- **Stereo tab** in the web UI, with the server routes behind it: the
  pair's `camserver`, which camera is which eye, per-eye orientation, and
  a nominal baseline that a stereo solve in the workbench later replaces
  with the measured one.
- **`cad/v2/` — the second-generation mechanics** as STL, STEP and FreeCAD
  ([`cad/v2/README.md`](cad/v2/README.md)): two near-identical rotary
  drive modules on an aluminium tube frame, 6008 bearings, taper inserts,
  an AS5600 on both axes. The server's rig model, the 3D viewer and this
  README follow v2 (its GLB is `cad/v2/assembly/Orbiter.glb`).
- **Firmware** reads an AS5600 on each axis (one I²C bus per sensor), takes
  its stepper pins from a **Board wiring** choice in `menuconfig` — the
  jumper-wire bench or the Orbiter-Simple carrier — and updates over Wi-Fi
  with `POST /ota` ([§10.1](#101-build--flash)).
- The docker-compose stack starts on a fresh checkout.

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
16. [What's out of scope](#16-whats-out-of-scope)
17. [License](#17-license)
18. [Repository map](#18-repository-map)

---

## 1. Project overview

Orbiter is a bench-top hemisphere camera rig: an object sits on a rotating platform (azimuth axis), a camera arm sweeps over it (elevation axis), and a host server records each photo together with the (az, el) pose at the moment of capture. The output is a folder of photos with extrinsics, suitable for feeding straight into COLMAP as SfM priors.

Orbiter is a **lab kit**, not a finished product. Expect rough edges; the pieces work — we use them — but they were extracted from an active research project rather than polished for distribution. Issues and PRs welcome.

### What you get

| Layer       | What it does                                                                                  |
|-------------|-----------------------------------------------------------------------------------------------|
| **CAD**     | v2 (current): STL, STEP, FreeCAD and the GLB the software reads, in `cad/v2/`. v1: the first rig's printed parts + assembly in `cad/` |
| **Firmware**| ESP32 sketch that exposes the table as a **generic 2-axis IP actuator** with encoder feedback |
| **Server**  | FastAPI service that owns scan-session state, drives the actuator, stores photos with poses   |
| **UI**      | Browser app (Vite + React + react-three-fiber) — live 3D view + photo library on a hemisphere + the stereo rig settings |
| **Native**  | Desktop CV workbench (PySide6) for a binocular pair: calibration, laser-stripe scanning, photo capture, `orbiter-recon` dense chain |
| **COLMAP**  | Optional container + UI panel: feed your photo session into COLMAP with our poses as SfM priors |

### The "lab kit" framing

The hardware is whatever was within arm's reach during prototyping: A4988 drivers (cheap, forgiving, the silicon is everywhere), 1.8°/step NEMA-17 motors (24 mm body) on GT2 belts — 7:1 on azimuth, 10:1 on elevation — a 25 × 25 mm aluminium tube frame with printed drive housings on 6008 bearings, an AS5600 (I²C) on each axis, an old ESP32-D0WD-V3 dev board, and an Android phone running [IP Webcam](https://play.google.com/store/apps/details?id=com.pas.webcam) for the camera. Substitution points are called out in each section so you can swap parts.

### Three licenses, one repo

| What                                                          | License                          |
|---------------------------------------------------------------|----------------------------------|
| Source code (firmware, server, UI, scripts, configs)          | [Apache-2.0](LICENSE)            |
| Hardware designs (CAD, STL, FBX, GLB) in `cad/`               | [CERN-OHL-S-2.0](cad/LICENSE)    |
| Documentation in `docs/`                                      | [CC-BY-4.0](docs/LICENSE)        |

---

## 2. Quick start

The minimum-viable path from "I have the parts" to "I have a session of photos":

1. **Print the STLs** in `cad/v2/stl/printed/` ([§8](#8-printable-parts--print-profile)).
2. **Wire up** the ESP32, drivers, motors, and encoders per [§6](#6-wiring--pin-map).
3. **Flash the firmware** ([§10.1](#101-build--flash)), set Wi-Fi credentials and the board wiring via `idf.py menuconfig`, note the IP that appears in the boot log.
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

**Why.** Off-by-unit / off-by-frame 3D math bugs compound very fast when two ends of a WebSocket disagree on whether they're in millimetres or metres, or which axis is up. Keeping the math on the server, and never letting it leak to the browser, killed that whole class of bugs. It's why the kit codebase is small enough to extract and ship.

**Subsystem boundaries:**

| Subsystem | Owns                                            | Doesn't know about                  |
|-----------|-------------------------------------------------|-------------------------------------|
| Firmware  | Steppers, encoders, closed-loop control         | Photos, sessions, world frame       |
| Server    | Scene graph, scan session, poses, photo files, the stereo rig baseline | Stepper microsteps, encoder bits |
| UI        | Rendering and user interaction                  | Anything geometric                  |
| Native    | Pixels: detectors, calibration solves, stripe triangulation, point clouds, photo sessions | The turntable's motion |

The firmware presents itself as a **generic 2-axis IP actuator** — `POST /move {azimuth_deg, elevation_deg}`, `GET /state`. Drop the firmware on any 2-axis mechanism (gimbal, antenna rotator) and the server still talks to it.

**The native workbench sits beside the browser, not behind it.** The browser can display a camera stream but cannot read its pixels (a cross-origin `<img>` taints the canvas), so the frame-rate computer vision for the binocular pair runs in a desktop app: it reads the rig the Stereo tab set (`GET /config`), pulls MJPEG straight from the cameras' `camserver`, and keeps its results on disk and in the server's config — a stereo solve there replaces the nominal baseline with the measured one. See [§12.4](#124-native-cv-workbench).

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
| Stepper ×2         | NEMA-17, 24 mm body                     | 1.8°/step (200 step/rev)                                                                   | Any NEMA-17 that fits the mounts; the step angle changes only the step rate per degree ([§7](#7-motion-math)). |
| Encoder ×2         | AS5600 magnetic angle sensor module     | 12-bit, I²C @ 0x36                                                                         | AS5048B (I²C, 14-bit) for more resolution — it would need a driver; the firmware talks to the AS5600 only. |
| Encoder magnet ×2  | Diametrically magnetised disc           | Sits in a printed `HexPlug` in the end of the spindle and of the shaft                     | Anything diametric that fits the plug.                                                      |
| Belts              | GT2 closed loop, 6 mm wide              | 280 mm on AZ; EL ≈ 347 mm at the modelled centre distance (the model names no length)      | Any GT2-compatible belt (2 mm pitch, 6 mm wide).                                            |
| Pulleys            | GT2 16-tooth on the motors              | Printed 112-tooth (AZ, 7:1) and 160-tooth (EL, 10:1) output pulleys on splines             | Reprint the output pulley for another ratio.                                                |
| Power supply       | 12 V, ≥ 2 A                             | Lab bench supply during dev                                                                | 12 V / 3 A "LED strip" brick for headless.                                                  |
| Step-down (logic)  | DC-DC buck (for headless)               | Feeds ESP32 `Vin` from the 12 V rail                                                       | MP1584 / LM2596 module.                                                                     |
| Camera             | Android phone + IP Webcam app           | Any source that exposes MJPEG/JPEG over HTTP                                               | Raspberry Pi + CSI cam + `mjpg-streamer`; USB webcam via `motion`.                          |

### 5.2 Mechanical + fasteners

The full list with quantities is [`cad/v2/parts.csv`](cad/v2/parts.csv).

| Group                    | What                                              | Qty       | Notes                                                                  |
|--------------------------|---------------------------------------------------|----------:|------------------------------------------------------------------------|
| Frame                    | 25 × 25 mm aluminium square tube                  | 4         | 3 × 253 mm (main rail, cross rail, orbit arm), 1 × 93 mm (vertical bar). |
| Bearing                  | 6008-2Z (40 × 68 × 15 mm)                         | 4         | Two per axis, clamped by a printed M32×1 nut.                          |
| Socket-head screws M5    | ISO 4762 × 10 / 12 / 16 / 25 / 30 / 35 / 40 / 70 mm | 3 / 1 / 3 / 1 / 1 / 2 / 2 / 1 | The 70 mm one is the main stack bolt through the vertical bar. |
| Socket-head screws M3    | ISO 4762 M3 × 8                                   | 8         | Motor mounts.                                                          |
| Nuts and washers M5      | ISO 4032 nut / ISO 4035 thin nut / ISO 7093 washer 15 × 1.2 | 6 / 1 / 6 |                                                              |
| Heat-set inserts         | M5 × 7                                            | 7         |                                                                        |
| Wiring sundries          | Dupont wires, screw terminals, perfboard          | —         |                                                                        |
| Camera mount             | Phone holder clamped to the arm                   | 1         | Mount at the **end of the arm** for the longest parallax baseline.     |

### 5.3 Frame (3D-printed)

PLA is enough for a desk-bound rig; PETG / ABS if the lab is warm (PLA softens above ~55 °C). The parts are in [§8](#8-printable-parts--print-profile).

---

## 6. Wiring & pin map

GPIO numbers (not silkscreen) for the ESP32-D0WD-V3.

**Steppers (A4988):** two wirings, picked at build time in `idf.py menuconfig` → Orbiter Configuration → **Board wiring** ([§10.2](#102-menuconfig-settings)).

| Signal              | `ORBITER_BOARD_WIRED` (default) | `ORBITER_BOARD_SIMPLE` | Notes                                                                                                |
|---------------------|--------------------------------:|-----------------------:|------------------------------------------------------------------------------------------------------|
| Stepper 1 STEP (AZ) |                              25 |                     26 |                                                                                                      |
| Stepper 1 DIR  (AZ) |                              26 |                     25 |                                                                                                      |
| Stepper 2 STEP (EL) |                              27 |                     18 |                                                                                                      |
| Stepper 2 DIR  (EL) |                              33 |                     19 | On the wired bench, moved off JTAG MTMS (GPIO 14), which floated to ~0.9 V at boot and clocked phantom steps. Avoid strapping / JTAG pins on your dev board. |
| ENABLE (shared)     |                              32 |                     27 | Active LOW. Wire to both drivers in parallel.                                                        |

`ORBITER_BOARD_WIRED` is the jumper-wire bench (A4988 carriers on jumper wires). `ORBITER_BOARD_SIMPLE` is the Orbiter-Simple carrier PCB. The firmware does not drive MS1 / MS2 / MS3; on the wired bench they are tied to 3.3 V for 1/16 microstepping (pull-up to VCC).

On Orbiter-Simple the load outputs OUT A–D (GPIO 12, 2, 5, 17) and the DSLR remote FOCUS / SHUTTER (GPIO 32, 33) are driven low at boot, before anything else, so a floating input cannot switch a 12 V load or trip the camera. The firmware does not use them otherwise. On the wired bench GPIO 32 / 33 are ENABLE and EL DIR, so this is not done there.

**Encoders — AS5600 on both axes (I²C @ 400 kHz, address 0x36):**

The AS5600's address is fixed, so each sensor needs its own I²C bus. The pins are the same on both stepper wirings.

| Axis | Signal | GPIO | Notes                                                |
|------|--------|-----:|------------------------------------------------------|
| AZ   | SDA    |   22 | 4.7 kΩ pull-up to 3.3 V (often already on breakout). |
| AZ   | SCL    |   23 | 4.7 kΩ pull-up to 3.3 V.                             |
| EL   | SDA    |    4 | 4.7 kΩ pull-up to 3.3 V (often already on breakout). |
| EL   | SCL    |   16 | 4.7 kΩ pull-up to 3.3 V.                             |
| both | DIR    |  GND | On the sensor board, not an ESP32 pin. **Strap it to GND** — a floating DIR makes the angle jump. |

Both sensors are 12-bit: 4096 counts per turn, 0.088° per count.

**Wiring sketch** (stepper pins as on `ORBITER_BOARD_WIRED`; for Orbiter-Simple use the table above):

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

       AS5600 (AZ)      AS5600 (EL)
        SDA─22           SDA─4
        SCL─23           SCL─16
        DIR─GND          DIR─GND

       USB ── ESP32 ── Wi-Fi to your network
```

ESP32 can be USB-powered for bench work. For headless deployment, feed `Vin` from the same 12 V rail through the step-down.

---

## 7. Motion math

| Parameter                       | Azimuth   | Elevation | Source                                            |
|---------------------------------|-----------|-----------|---------------------------------------------------|
| Motor step angle                | 1.8°      | 1.8°      | Motor datasheet (200 step/rev).                   |
| Microstepping                   | 1/16      | 1/16      | A4988 with MS1/MS2/MS3 high.                      |
| Belt reduction                  | 7 : 1     | 10 : 1    | GT2 16-tooth motor pulley → 112 / 160-tooth output pulley. |
| Microsteps per output rev       | 22 400    | 32 000    | 200 × 16 × 7, 200 × 16 × 10.                      |
| Degrees per microstep at output | 0.0161°   | 0.01125°  | 360 / 22 400, 360 / 32 000.                       |
| Travel                          | endless   | −25° … +90° | The arm meets the cross rail at −28.8° in the v2 model. |

The firmware does not use these numbers — it closes the loop on the encoders — but its speed caps are step rates: the default 2100 Hz on AZ is about 33.8 °/s at the platform, 350 Hz on EL about 3.9 °/s at the arm. The geometry the server uses (axes, platform, arm) is in [docs/HARDWARE.md](docs/HARDWARE.md#mechanics).

**Open-loop micro-step accuracy is a fiction at low currents.** The A4988 micro-step table assumes balanced phase currents; real ones drift. The firmware uses the magnetic encoders as the source of truth and runs a **P-loop in velocity mode** rather than counting steps. It will still drive open-loop if you skip the encoders — but the reported angle is then unreliable.

---

## 8. Printable parts & print profile

The v2 rig has 25 printed files, 32 pieces (three of them are tools), about 830 g of PLA at 100 % infill. The list with colour, quantity, thread, mass and role of each part is in [`cad/v2/README.md`](cad/v2/README.md#printed-parts); STL in `cad/v2/stl/printed/<colour>/`, STEP in `cad/v2/step/printed/<colour>/`, FreeCAD files in `cad/v2/freecad/`. There is no per-part print profile (orientation, infill, supports) for v2 yet.

The assembly the software reads is [`cad/v2/assembly/Orbiter.glb`](cad/v2/assembly), exported from the Fusion model's Blender scene by [`cad/v2/export_glb.py`](cad/v2/export_glb.py). For structural changes (arm length, pulley ratio) change the CAD and export the GLB again — the server takes the axes, the platform and the arm from it.

The first rig (printed frames, 608 bearings, 1:10 belts) stays in [`cad/`](cad/README.md) for reference.

---

## 9. Assembly

There is no step-by-step guide for v2 yet. This is the order of operations; the CAD is the source of truth — open `cad/v2/assembly/Orbiter.glb` or `cad/v2/freecad/Orbiter_assembly.FCStd` to see how the pieces fit, and the section view in [`cad/v2/README.md`](cad/v2/README.md) for the drives.

1. **Print everything.** Inspect for blobs / strings on the pulley teeth — they cost backlash if you don't clean them up.
2. **Build the two drives**: two 6008 bearings per housing with the spacers between the rings, the spindle / shaft through them, clamped by the M32×1 nut (the printed AF32 wrench turns it). The output pulley goes on the splines.
3. **Press the diametric magnets** into the two `HexPlug`s and the plugs into the end of the spindle and of the shaft.
4. **Mount the AS5600 modules** on `BaseSensorRing` (AZ) and `AngleSensorMount Ribbed` (EL), about 1 mm from the plug. If the AS5600 reports unstable angles, check centring and gap — its `MD/ML/MH` register reports magnet field status — and that its `DIR` pin is strapped to GND (a floating `DIR` makes the angle jump).
5. **Belt and pulleys** — install with the belt under modest tension. If teeth skip during slow moves, tighten; if motors stall, loosen.
6. **Wire it up** per [§6](#6-wiring--pin-map). Triple-check stepper coil pairs — getting one wrong reverses direction or causes high-current rattling.
7. **Flash the firmware** ([§10](#10-firmware)), set Wi-Fi and the board wiring, watch the boot log.
8. **Calibrate the encoders** with the arm horizontal and the platform facing forward ([§10.3](#103-first-run-encoder-calibration)).
9. **Start the server + UI** ([§13](#13-docker-stack)) and verify a manual move works end-to-end.

**Camera mounting.** Clamp a phone holder to the 25 × 25 mm arm (a friction-grip car-mount holder works fine). Mount the camera at the **end of the arm** — the tip is 225 mm from the elevation axis; farther from the AZ pivot = longer parallax baseline for photogrammetry. Measure `arm_radius` (pivot → camera sensor along the arm), `base_height` (pivot above the platform surface), `camera_offset` (perpendicular distance from arm to camera) with callipers, and enter these in the UI. Eyeball numbers are fine for turntable-GIF use; measure carefully if you'll feed COLMAP as priors.

---

## 10. Firmware

**ESP-IDF v6** project at `firmware/`. ~1500 lines of C, builds on Windows / Linux / macOS.

### 10.1 Build & flash

```bash
cd firmware/
idf.py set-target esp32         # or esp32s3
idf.py menuconfig               # set Wi-Fi credentials, board wiring, axis limits
idf.py -p COM3 flash monitor    # COM3 / /dev/ttyUSB0 / etc.
```

The first flash of a board goes over USB. After that, update over Wi-Fi ([below](#updating-over-wi-fi)).

Headless / CI build (no host ESP-IDF install):

```bash
docker run --rm -v "$PWD:/project" -w /project \
    espressif/idf:release-v6.0 \
    idf.py build
```

A successful boot prints something like:

```
I (xxxx) wifi:connected → ip = 192.168.1.42
I (xxxx) encoder: init — AS5600(AZ) SDA=22 SCL=23 | AS5600(EL) SDA=4 SCL=16
I (xxxx) calibration loaded: az_zero=0.000° el_zero=138.296°
I (xxxx) http: server started on port 80
```

Write down the IP — the server needs it.

#### Updating over Wi-Fi

Once a board runs this firmware, `POST /ota` replaces it without a cable. Build, then send the raw app image as the request body:

```bash
cd firmware/
idf.py build
curl --data-binary @build/orbiter.bin http://<esp-ip>/ota
```

| What                | Behaviour                                                                                                    |
|---------------------|--------------------------------------------------------------------------------------------------------------|
| Flash layout        | 4 MB flash, two OTA slots ([`firmware/partitions.csv`](firmware/partitions.csv)). `nvs` stays at `0x9000` / `0x6000`, so calibration survives. |
| Refused             | An image from another project (`400`), an empty body (`400`), an image larger than the slot (`413`), and any request while the axes move (`409`). |
| While writing       | The motors are disabled; on success the device reboots into the new image.                                   |
| Rollback            | A new image confirms itself once its HTTP server is up. If that does not happen within 120 s, the bootloader goes back to the previous image. |

The first flash onto a board that still has the old single-app layout **must** go over USB (`idf.py -p COM3 flash`), because the partition table changes. Some ESP32 dev-board clones do not enter download mode by themselves: `esptool` fails with `Wrong boot mode detected (0x17)`. Hold the **BOOT** button while `esptool` prints `Connecting...`. From then on use `/ota`. The response codes are in [docs/API.md](docs/API.md#post-ota).

### 10.2 `menuconfig` settings

`idf.py menuconfig` → **Orbiter Configuration**:

| Symbol                  | Purpose                                  | Default       |
|-------------------------|------------------------------------------|---------------|
| **Board wiring** (choice) | Which stepper pins the firmware drives: `ORBITER_BOARD_WIRED` — the jumper-wire bench (AZ STEP/DIR 25/26, EL STEP/DIR 27/33, ENABLE 32); `ORBITER_BOARD_SIMPLE` — the Orbiter-Simple carrier (AZ 26/25, EL 18/19, ENABLE 27). Encoder pins are the same on both ([§6](#6-wiring--pin-map)). | `ORBITER_BOARD_WIRED` |
| `ORBITER_WIFI_SSID`     | 2.4 GHz SSID                             | — (required)  |
| `ORBITER_WIFI_PASSWORD` | Wi-Fi password                           | — (required)  |
| `ORBITER_WIFI_MAX_RETRY`| Connect retries before backoff           | 10            |
| `ORBITER_EL_MIN_DEG`    | Lower limit on EL                        | -25           |
| `ORBITER_EL_MAX_DEG`    | Upper limit on EL                        | 90            |
| `ORBITER_AS5600_I2C_TRACE` | Log every I²C register read of both AS5600 sensors at INFO, with retries and error codes (verbose; rebuild after changing) | off |

Credentials are baked in at build time. `sdkconfig` is gitignored by default. For runtime provisioning (captive portal, BLE, SmartConfig) you'll need to add it — out of scope for this kit.

### 10.3 First-run encoder calibration

Out of the box, encoder zeros are placeholders (`az_zero_raw = 0.0°`, `el_zero_raw = 138.296°`) — almost certainly wrong for your build. To fix:

1. Move the elevation arm to horizontal (spirit level helps). Arm points along `+X` (away from the user).
2. `curl -X POST http://<device-ip>/calibrate -H 'Content-Type: application/json' -d '{"axis":"el","mode":"current"}'`
3. Repeat for AZ with the platform facing forward.
4. `GET /state` should now report `el ≈ 0.0°`, `az ≈ 0.0°`.

Values are persisted to NVS and survive reboot.

**After swapping a sensor,** calibrate that axis again. For EL also check the sign on the rig: raise the arm and `el` in `GET /state` must increase. The firmware expects the EL sensor's raw reading to *decrease* as the arm rises and negates it; an AS5600 reverses with its DIR strap and the magnet side (see `el_user_from_raw` in `firmware/main/motion.c`).

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
| POST   | `/ota`          | Firmware update: the body is the raw app image (`build/orbiter.bin`); reboots into it ([§10.1](#updating-over-wi-fi)). |
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

**Arrival tolerance:** ± 1.0° on both axes (AS5600 step ≈ 0.088° on both axes, debounced over two readings).

**`/ws/log` frames:**

```json
{ "seq": 142, "ts_ms": 83400, "lvl": "I", "tag": "motion", "msg": "move done in 1840 ms — AZ=90.00° EL=45.00° OK" }
```

Up to 4 concurrent clients. Only operational events (`motion`, `http`, `encoder`) — **not** a mirror of `ESP_LOGx`.

### 10.5 What the firmware does not do

- No closed-loop trajectory tracking — point-to-point only.
- No homing routine — encoders are absolute.
- No on-device storage of photos; no on-device web UI.

Memory footprint at idle: ~720 KB flash, ~210 KB free heap. Comfortable on a stock ESP32 with 4 MB flash split into two 1.875 MiB OTA slots.

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
| GET    | `/stereo/upstream/state`          | The binocular pair's `camserver` state, proxied (no-store).           |
| GET/POST | `/stereo/upstream/controls/{cam_id}` | Read / set a camera's V4L2 controls through the proxy.           |
| GET    | `/stereo/upstream/formats/{cam_id}` | List a camera's sizes, FourCCs, frame rates.                        |
| POST   | `/stereo/upstream/format/{cam_id}` | Set a camera's size, FourCC, fps.                                    |
| GET    | `/firmware/health`                | The firmware's `GET /health` (chip, IDF, uptime, heap), proxied.      |
| GET    | `/firmware/encoder`               | The firmware's `GET /test/encoder`: AS5600 readings, magnet status, I2C line levels. |
| POST   | `/firmware/ota`                   | Body = `firmware/build/orbiter.bin`; streamed to the firmware's `POST /ota`, its status relayed. |
| GET    | `/command`                        | The command names the server accepts.                                 |
| POST   | `/command/{name}`                 | Run a command over HTTP — the same handlers as the WS channel; the native workbench uses it (`set_stereo_rig`, …). |
| WS     | `/ws/scene`                       | Scene + model diffs, command channel ([§12.3](#123-wire-protocol)).   |

### 11.3 Environment variables

| Variable                       | Default                | Purpose                                        |
|--------------------------------|------------------------|------------------------------------------------|
| `ORBITER_STORAGE_DIR`          | `./data`               | Root for `scans/` + `captures/`.               |
| `ORBITER_ESP_IP`               | `orbiter.local`        | ESP32 firmware host (mDNS auto-discovery or the UI override it). |
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

Vite + React + react-three-fiber + zustand + Radix Tabs. Four tabs: **Scaner** (live 3D, motion controls, motion planner, scan-session controls), **Stereo** (the binocular pair: its `camserver`, eyes, orientation, nominal baseline), **Library** (saved scans, per-scan **Export SfM priors** — which also stages `photos/` for the COLMAP container — and a placeholder **Run COLMAP** button reserved for a future release) and **Help**.

### 12.1 Dev / build

```bash
cd ui/
npm install
npm run dev          # http://localhost:5173 (Vite proxies API to localhost:8000)
npx tsc --noEmit     # type-check without building
```

Production (Docker, two-stage `node:20-alpine` build → `nginx:alpine` serve on port 80):

```bash
docker build -t orbiter-ui .
docker run --rm -p 8080:80 orbiter-ui
```

### 12.2 Tabs

- **Scaner.** Single persistent WebSocket to `ws://server:8000/ws/scene`. Server pushes `scene_snapshot` + `scene_update` diffs (rendered via `nodeRegistry.ts`) and model state. Commands (`move`, `take_shot`, `start_scan`, …) go back over the same socket.
- **Library.** REST-only: `GET /scans`, `GET /config`, `POST /scans/{sid}/sfm_priors`, `DELETE /scans/{sid}`.
- **Stereo.** LEFT | RIGHT previews straight from `camserver` (an `<img>` each, so two MJPEG streams stay off the server), per-eye camera select, rotation and mirrors, Swap L/R, nominal baseline, draft → **Apply**. Apply sends `set_stereo_rig` over the scene socket; the server persists it and refuses two eyes on one camera. Orientation is flip-then-rotate, a contract shared with the server's command handler and the native workbench.

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

### 12.4 Native CV workbench

`native/` is a PySide6 desktop app for a second capture rig: two UVC cameras ~15 cm apart behind `camserver` on the LAN, and a line laser. It needs Python ≥ 3.11 and the sibling `server/` package (the ChArUco detection and pose code is imported from there, not copied):

```bash
python -m venv native/.venv                                   # Scripts/ below is the Windows layout; bin/ on Linux
native/.venv/Scripts/pip install -e ./server -e ./native
native/.venv/Scripts/pip install -e "./native[gpu]" --extra-index-url https://download.pytorch.org/whl/cu128   # optional: nvJPEG + torch; CUDA torch is not on PyPI
native/.venv/Scripts/orbiter-native --server http://localhost:8000
```

What it does, in the order you use it:

- **Calibrate**, guided stage by stage: per-eye intrinsics from diverse board views with per-view residuals, the stereo pair from views both eyes saw at one instant (paired on `camserver`'s capture clock, rejected if the board slid more than 0.3 px between the exposures), the laser plane, and the rolling-shutter readout time measured from the board in motion. Each solve lands in the server's config with its residual and view count; the panel shows the expected one-sigma error of a scanned point (the derivation is [`native/docs/ERROR_BUDGET.md`](native/docs/ERROR_BUDGET.md)). `orbiter-rigcheck` measures whether the rig can deliver what a scan needs.
- **Scan**: the board is tracked between ChArUco passes, the stripe is found on the GPU when there is one, each point is placed through the pose at its own row's instant, and the cloud accumulates in the board's frame with the colour of the surface beside the stripe. Frame path and thresholds: [`native/docs/PIPELINE.md`](native/docs/PIPELINE.md).
- **Photograph** alongside the scan: a photo session with poses and sidecars, then `orbiter-recon` (COLMAP 4 in Docker, or a local build) meshes the laser cloud and textures it from the photographs by default (`texture-only`, no GPU needed); `--mode dense` adds the images with the stripe masked and inpainted, our poses as the model, PatchMatch on one GPU, fusion, and the merge with the laser cloud ([`native/docs/PHOTOGRAMMETRY.md`](native/docs/PHOTOGRAMMETRY.md)). `native/docker/colmap-cuda128/` builds COLMAP for sm_75/89/120.

The three docs under `native/docs/` are in Russian; `native/README.md` (English) carries the design notes and what was measured on the bench.

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

Output ends up in `<storage>/scans/<sid>/colmap/`.

Before the first run, hit **Export → SfM priors** on the session (UI → Library) — besides writing `sfm_priors.json` it materializes the capture originals into `scans/<sid>/photos/`, the layout the runner reads.

**Option B — hand-off.** The same **Export → SfM priors** leaves a self-contained session (`photos/` + `sfm_priors.json`) you can feed to your own COLMAP install; the conversion logic is in `server/orbiter_server/sfm_export.py` and `colmap/sfm_priors_to_colmap.py`.

### 14.2 `sfm_priors.json` schema

```jsonc
{
  "schema": "orbiter.sfm_priors.v1",
  "camera_intrinsics": {
    "model": "PINHOLE",               // "OPENCV" once calibrated with distortion
    "width":  4080, "height": 3060,   // stored-photo pixels, not a guess
    "fx": 3122.9, "fy": 3122.1,
    "cx": 2039.5, "cy": 1529.5
  },
  "images": [
    {
      "file": "photos/001_az000_el+15.jpg",
      "qw":  0.707, "qx": 0, "qy": 0.707, "qz": 0,   // Hamilton quaternion
      "tx":   220,  "ty": 0, "tz":  45               // translation in mm
    }
    // ...
  ],
  "turntable": {                      // present once the rig is calibrated
    "axis_xy_mm": [1.8, 0.0],         // AZ rotation axis in world XY
    "board": { "rvec": [/*…*/], "t": [/*…*/],
               "width_mm": 288, "height_mm": 288 }   // nominal board pose
  }
}
```

Quaternion convention: **Hamilton** (w, x, y, z). Translations in **millimetres** in the world frame from [§4](#4-coordinate-system--data-model). The transform takes world points into camera space (COLMAP's convention). Intrinsics come from the ChArUco solve when the rig is calibrated (rotated to match the stored-pixel orientation); otherwise a scaled IP-Webcam guess. The optional `turntable` block feeds the mask tool's geometry stamps ([§14.5](#145-object-masks-optional)).

### 14.3 Container pipeline

The wrapper executes seven steps, halting on the first error:

| # | Step                              | Notes                                                  |
|--:|-----------------------------------|--------------------------------------------------------|
| 1 | `colmap feature_extractor`        | Camera model + params pinned from the priors; picks up `masks/` automatically ([§14.5](#145-object-masks-optional)); `--gpu` → `FeatureExtraction.use_gpu`. |
| 2 | `colmap exhaustive_matcher`       | Pairwise match.                                        |
| 3 | `sfm_priors_to_colmap.py --database` | Priors JSON → COLMAP text model, image/frame ids synced to the database; also fills the `pose_priors` table and emits `rigs.txt`/`frames.txt` for COLMAP ≥ 3.13. |
| 4 | `colmap point_triangulator`       | Uses the prior sparse as input.                        |
| 5 | `colmap image_undistorter`        | Sparse → dense workspace.                              |
| 6 | `colmap patch_match_stereo`       | Per-image depth maps (slow on CPU; huge GPU speedup).  |
| 7 | `colmap stereo_fusion`            | Depth maps → fused `.ply` point cloud.                 |

Each step is wrapped in `[PROFILE]` start/end markers with durations, so a slow run tells you where it went. Output layout under `<storage>/scans/<sid>/colmap/`: `sparse_priors/`, `database.db`, `sparse/0/`, `dense/{images,sparse,stereo,fused.ply}`.

### 14.4 How accurate are the priors?

With a calliper-measured arm and the standard AS5600 encoders:

| Quantity                | Typical    |
|-------------------------|------------|
| Per-shot rotation error | 0.5° – 1.5° |
| Per-shot position error | 2 – 10 mm  |

Not enough for "feature-free" reconstruction, but plenty as a warm start — COLMAP's bundle adjustment will polish them. Run the ChArUco calibration (**Machine config → Calibrate from board**) and both improve: the solve refines the machine geometry *and* embeds the solved camera intrinsics (with distortion) into the priors; without it a scaled IP-Webcam guess is used — override in **Camera config** if you've calibrated separately.

### 14.5 Object masks (optional)

Static surroundings — table, walls, the rig frame — are identical in every frame, so COLMAP matches features on them and drifts toward reconstructing the *room* instead of the object. The [`colmap/masks/`](colmap/masks/) tool suppresses them with per-image masks COLMAP honours via `--ImageReader.mask_path` (white = keep features, black = ignore). Everything that **rotates with the platform** — object, board, disc — stays white; the static room goes black.

Three methods, from zero-dependency up:

| Method      | Needs                          | Idea                                                                 |
|-------------|--------------------------------|----------------------------------------------------------------------|
| `motion`    | nothing extra (in-container)   | Within one elevation ring the camera is physically static (AZ spins the *platform*), so a per-pixel temporal median over the ring reconstructs the empty scene — no empty-table shots — and `‖frame − plate‖` is the moving assembly. The most robust method on this rig. |
| `geometric` | nothing extra (in-container)   | Projects the calibrated working volume through each pose — coarse silhouette, pure geometry. |
| `sam2`      | torch + SAM 2.1 weights (host) | Pixel-accurate segmentation, prompted and clamped by the projected geometry. Also sharpens `motion` edges when available (`--edge-refine`). |

```bash
docker compose --profile colmap run --rm colmap \
    generate_colmap_masks \
        --images  /data/scans/<sid>/photos \
        --masks   /data/scans/<sid>/masks \
        --poses   /data/scans/<sid>/sfm_priors.json \
        --preview /data/scans/<sid>/masks_preview \
        --method  motion
```

`run_colmap_session.sh` auto-detects the session `masks/` directory and adds the flag for you, and the server serves each frame's mask and overlay at `/scans/<sid>/photos/<idx>/mask` and `…/mask_preview` for eyeballing. Parameters and tuning: [`colmap/masks/README.md`](colmap/masks/README.md).

---

## 15. Troubleshooting

| Symptom                                                  | Likely cause                                              | Fix                                                                              |
|----------------------------------------------------------|-----------------------------------------------------------|----------------------------------------------------------------------------------|
| Motors hum but don't move                                | Driver `EN` floating, or `STEP` not toggling.             | Probe with a scope. Check the ENABLE wiring (GPIO 32 on the wired bench, 27 on Orbiter-Simple). |
| Drivers get hot fast                                     | Current trimpot too high.                                 | Aim for ~70 % of rated motor current.                                            |
| Reported AZ or EL jumps randomly                         | Magnet not concentric, stray field nearby, or the sensor's `DIR` pin floating. | Move encoder away from motors; check AS5600 `MD/ML/MH` register; strap `DIR` to GND. |
| EL keeps spinning past `90°`                             | Encoder zero off, magnet flipped, or EL sign reversed after a sensor swap. | Re-run calibration and check the EL sign ([§10.3](#103-first-run-encoder-calibration)). |
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

## 16. What's out of scope

The full Orbiter project (in the parent repo) includes more — left out of the kit on purpose:

- **The multi-user library service** (`orbiter_library`, leases, derived-data generations) and its UI pages. The kit keeps one flat server and one Library tab.
- **The parent project's post-processing and deployment scripts.** `orbiter-recon` ends at a textured COLMAP mesh (`mesh/model.glb`); what the parent project does with it after that, and how it deploys, are not here.

Two things that *were* out of scope until v0.2 are in now: the laser-stripe scanner lives in `native/` ([§12.4](#124-native-cv-workbench)), and photogrammetry has an offline runner, `orbiter-recon`, beside the single-session COLMAP container. Both are built for the binocular rig; the single-camera turntable path is unchanged (one NaN guard in the board-pose solver aside).

ChArUco hand-eye geometry calibration *is* shipped here — the **Machine config → Calibrate from board** flow in the UI. The default board spec is an 8×8 ChArUco, `DICT_5X5_100`, 36 mm squares / 26.64 mm markers (generate one at [calib.io](https://calib.io/pages/camera-calibration-pattern-generator), print, **measure a square with a ruler** and set the real size in Machine config → Board — a 20 % size error biases every solved distance by the same 20 %). The v0.2 solver resolves the flat-board pose ambiguity, refines the EL-axis tilt and reports per-view residuals; per-shot errors land well under a degree / a few mm.

The architecture leaves clean seams where these can plug back in — the server already speaks in terms of `Manifest`, `MotionPlan`, `Pose`. The UI's Library tab carries a disabled **Run COLMAP** button reserved for a future release.

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
├── cad/                    ← CAD: the v1 rig (assembly + per-part STLs + screenshots)
│   ├── v2/                 ← the current mechanics (STL, STEP, FreeCAD, assembly GLB)
│   └── LICENSE             ← CERN-OHL-S-2.0 (hardware)
├── firmware/               ← ESP-IDF project — the 2-axis IP actuator
├── server/                 ← FastAPI — scene state + ESP proxy + storage
├── ui/                     ← Vite + React + react-three-fiber — viewer, library, stereo rig
├── native/                 ← PySide6 CV workbench — calibration, laser scanning, orbiter-recon
├── docker/                 ← docker-compose stack
└── colmap/                 ← COLMAP container + integration notes
```

| Top-level file / dir | One-line role                                                                          |
|----------------------|----------------------------------------------------------------------------------------|
| `README.md`          | This file — consolidated single-page reference.                                        |
| `firmware/`          | ESP-IDF v6 source for the 2-axis IP actuator.                                          |
| `server/`            | FastAPI app, `ModelState`, scene-graph builder, scan orchestration, COLMAP-priors exporter. |
| `ui/`                | Single-page React app over the scene-graph WebSocket.                                  |
| `native/`            | Desktop CV workbench for the binocular pair: `orbiter-native`, `orbiter-rigcheck`, `orbiter-recon`; docs in `native/docs/`. |
| `cad/`               | `cad/v2/`: the current mechanics as STL, STEP, FreeCAD and the assembly GLB the server and viewer read. The rest: the v1 rig (`.glb`/`.fbx`/`.obj` assembly, per-part `.stl`, previews). |
| `docker/`            | `docker-compose.yml` with three services and a `colmap` profile.                       |
| `colmap/`            | Thin container over `colmap/colmap:latest` + an SfM-priors → COLMAP-text converter.    |
| `docs/`              | Per-subsystem deep dives (architecture, hardware, firmware, API, assembly, COLMAP).    |
