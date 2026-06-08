# Orbiter Firmware

ESP-IDF v6 project. Exposes an ESP32 as a **2-axis IP actuator with
absolute encoder feedback** over HTTP + WebSocket.

The firmware doesn't know it's a turntable. The host (`server/`) interprets
the axes as azimuth and elevation, but you can talk to this firmware
directly to drive anything that needs two motors and a known position.

## Build

```bash
cd OrbiterV0.1/firmware
idf.py set-target esp32              # or esp32s3, etc.
idf.py menuconfig                    # set Wi-Fi credentials + axis limits
idf.py -p COM3 flash monitor
```

See [`../docs/FIRMWARE.md`](../docs/FIRMWARE.md) for prerequisites and the
first-time setup steps.

## What's in here

| File | What it does |
|------|--------------|
| `main/orbiter_main.c` | App entry. Brings up Wi-Fi, NVS, motion, HTTP server. |
| `main/motion.c/.h` | Motion controller — closed-loop position via a P-loop in velocity mode using the encoder feedback. |
| `main/motion_runner.c/.h` | Background task that owns the active move and serves `/move` request lifetimes. |
| `main/stepper.c/.h` | Low-level stepper driver — STEP/DIR/EN pulse generation. |
| `main/encoder.c/.h` | Encoder dispatcher (per-axis read with debounce). |
| `main/enc_as5600.c/.h` | AS5600 I2C driver (AZ). |
| `main/http_handlers.c/.h` | REST handlers — `/health`, `/state`, `/move`, `/spin`, `/calibrate`, `/reboot`. |
| `main/log_bus.c/.h` | Curated event ring buffer + WebSocket fanout for `/ws/log`. |
| `main/Kconfig.projbuild` | `menuconfig` options (Wi-Fi creds, axis limits, verbose tracing). |
| `sdkconfig.defaults` | Sensible defaults — bumps LWIP socket count for parallel polling. |

## API

The HTTP / WebSocket contract is documented in [`../docs/API.md`](../docs/API.md).

## What's *not* here

- No SD card storage / no photo handling on-device.
- No on-device web UI.
- No closed-loop trajectory tracking — point-to-point only.
- No homing routine — encoders are absolute, no need.
- No firmware-side scan logic — that lives on the server.

If you need any of those, the source is small (~1500 lines of C). Fork
and bolt them on.

## Memory footprint

| Region | Approx |
|--------|--------|
| Flash (text + rodata) | ~720 KB |
| RAM (free heap at idle) | ~210 KB |

Well within an ESP32 with 4 MB flash and the default partition table.

## Wi-Fi credentials

Set via `menuconfig`, **not** hardcoded:

```
Orbiter Configuration
  └─ WiFi SSID:     "<your network>"
  └─ WiFi password: "<your password>"
```

These are baked into the firmware at build time. `sdkconfig` is gitignored
by default — your creds won't end up in version control.

If you'd rather provision at runtime (BLE / captive portal / SmartConfig),
you'll need to add it. Out of scope for v0.1.
