# Firmware

ESP-IDF v6 project. Builds on Windows / Linux / macOS.

## Prerequisites

Install **ESP-IDF v6.0** following [the official instructions](https://docs.espressif.com/projects/esp-idf/en/v6.0/esp32/get-started/index.html).
On Windows the easiest path is the **ESP-IDF Tools Installer** — it sets up
the toolchain, Python, and a `idf.py` shortcut in your start menu.

Test:
```bash
idf.py --version
```

## Build & flash

From [`firmware/`](../firmware/):

```bash
idf.py set-target esp32        # or esp32s3, etc.
idf.py menuconfig              # set Wi-Fi credentials — see below
idf.py -p COM3 flash monitor   # COM3 / /dev/ttyUSB0 / whatever your board enumerates as
```

After boot you should see:

```
I (xxxx) wifi:connected → ip = 192.168.1.42
I (xxxx) encoder: AS5600 found at 0x36
I (xxxx) encoder: AS5048A SPI OK
I (xxxx) calibration loaded: az_zero=0.000° el_zero=138.296°
I (xxxx) http: server started on port 80
```

Write down the IP — the server will need it.

## `menuconfig` settings

Open `idf.py menuconfig` → **Orbiter Configuration**:

| Symbol | Purpose | Default |
|--------|---------|---------|
| `ORBITER_WIFI_SSID` | Your 2.4 GHz Wi-Fi SSID | — (must be set) |
| `ORBITER_WIFI_PASSWORD` | Wi-Fi password | — (must be set) |
| `ORBITER_WIFI_MAX_RETRY` | Connect retries before backoff | 10 |
| `ORBITER_AZ_MAX_DEG` | Soft upper bound on AZ (`0` = unbounded) | 0 |
| `ORBITER_EL_MIN_DEG` | Lower limit on EL | -25 |
| `ORBITER_EL_MAX_DEG` | Upper limit on EL | 90 |

The Wi-Fi credentials are baked into the firmware at build time. If you want
runtime configuration (a captive portal, BLE provisioning, etc.) you'll need
to add it — out of scope for v0.1.

> **2.4 GHz only.** The ESP32 radio doesn't support 5 GHz. If your router
> exposes both bands under split SSIDs (`<name>` and `5G-<name>`), pick the
> 2.4 GHz one — `idf.py monitor` will otherwise loop on
> `wifi: 10 retries exhausted` even though the password is correct.

> **Don't put Wi-Fi credentials into source control.** Set them via
> `menuconfig` (which writes `sdkconfig`, gitignored by default) or pass
> them on the build command line: `idf.py -DCONFIG_ORBITER_WIFI_SSID=... build`.

## First-run encoder calibration

Out of the box, encoder zeros are:

- `az_zero_raw = 0.0°` — wherever the AS5600 happens to land at power-up.
- `el_zero_raw = 138.296°` — a placeholder from the original prototype.

Both are almost certainly wrong for your build. To fix:

1. Manually move the elevation arm to horizontal (use a spirit level if you
   want to be careful). The arm should point along `+X` in the world frame
   (i.e. away from the user).
2. From a terminal:
   ```bash
   curl -X POST http://<device-ip>/calibrate -H 'Content-Type: application/json' \
        -d '{"axis":"el","mode":"current"}'
   ```
3. Repeat for AZ with the platform facing forward.
4. `GET /state` should now report `el ≈ 0.0°` and `az ≈ 0.0°`.

The values are persisted to NVS and survive reboot.

## Building without a board

The project builds headless — useful for CI. If you don't have ESP-IDF set
up locally:

```bash
docker run --rm -v $PWD:/project -w /project \
    espressif/idf:release-v6.0 \
    idf.py build
```

(Flashing from inside a container is fiddly — easier to do that on a host
that sees the USB device.)

## Pin map

See [`HARDWARE.md`](HARDWARE.md). If you want to change pins, edit
`main/Kconfig.projbuild` and rebuild.

## What the firmware *doesn't* do

- No closed-loop trajectory tracking — only position control with a P-loop
  in velocity mode. Good to ~1° at modest speeds; jitter at high speeds.
- No homing routine — the encoders are absolute, no homing needed.
- No SD card / no on-device storage of photos. The server holds the data.
- No on-device web UI. The server provides that.

If any of these matter for your application, the source is small enough
(~1500 lines C) that bolting them on is reasonable.
