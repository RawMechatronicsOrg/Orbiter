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
idf.py menuconfig              # set Wi-Fi credentials and board wiring — see below
idf.py -p COM3 flash monitor   # COM3 / /dev/ttyUSB0 / whatever your board enumerates as
```

The first flash of a board goes over USB, as above. After that, update over
Wi-Fi — see [Updating over Wi-Fi](#updating-over-wi-fi).

After boot you should see:

```
I (xxxx) wifi:connected → ip = 192.168.1.42
I (xxxx) encoder: init — AS5600(AZ) SDA=22 SCL=23 | AS5600(EL) SDA=4 SCL=16
I (xxxx) calibration loaded: az_zero=0.000° el_zero=138.296°
I (xxxx) http: server started on port 80
```

Write down the IP — the server will need it.

## Updating over Wi-Fi

`POST /ota` replaces the firmware without a cable. Build, then send the raw
app image as the request body:

```bash
idf.py build
curl --data-binary @build/orbiter.bin http://<esp-ip>/ota
```

What to know:

- **Two OTA slots.** The 4 MB flash holds two app slots of 1.875 MiB each
  ([`partitions.csv`](../firmware/partitions.csv)); the update goes into the
  one that is not running. `nvs` stays at `0x9000` (size `0x6000`) as in the
  old single-app layout, so the stored calibration survives.
- **What it refuses.** An image from another project (`400`), an empty body
  (`400`), an image larger than the slot (`413`), and any request while the
  axes are moving (`409`). Full list: [API.md](API.md#post-ota).
- **Motors.** They are disabled while the image is written. On success the
  device reboots into the new image.
- **Rollback.** A new image confirms itself once its HTTP server is up. If
  that does not happen within 120 s, the bootloader boots the previous image.
  So a build that cannot join Wi-Fi still leaves the device reachable.
- **The first flash must be over USB.** A board that still has the old
  single-app partition table has no second slot: flash it once with
  `idf.py -p COM3 flash`.

> **Dev-board clones and download mode.** Some ESP32 dev boards do not enter
> download mode by themselves: `esptool` stops with
> `Wrong boot mode detected (0x17)`. Hold the **BOOT** button while `esptool`
> prints `Connecting...`. Once the board is flashed this way, use `/ota`.

## `menuconfig` settings

Open `idf.py menuconfig` → **Orbiter Configuration**:

| Symbol | Purpose | Default |
|--------|---------|---------|
| **Board wiring** (choice) | Which stepper pins the firmware drives: `ORBITER_BOARD_WIRED` — the jumper-wire bench; `ORBITER_BOARD_SIMPLE` — the Orbiter-Simple carrier. Pins for both are in [`HARDWARE.md`](HARDWARE.md#steppers-a4988). The encoder pins are the same on both. | `ORBITER_BOARD_WIRED` |
| `ORBITER_WIFI_SSID` | Your 2.4 GHz Wi-Fi SSID | — (must be set) |
| `ORBITER_WIFI_PASSWORD` | Wi-Fi password | — (must be set) |
| `ORBITER_WIFI_MAX_RETRY` | Connect retries before backoff | 10 |
| `ORBITER_EL_MIN_DEG` | Lower limit on EL | -25 |
| `ORBITER_EL_MAX_DEG` | Upper limit on EL | 90 |
| `ORBITER_AS5600_I2C_TRACE` | Log every I2C register read of both AS5600 sensors at INFO, with retries and error codes. Verbose; rebuild after changing. | off |

The Wi-Fi credentials are baked into the firmware at build time. If you want
runtime configuration (a captive portal, BLE provisioning, etc.) you'll need
to add it — out of scope for this kit.

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

After swapping a sensor, calibrate that axis again. For EL also check the
sign on the rig: raise the arm and `el` in `GET /state` must increase. The
firmware expects the EL sensor's raw reading to *decrease* as the arm rises
and negates it; an AS5600 reverses with its DIR strap and the magnet side
(see `el_user_from_raw` in [`motion.c`](../firmware/main/motion.c)).

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

See [`HARDWARE.md`](HARDWARE.md). The stepper wiring is the **Board wiring**
choice in `menuconfig` (above). The encoder pins are fixed in
`main/encoder.c` and the stepper pins in `main/stepper.c`; to change them,
edit the source and rebuild.

Each AS5600 needs its own I2C bus (its address is fixed at 0x36), and its
own `DIR` pin, on the sensor board, must be strapped to GND — a floating
`DIR` makes the angle jump.

## What the firmware *doesn't* do

- No closed-loop trajectory tracking — only position control with a P-loop
  in velocity mode. Good to ~1° at modest speeds; jitter at high speeds.
- No homing routine — the encoders are absolute, no homing needed.
- No SD card / no on-device storage of photos. The server holds the data.
- No on-device web UI. The server provides that.

If any of these matter for your application, the source is small enough
(~1500 lines C) that bolting them on is reasonable.
