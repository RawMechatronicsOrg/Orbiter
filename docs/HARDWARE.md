# Hardware

This is what *we* used. None of it is precious. Where there's an obvious
substitution, it's called out.

## Bill of materials

The mechanics are the second generation (v2): the full parts list with
quantities, printed parts and CAD is in [`cad/v2`](../cad/v2/README.md).
The electronics side is below.

| Group | Part | What we used | Sensible alternatives |
|-------|------|--------------|-----------------------|
| MCU | ESP32 dev board | ESP32-D0WD-V3 (rev 3.1), 4 MB flash | Any ESP32 with Wi-Fi (ESP32-WROOM, ESP32-S3 — pin map will shift) |
| Motor driver × 2 | A4988 stepper driver carrier | Generic Pololu-style clone | DRV8825 (same footprint, swap microstep table). TMC2208/2209 if you want quiet — wire the same `STEP/DIR/EN`, ignore UART for now. |
| Stepper × 2 | NEMA-17, 24 mm body | 1.8°/step (200 steps/rev) | Any NEMA-17 that fits the motor mounts; another step angle changes only the step rate per output degree (see [Motion math](#motion-math)). |
| Encoder × 2 | AS5600 magnetic angle sensor module | 12-bit, I2C @ 0x36 | AS5048B (I2C, 14-bit) for more resolution. |
| Encoder magnets × 2 | Diametrically magnetised disc | — | Anything diametric. It sits in a printed `HexPlug` in the end of the azimuth spindle and of the elevation shaft, facing the sensor. |
| Belts | GT2 closed loop, 6 mm wide | 280 mm on azimuth | Elevation: the model does not name a length; at the modelled centre distance (69.6 mm) the loop is about 347 mm. |
| Pulleys | GT2 16-tooth on the motors; printed 112-tooth (AZ) and 160-tooth (EL) | 7 : 1 on azimuth, 10 : 1 on elevation | The printed pulleys sit on splines, so another tooth count is a reprint (`ReplacablePulley` on EL). |
| Bearings × 4 | 6008 (40 × 68 × 15 mm) | Two per axis | 6008-2Z / -2RS. |
| Frame | 25 × 25 mm aluminium tube | 3 × 253 mm, 1 × 93 mm | Any 25 × 25 square tube; the printed caps and clamps take the rest. |
| Power supply | 12 V, ≥ 2 A | Lab bench supply during dev | A 12 V / 3 A "LED strip" brick is fine for headless deployment. |
| Camera | Android phone with [IP Webcam](https://play.google.com/store/apps/details?id=com.pas.webcam) | Anything that exposes MJPEG over HTTP | A Raspberry Pi with a CSI camera + `mjpg-streamer`. A USB webcam through `motion`. Anything with a URL that returns JPEGs. |
| Printed parts | PLA | 25 files, 32 pieces, see [`cad/v2`](../cad/v2/README.md#printed-parts) | PETG or ABS if you live somewhere hot. |

## Mechanics

The geometry the software uses comes from the v2 CAD
([`cad/v2/assembly/Orbiter.glb`](../cad/v2/assembly), exported from the
Fusion model, version 225). The world frame is millimetres, Z up, the
machine standing on z = 0, the azimuth axis on the Z axis.

| Quantity | Value | Where it comes from |
|----------|-------|---------------------|
| Azimuth axis | vertical, x = y = 0 | `ORB_ROTOR_base` in the GLB: centre of the two 6008 inner rings |
| Elevation axis | along X, at y = 0, z = 127.18 mm | `ORB_ROTOR_elevation` in the GLB: centre of the 6008 inner rings |
| Axes crossing | yes — the elevation axis passes through the azimuth axis | both rotors at y = 0 |
| Platform | 120 mm disc, top at z = 118.64 mm (8.5 mm under the elevation axis) | `BasePlatformInsert Taper` |
| Orbit arm | 25 × 25 mm tube along +Y at el = 0; tip 225.0 mm from the elevation axis, centred at x = 179 mm | `OrbitArm` |
| Elevation travel | −25° … +90° (0 = arm horizontal) | the arm meets the cross rail at −28.8° in the model; nothing stops it upward. `ORBITER_EL_MIN_DEG` / `_MAX_DEG` in firmware menuconfig |

In the model the arm rests 0.84° below horizontal. Where the camera sits on
the arm is not part of the model: the server takes it from the hand-eye
calibration, and before that from the arm radius and camera offset in the
UI (an unset arm radius defaults to the arm's 225 mm reach).

## Pin assignment (ESP32-D0WD-V3 dev board)

Numbers are GPIO numbers (not silkscreen).

### Steppers (A4988)

| Signal | GPIO | Notes |
|--------|------|-------|
| Stepper 1 STEP (AZ) | 25 | |
| Stepper 1 DIR (AZ) | 26 | |
| Stepper 2 STEP (EL) | 27 | |
| Stepper 2 DIR (EL) | 33 | Was originally 14 — moved off JTAG MTMS pin (floated to ~0.9 V at boot and clocked phantom steps). Keep clear of strapping/JTAG pins on your dev board. |
| ENABLE (shared) | 32 | Active LOW. Wire to both drivers in parallel. |
| MS1 / MS2 / MS3 | tied to 3.3 V | 1/16 microstepping on both drivers. Pull-up to VCC, no MCU control needed. |

### Encoders

The v2 mechanics carry an AS5600 on both axes. The firmware still drives
the elevation encoder as an AS5048A over SPI, so until that changes the
elevation sensor is wired as below.

**AZ — AS5600 (I2C @ 400 kHz, address 0x36 fixed)**

| Signal | GPIO | Notes |
|--------|------|-------|
| SDA | 21 | 4.7 kΩ pull-up to 3.3 V (often already on breakout). |
| SCL | 22 | 4.7 kΩ pull-up to 3.3 V. |
| DIR | GND | Tie low for CCW = increasing angle. Flip to VCC to invert. |

**EL — AS5048A (SPI Mode 1, manual chip select)** — what the firmware reads today

| Signal | GPIO | Notes |
|--------|------|-------|
| CLK  | 18 | |
| MISO | 19 | |
| MOSI | 23 | The chip uses MOSI to receive read commands — don't omit it. |
| CS   | 17 | Manual GPIO (we don't use the SPI peripheral's hardware CS). |

> `GPIO 5` is free on this board (it was the chip-select for an older
> AS5048A on AZ that's been replaced with AS5600). You can use it for an
> LED, a button, anything.

## Motion math

| Parameter | Azimuth | Elevation | Where it comes from |
|-----------|---------|-----------|---------------------|
| Motor step angle | 1.8° (200 steps/rev) | 1.8° (200 steps/rev) | Motor datasheet |
| Microstepping | 1/16 | 1/16 | A4988 with MS1/MS2/MS3 high |
| Belt reduction | 7 : 1 | 10 : 1 | GT2 16-tooth motor pulley → 112 / 160-tooth output pulley |
| Microsteps / output rev | 22 400 | 32 000 | 200 × 16 × 7, 200 × 16 × 10 |
| Microsteps / output degree | 62.2 | 88.9 | |
| Degrees / microstep at output | 0.0161° | 0.01125° | |
| Encoder step | 0.088° | 0.088° (AS5600) | 360 / 4096 |

The firmware never uses these numbers: it closes the loop on the encoders
(below), so a different motor or pulley needs no firmware change. Its speed
limits are step rates, though — `CL_HZ_MAX_AZ` = 2100 Hz is about 33.8 °/s
at the platform and `CL_HZ_MAX_EL` = 350 Hz about 3.9 °/s at the arm, and
the UI speed knobs are in the same Hz. With 1.2° motors they would be 1.5×
slower.

> **Open-loop micro-step accuracy is a fiction at low currents.** The
> A4988 micro-step table assumes balanced phase currents — real ones drift.
> That's why we use the magnetic encoders as the actual source of truth and
> the firmware runs a P-loop in velocity mode rather than counting steps.
> If you skip the encoders the firmware will still drive open-loop but you
> shouldn't trust the reported angle.

## Wiring sketch

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
                    
       AS5600  AS5048A (firmware EL driver; v2 mounts an AS5600)
        SDA─21  CLK─18
        SCL─22  MISO─19
                MOSI─23
                CS  ─17
                
       USB ── ESP32 ── Wi-Fi to your network
```

The ESP32 board can be powered over USB during bench tests. For headless
running, feed `Vin` from the same 12 V rail through a small step-down
(MP1584 / LM2596 / your favourite).

## Connecting a camera

Install [IP Webcam](https://play.google.com/store/apps/details?id=com.pas.webcam)
on an Android phone. Start the server, note the URL (e.g.
`http://192.168.1.42:8080/`). Enter that URL in the server config. The
server fetches still JPEGs from `/shot.jpg` for each capture.

> If you want something other than IP Webcam, the server uses a
> [`CameraAdapter`](../server/orbiter_server/camera_adapter.py) interface —
> implement `take_photo() -> bytes` and you're done.

## Mounting the encoders

Magnets are diametrically magnetised. In v2 each one sits in a `HexPlug`
pressed into the end of the azimuth spindle and of the elevation shaft, so
it is concentric with the axis by construction. The sensors sit on a printed
`BaseSensorRing` under the spindle and on the three-legged
`AngleSensorMount Ribbed` on the elevation housing, about 1 mm from the
plug.

If the AS5600 reports unstable angles, check the gap and the centring. The
`MD/ML/MH` register will tell you if the magnet field is in spec.
