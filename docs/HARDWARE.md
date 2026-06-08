# Hardware

This is what *we* used. None of it is precious. Where there's an obvious
substitution, it's called out.

## Bill of materials

| Group | Part | What we used | Sensible alternatives |
|-------|------|--------------|-----------------------|
| MCU | ESP32 dev board | ESP32-D0WD-V3 (rev 3.1), 4 MB flash | Any ESP32 with Wi-Fi (ESP32-WROOM, ESP32-S3 — pin map will shift) |
| Motor driver × 2 | A4988 stepper driver carrier | Generic Pololu-style clone | DRV8825 (same footprint, swap microstep table). TMC2208/2209 if you want quiet — wire the same `STEP/DIR/EN`, ignore UART for now. |
| Stepper × 2 | NEMA-17 | 1.2°/step (300 steps/rev), ~1 A | Standard 1.8°/step (200 steps/rev) is fine — just adjust microstepping. Any NEMA-17 with enough torque for a 1:10 belt reduction. |
| Encoder AZ | AS5600 magnetic angle sensor breakout | 12-bit, I2C @ 0x36 | AS5048A (SPI, 14-bit) — see EL row. AS5048B (I2C, 14-bit) is a drop-in if you need more resolution on AZ. |
| Encoder EL | AS5048A magnetic angle sensor breakout | 14-bit, SPI Mode 1 | AS5600 (cheaper, 12-bit) if 0.088° resolution is enough. |
| Encoder magnets × 2 | Diametrically magnetised disc | 6 mm × 2.5 mm cylindrical N35 | Anything diametric. ~5–8 mm diameter, ~1.5–3 mm thick. Glue or press-fit into the back of the output pulley. |
| Belt × 2 | GT2 closed-loop | `2M-348-6` (2 mm pitch, 348 mm circumference, 6 mm width) — runs fine but is slightly long for this frame; a shorter belt would tension cleaner. | Any closed-loop GT2-compatible belt (2 mm pitch, 6 mm wide). Pick the next length down from 348 mm if you want less idle slack. |
| Pulley × 2 | GT2 10-tooth pinion + 100-tooth output | 1 : 10 reduction | Any pair that gives 1:8 to 1:15. Lower ratio = less torque, higher speed; higher ratio = slower, smoother. |
| Power supply | 12 V, ≥ 2 A | Lab bench supply during dev | A 12 V / 3 A "LED strip" brick is fine for headless deployment. |
| Camera | Android phone with [IP Webcam](https://play.google.com/store/apps/details?id=com.pas.webcam) | Anything that exposes MJPEG over HTTP | A Raspberry Pi with a CSI camera + `mjpg-streamer`. A USB webcam through `motion`. Anything with a URL that returns JPEGs. |
| Frame | 3D-printed in PLA | Default 0.2 mm layer, 30% infill | PETG or ABS if you live somewhere hot. |

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

**AZ — AS5600 (I2C @ 400 kHz, address 0x36 fixed)**

| Signal | GPIO | Notes |
|--------|------|-------|
| SDA | 21 | 4.7 kΩ pull-up to 3.3 V (often already on breakout). |
| SCL | 22 | 4.7 kΩ pull-up to 3.3 V. |
| DIR | GND | Tie low for CCW = increasing angle. Flip to VCC to invert. |

**EL — AS5048A (SPI Mode 1, manual chip select)**

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

| Parameter | Value | Where it comes from |
|-----------|-------|---------------------|
| Motor step angle | 1.2° (300 steps/rev) | Datasheet of the specific motor we bought |
| Microstepping | 1/16 | A4988 with MS1/MS2/MS3 high |
| Gear ratio | 1 : 10 | 10-tooth pinion / 100-tooth output pulley |
| Steps / output rev | 48 000 | 300 × 16 × 10 |
| Degrees / microstep at output | 0.0075° | 360 / 48 000 |

If your motor is 1.8°/step (200 steps/rev) and you keep 1/16 microstep + 1:10
gearing, you get 32 000 microsteps/rev or 0.01125°/microstep. Adjust in
firmware menuconfig.

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
                    
       AS5600  AS5048A
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

Magnets are diametrically magnetised. They need to be:

- **Concentric** with the rotation axis (within ~0.3 mm). The pulley
  centre is usually good enough.
- **Close** to the sensor IC (~1–2 mm gap, not touching). Check the
  datasheet of your specific breakout.

If the AS5600 reports unstable angles, check both. The `MD/ML/MH` register
will tell you if the magnet field is in spec.
