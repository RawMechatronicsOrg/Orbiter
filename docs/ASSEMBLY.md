# Assembly

This is a sketch, not a step-by-step manual. The CAD geometry is the
source of truth — open [`cad/assembly/Orbiter.glb`](../cad/assembly/Orbiter.glb)
in Blender / Meshlab / your slicer and you'll see how the pieces fit.

## Print orientation

| Part | Orientation | Infill | Layer | Supports |
|------|-------------|--------|-------|----------|
| `AxisFrame.stl` | flat on the bed | 30 % | 0.2 mm | none |
| `OrbitFrame.stl` | as imported | 30 % | 0.2 mm | yes — small overhang on the bearing seat |
| `TableSupportFrame.stl` | flat on the bed | 25 % | 0.2 mm | none |
| `PlatformInsert.stl` | flat side down | 25 % | 0.2 mm | none |
| `BarHolderInsert.stl` | any | 30 % | 0.2 mm | none |
| `GT2 Pulley.stl` | flat on bed, hub up | 40 % | 0.15 mm | none — slow down to keep teeth crisp |
| `Hall Sensor Mount.stl` | flat | 25 % | 0.2 mm | none |

PLA at 30 % infill is plenty for a desk-bound rig. If you live somewhere
warm (PLA softens above ~55 °C) or you want a stiffer machine, PETG / ABS
work the same way — re-tune the printer.

## Hardware checklist

Before you start gluing things together:

- 8 × M3 × 12 mm socket-head cap screws — frame to base
- 4 × M3 × 8 mm — encoder breakouts
- 4 × M3 nylon spacers (4 mm height) — under the encoder breakouts
- 2 × 5 mm round shafts cut to length — output axes
- 2 × 5 mm bore × 8 mm OD ball bearings (608 type) — output axis support
- 2 × small diametric magnets (6 mm × 2.5 mm) — glued into pulley hubs
- M2.5 × 6 mm — A4988 carriers to the ESP32 board (or jumper wires)
- GT2 belt × 2 — measure on the printed frame, cut to length
- Misc dupont wires, screw terminals, breadboard or perfboard

## Order of operations

1. **Print everything.** Inspect for blobs / strings on the pulley teeth —
   they'll cost you ~0.5° of backlash if you don't clean them up.
2. **Press-fit the bearings** into `OrbitFrame` and the base.
3. **Glue the diametric magnets** into the back of the 100-tooth output
   pulleys, centred as well as you can manage (concentricity matters more
   than depth).
4. **Mount the encoder breakouts** on their printed seats with 4 mm
   spacers. Aim for a 1.5–2 mm gap between IC and magnet.
5. **Belt and pulleys** — install with the belt under modest tension. If
   teeth skip during slow moves, tighten; if motors stall, loosen.
6. **Wire it up** following [`HARDWARE.md`](HARDWARE.md). Triple-check the
   stepper coil pairs — getting one wrong reverses direction or causes
   high-current rattling.
7. **Flash the firmware**, set Wi-Fi, watch for the boot log.
8. **Calibrate the encoders** at horizontal arm / forward platform — see
   [`FIRMWARE.md`](FIRMWARE.md).
9. **Start the server + UI** and verify a manual move works end-to-end.

## Mounting the camera

The arm has slots for a generic phone holder — a friction-grip phone
mount from a car kit works. The camera should sit at the **end of the arm**
(farther from the AZ pivot is better — longer baseline = more parallax for
photogrammetry).

Measure `arm_radius` = distance from the elevation pivot to the camera
sensor along the arm. Enter it in the UI when creating a session.
`base_height` is the elevation pivot above the platform surface;
`camera_offset` is the perpendicular distance from the arm to the camera.

If you don't care about absolute pose accuracy (you're just taking pretty
photos for a turntable GIF), eyeball the numbers. If you're going to feed
them to COLMAP as priors, measure with a calliper.

## Troubleshooting

| Symptom | Likely cause |
|---------|--------------|
| Motors hum but don't move | Driver `EN` floating, or `STEP` not toggling. Probe with a scope. |
| Drivers get hot fast | Current trimpot turned up too far. Aim for ~70 % of rated motor current. |
| Reported AZ jumps randomly | Magnet not concentric, or stray field nearby. Move the encoder away from the motors and re-check. |
| EL keeps spinning past `90°` | Encoder zero is off, or magnet flipped. Re-run calibration. |
| Belts squeal | Too tight. Loosen tensioner. |
| Stepper skips at speed | Too fast for the current setting. Reduce `rate_deg_per_s` or increase driver current. |

Bigger problems: check the firmware log over WebSocket. It's verbose by
design.
