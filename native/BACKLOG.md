# Backlog — native scanner

What is planned and why, in the order it is worth doing. Each entry says what
it buys and what it costs, so a session can pick the next one without
re-deriving the reasoning. Done work moves to the README's design notes.

## Laser strobing — background subtraction and true colour  ★ must

**What.** Toggle the laser per frame from the ESP32 (GPIO on the laser's
TTL/enable), so the stream alternates laser-on and laser-off frames.
Subtracting the off frame from the on frame leaves only the laser's light;
the off frame is the subject's true colour.

**Why it is the largest SNR lever left.** Every stripe detector we have fights
the scene: the redness score, the excess-over-background opening, the
right-eye veto, the reach and jump gates all exist because the stripe shares
the frame with red wires, glints, coloured surfaces and room light. `on − off`
removes the scene by construction: dark rubber that reflects 14 levels of red
becomes 14 levels above zero instead of 14 above a background of 40; a blue
battery, a red switch, a glint of the desk lamp all cancel; the threshold
becomes a number about the laser, not about the bench. The calibration-mode
line fits get the same stripe. And the off frame gives each scanned point its
real colour — today points are laser-tinted, sampled under the laser (the old
handheld pipeline had a laser-off frame for exactly this and lost it).

**Why it is weeks, not days.** Two things have to be true at once:

1. *The toggle must land in the vertical blanking.* These sensors are rolling
   shutter with a readout of ~T (measured per eye by the readout solve; expect
   15-30 ms of a 33.3 ms frame). A toggle during the readout leaves a frame
   whose top rows saw one state and bottom rows the other. The window for a
   clean toggle is the remainder, and the phase has to be held to a few ms.
2. *Nothing tells the ESP32 when a frame starts.* UVC cameras expose no
   strobe/sync pin, and camserver only timestamps frames after the fact
   (`X-Capture-Monotonic`, start-of-exposure). So the phase has to be closed
   from the images.

**Design sketch (software PLL).**

- ESP32 runs a free timer at exactly half the frame rate (15 Hz toggle for
  30 fps), with an HTTP endpoint to nudge its phase by ±N ms and to set the
  period — the same bare-path HTTP style as ActuatorController.
- The app classifies every frame: fully lit, fully dark, or *split* (the
  stripe present only above/below some row — the readout solve's `T` says
  which row the toggle landed on, and the sign says whether the toggle is
  early or late). Split frames are the error signal; the app nudges the
  ESP32's phase until they vanish, then keeps nudging slowly against drift
  (two free-running clocks, ppm apart).
- Pairing: an on frame and its nearest off frame (33 ms apart) form the
  subtraction pair. The subject moves ~1 mm in 33 ms at a hand's 30 mm/s;
  either accept it (the stripe is thin, the background smooth) or slide the
  off frame by the board's twist (`rolling.Motion`) before subtracting.
- The stripe pipeline runs on `on − off` (GPU: one subtraction before
  `stripe_score`, no opening needed); the point colour is sampled from the
  off frame at the centroid.
- Both cameras see the same laser, so one strobe serves both; camserver's
  pair sync keeps their frames aligned (skew 0.05 ms measured).

**Steps.**

1. Firmware: laser enable on a GPIO, 15 Hz timer, `POST /strobe {period_us,
   phase_us}` + `GET /strobe`. Bench-check with a phone camera that the laser
   actually blanks (some modules keep a dim idle glow — measure it).
2. Source: tag frames on/off/split from the stripe score's row profile;
   expose the tag on `Frame`.
3. PLL: phase estimate from split frames, nudge loop at ~1 Hz, lock indicator
   in the eye overlay (`strobe: locked / hunting / no strobe`).
4. Scan on `on − off`; colour from the off frame; PLY with RGB
   (`write_ply` grows `red green blue`, `read_ply` already reads them).
5. Calibration-mode line fit and the plane collector on `on − off`.
6. Measure: stripe SNR on the drill's black rubber and blue battery before
   and after; false stripe pixels per frame; point noise (the error budget's
   stripe term should fall with a cleaner profile).

**Costs.** Half the frame rate for the scan (15 pairs/s); ESP32 wiring to the
laser; a second clock to keep honest; split frames wasted while hunting.

**Done when.** The overlay says `strobe: locked` for a minute without split
frames, the scan runs on subtracted frames, points carry the subject's colour,
and the black rubber that no measure recovered today appears in the cloud.

## Exposure loop from the detectors

Both cameras run auto-exposure metering their own scene; the left ran at 20 fps
under a 50 ms exposure, blurring the stripe and halving the pairing rate. Drive
`exposure_time_absolute` per camera from what the app measures — the stripe's
peak just under saturation, the board's white squares at a mid grey — with a
slow P-loop over camserver's control API. Same targets for both eyes. (The power line frequency is
already at 50 Hz on both cameras.)

## Colour normalisation from the board

These cameras expose no white balance (AWB off, no temperature, red/blue
balance read-only). The ChArUco's white squares are a neutral both eyes see:
per-channel gains from their means, applied on the GPU before the stripe
score and the display, EMA-smoothed, held when the board leaves. Optionally a
two-point fit from black and white squares to match contrast too. Makes the
redness threshold mean the same thing in both eyes, which the veto assumes.

## Small ones

- Pinned host buffers for the GPU downloads (0.5 → ~0.2 ms per frame).
- `--cv-threads 1` on the GPU path: 16 core-ms per pair against 22 at two,
  once the full ChArUco pass fits a frame at one thread.
- Per-voxel normals from the merged cloud, for shading and for meshing.
  Mind that `PointCloud.points()` is a view into an array `_reserve`
  rebinds as the cloud grows — copy before holding it across an `add`.
