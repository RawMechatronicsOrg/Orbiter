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
under a 50 ms exposure, blurring the stripe and halving the pairing rate.

**These cameras do not accept `exposure_time_absolute`.** Measured 2026-09-02:
the write reports `honoured=false` with a readback of 0, in Shutter Priority
the ioctl returns EACCES, and `exposure_dynamic_framerate` returns EIO. Tested
against the ends of the range — 200 ms and 1 ms give the same frame. What does
work is `auto_exposure=1` (Manual Mode), and it is most of the win: the frame
rate went 20.9 → 28.5 fps and blown highlights 1.86% → 0.2%, because the
camera stops hunting and holds a fixed, shorter exposure. `brightness` is a
digital level shift, not exposure (mean 122 → 49, frame rate unchanged), and
there is no gain control at all.

So the loop has nowhere to push on this hardware. It waits on either a camera
that honours the control, or the strobe above, which makes the exposure
question mostly moot. Note also that camera controls do not always survive a
stream being reopened — read them back (`orbiter-rigcheck`) after a restart.

## Colour normalisation from the board

These cameras expose no white balance (AWB off, no temperature, red/blue
balance read-only). The ChArUco's white squares are a neutral both eyes see:
per-channel gains from their means, applied on the GPU before the stripe
score and the display, EMA-smoothed, held when the board leaves. Optionally a
two-point fit from black and white squares to match contrast too. Makes the
redness threshold mean the same thing in both eyes, which the veto assumes.

## The sheet cannot be re-solved after a restart

Captured views survive a restart (`~/.orbiter-native/calib-views.npz`); the
plane collector's raw frames do not. So a session that improves the intrinsics
cannot re-derive the sheet through them until the operator sweeps the stripe
across the board again — and until it does, what the server holds is a camera
from this session and a sheet from a previous one. That mixture is exactly
what the right-eye veto refuses, and on 2026-09-02 it cost an evening: the
eyes disagreed by 52 px against a 3 px tolerance, then by 10 px once the
intrinsics were re-solved, and the sheet could not follow.

Persist the plane collector the way the sample set is persisted: the raw
stripe pixels, corners, ids and `R_hint` per frame, keyed to the board spec.
Then a restart can refit the sheet through whatever intrinsics are in force,
which is what every cycle already tries to do.

## Small ones

- Pinned host buffers for the GPU downloads (0.5 → ~0.2 ms per frame).
- `right.mask(confirm_px)` and `scan._whole_blobs` each allocate a full-frame
  `uint8` image and dilate it (7x7 and 17x1 at defaults) on every pair, while
  the pixels that matter live inside `stripe_rows`. Two 2 MP passes at 1080p
  that could be band-sized.
- `laser._lit` decides `along_x` from the extent of the lit pixels, and in
  scan mode it only ever sees the row band — whose height is a fraction of the
  frame's width, so the answer is pinned to True. Harmless on this rig, where
  the sheet does lie across the rows, but it is not a measurement any more.
  Decide it from the geometry (`stripe_rows` already has the sheet) instead.
- `average_still` sorts and trims each axis independently, so the point it
  produces may match no observation, and a glint offset mainly along one axis
  is trimmed only on that axis. A trim on distance from the median point would
  keep the observations whole.
- `rolling.solve_readout` accepted 59 ms for the right eye — longer than the
  33 ms frame interval, which no rolling shutter can be. Refuse a readout
  longer than the frame interval rather than storing it.
- The bar for the pair geometry is dropped when new intrinsics are accepted
  (`_retire_dependants`), so the first stereo solve of the new generation
  lands whatever its residual. Correct — consistency beats a number measured
  through another camera — but it means one unmeasured solve each time. Seed
  the new bar from the same cycle instead.
- `--cv-threads 1` on the GPU path: 16 core-ms per pair against 22 at two,
  once the full ChArUco pass fits a frame at one thread.
- Per-voxel normals from the merged cloud, for shading and for meshing.
  Mind that `PointCloud.points()` is a view into an array `_reserve`
  rebinds as the cloud grows — copy before holding it across an `add`.
