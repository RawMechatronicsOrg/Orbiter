# Orbiter native CV workbench

A desktop client for the binocular camera pair — the frame-rate work that a
browser cannot do.

The web UI can *display* the pair, but a cross-origin `<img>` taints the
canvas, so it can never look inside a frame. Corner detection, laser search and
honest timing need the pixels; they live here.

## What owns what

| | |
|---|---|
| **Server** (`../server`) | Owns state. This app reads `GET /config`. |
| **Web Stereo tab** (`../ui`) | Sets the run's baseline: which camera is which eye, per-eye orientation, nominal distance between cameras. |
| **This app** | Consumes that baseline, pulls frames straight from camserver, runs the detectors. |

The panels here are deliberately read-only. Two editors for one setting is how
a rig ends up configured differently depending on which window you looked at
last. Change orientation in the web tab, press Apply, and this window follows
within a poll interval.

## Install

Needs Python ≥ 3.11 (the server package requires it) and the sibling
`orbiter-server` source tree for its calibration numerics.

```bash
python -m venv native/.venv
```
```bash
native/.venv/Scripts/pip install -e ./server -e ./native
```

## Run

```bash
native/.venv/Scripts/orbiter-native
```

`--server URL` points at a non-default Orbiter server (default
`http://localhost:8000`); `-v` turns on debug logging.

## Design notes

**Two threads per eye, not one.** The reader thread does nothing but drain the
socket and decode, always at line rate, keeping only the newest frame; the
detector thread takes whatever is newest whenever it is free. Frames arriving
during a detection pass are skipped, never queued. Reading and detecting in one
loop would leave the socket undrained while detection runs, and the TCP buffer
would fill with frames nobody has read — the view then silently becomes a
recording running further and further behind, at a frame rate that still looks
correct.

**The GUI thread only paints.** Each worker leaves its newest result in a
one-slot mailbox and a 30 Hz timer in the window takes whatever is there; the
scan maths runs in a thread of its own (`scanworker.py`) that pairs the eyes'
results by camserver's capture clock. Nothing is delivered through a queued
Qt signal per frame: Qt's queue never drops, so a GUI thread that falls behind
backlogs frames without bound. That was the first design, and at 1080p it was
doing more than a second of work per second — triangulation, a numpy
BGR→RGB pass and QImage copy per frame, a min/max over the whole cloud per
pair — and fell further behind the cameras the longer it ran.

**Orientation is flip-then-rotate**, and that order is a contract shared with
`server/orbiter_server/commands.py::_cmd_set_stereo_rig` and
`ui/src/viewer/StereoView.tsx::eyeTransform`. If the three drift, the operator
aligns the rig against a preview the solver never sees.

**Detection runs on original pixels, display is oriented afterwards.**
Detecting on the oriented image would report corner coordinates in a frame that
depends on a UI setting — useless to any calibration consumer.

**ChArUco detection is imported, not reimplemented.** `calibration.detect_board`
and `estimate_board_pose_disambiguated` already carry the flat-board planar-PnP
ambiguity handling; a second copy here would eventually disagree with the one
the calibration actually uses, invisibly.

## Known limits

**Board pose needs a calibration first.** `model.camera_fx/fy/cx/cy` and
`camera_distortion` were solved for the *phone* on the `camera_url` path — a
different lens at a different resolution from either camera of this pair.
Feeding them to `solvePnP` yields a plausible pose that is simply wrong, so the
panel says `pose needs per-eye intrinsics` until this pair has its own. Run the
calibration below; pose appears once the solve is saved. Stored intrinsics also
carry the resolution they were solved at and are refused against a frame of any
other size — camserver can be reconfigured under a running app, and 1280x720
intrinsics on a 1080p frame put the principal point in the wrong place and
scale the focal length by two thirds.

**The laser plane is not solved.** Scanning does not need it — triangulation
from two cameras gives the 3D point directly — but it would be a second,
independent check on every point, and it is not implemented.

**OpenCV version skew.** The server image pins `opencv-python-headless>=4.7`
and resolves to 4.x; this venv resolved 5.0. Both work — the aruco API and a
build-board/detect round trip were checked on 5.0 — but detection results are
not guaranteed bit-identical between the two.

## Laser

The stripe is found in **colour, not luminance**, and only **inside the board**.

Colour because the board is black and white and the laser is red. On a frame
from this rig there are 68142 board pixels brighter than gray 170 — the white
squares — and exactly 31 of them survive a redness threshold of 50. A
brightness threshold does not find the stripe on this board; it finds the white
squares. `redness = r - max(g, b)` has an on-board median of 0 against a stripe
peaking near 110.

Inside the board because the stripe carries on across the workbench, and those
points are not on the board plane. The mask is the convex hull of the DETECTED
ChArUco corners — deliberately conservative: on a circular board most corners
are missing, so the hull covers less than the physical disc and discards usable
signal near the edges. Fewer certainly-valid points beat more points of unknown
provenance.

The fit is RANSAC followed by total least squares. A plane meeting a plane is a
line, so the stripe must be straight and the fit doubles as a validity check —
the reported RMS is what tells the operator whether a frame is worth keeping.
The per-scanline centroid wanders where the stripe crosses a dark square; on a
real frame robust fitting gave 0.67 px RMS against 1.88 px for a plain fit over
the same points.

Live, both eyes with board and laser running together: ~29 fps each, RMS 0.61
and 0.69 px, ~8 ms for the laser stage.

Per frame the detector yields the inlier points in original image coordinates.
That is the calibration payload: back-project each point, intersect it with the
board plane, accumulate across board poses, fit the plane the laser sweeps.
Which needs intrinsics — see the known limits above.

## Calibration

Move the board around in front of the pair with **auto-capture** on. Views are
kept only when they show the solve something new — a different place in the
frame, a different distance, or a different tilt — and only when the board is
still, because motion blur rounds corners off and biases the solve.

They are captured in **pairs**, matched on camserver's capture clock (both
cameras are timed by that same clock, so it is what makes two frames
simultaneous; arrival times through two sockets are not comparable). Intrinsics
do not need the pairing. `stereoCalibrate` does, and sweeping the board twice
would be a waste of your time.

### Why the panel nags about tilt

`calibrateCamera` returns a confident answer from head-on views and that answer
can be badly wrong, with a reprojection RMS that looks perfect. Measured on
synthetic views of this board with 0.15 px corner noise:

```
tilt spread   RMS px   focal-length error
     5.27      0.204        +108 px
     5.45      0.204        +114 px
     6.04      0.204         +31 px
     8.12      0.205          +8 px
    12.09      0.205          +3 px
    16.06      0.206          +1 px
```

The RMS is flat to three decimals across a focal length wrong by 12%. With no
noise at all, a fronto-parallel set solves to RMS 0.0000 px and a focal length
27% low. Focal length and board distance are not separable in a head-on view —
only tilt separates them. So the solve is **refused** below a tilt spread of 8,
and the live figure is shown while you can still act on it. The coverage grids
are the same idea for distortion, which is only measurable where the board
actually went.

The distortion model frees k1, k2, p1, p2 and fixes k3: k3 is only identifiable
from views pushing the board into the frame corners, and left free on a modest
set it absorbs error and destabilises the focal length. Tangential terms stay
free because these are inexpensive sensors where a tilted element is real — the
server's phone-lens solve fixes them, but that was a different lens.

**Save to server** stores the result through `POST /command/set_stereo_rig`,
the same command the web tab uses, so the server remains the one owner of this
state.

## Scanning

With the pair and the laser plane calibrated and the laser detector on,
**scanning** turns the stripe into points and accumulates a cloud.

Each stripe pixel in the left eye is a ray, and the ray meets the calibrated
laser plane in one point. That point is projected into the right eye, and it
counts only if the right eye saw stripe there too — within 3 px, the
calibration's own slack. The survivors are averaged per scanline into a
sub-pixel centroid, and the centroid's ray meets the plane for the point that
is kept. No epipolar search, no matching threshold: the right eye vetoes, it
does not measure.

**Why not stereo triangulation.** The first version matched left stripe
centroids to the right eye's stripe along epipolar lines and triangulated. On
this bench 16-24% of image columns hold the stripe AND something else red — a
wire, a reflection — and the centroid of such a column is neither; where it
found no match, a one-eye fallback put it on the plane unverified. Checking
each *pixel* against the right eye before any centroid is taken removes the
wire from the average instead of averaging it in. The plane is also the more
precise instrument on this rig: the sheet passes 74 mm from the left camera,
so a 0.5 px centroid gives about 1.2 mm at half a metre, against 2.4 mm from
the 144 mm stereo baseline at its 1.95 px fit.

**Stripe on coloured surfaces.** `r - max(g, b)` loses the stripe on blue: the
surface's own blue exceeds the laser's red. The laser adds only red, and only
along a thin line, so the detector also measures redness *in excess of the
local background* — each channel minus its morphological opening — and takes
the larger of the two. Measured on the drill's blue battery: 66 against a
background of 46, where plain redness gave 47 against 40. Dark rubber reflects
almost no red at all, and no measure recovers it.

The volume is a **cylinder standing on the board**, in the board's frame, so
it stays put as the board moves and "above" keeps meaning above the board.
That frame is centred on the board with z out of the printed face
(`cvcore.estimate_pose`), not OpenCV's raw one — whose origin is a corner and
whose z points *into* the board, as measured on a straight-on view; in that
frame the box selected the space behind the board and kept nothing but the
board's own surface noise. A cylinder rather than a box because the board is a
disc and the subject stands on it: the wall behind the bench sits inside a
box's corners once the board is tilted, and outside a disc the size of the
board. Points below a 5 mm floor are the stripe on the board itself.

The board must be visible for scanning to work: it is what defines where the
volume is.

While scanning, the cloud so far is drawn over both eyes in orange, each eye
projecting it through its own board pose — so the two overlays disagreeing is
itself a sign that the board poses do. The overlay is decimated to about 40k
points; the export (**Export PLY**, binary little-endian) carries everything.

## Measured on this machine

Live 1280×720 frames from the pair, isolated:

```
imdecode COLOR     2.2 ms   (1.4 ms grayscale-only, + 0.2 ms for the gray view)
charuco detect    13.1 ms   (6.4 ms at half scale; 2.8 ms with no board in view)
redness           0.7 ms    (board bbox; 3.1 ms over the full frame)
laser line         ~8 ms    (redness + centroids + RANSAC + fit)
```

Scan mode at 1920×1080, live frames, ~1700 stripe points per eye, before and
after the kernels were rewritten (`laser.py`, `scan.py`, `orient.py`):

```
stripe points (no fit)    20.7 ms  →  5.3 ms   (OpenCV redness, band-limited centroids)
epipolar × polyline       33.6 ms  →  2.9 ms   (float32 sweep in 128-line chunks)
scan_frame, whole         31.9 ms  →  4.0 ms
orient mapping, 1700 pts   1.5 ms  →  0.01 ms  (vectorised)
cloud bounds at 1M pts    42.0 ms  →  0        (kept running)
frame to QImage, per eye   8.6 ms  →  0        (wrapped as BGR888, no copy)
```

Colour decode costs about 1 ms per frame per camera over grayscale, and buys
the only channel the stripe exists in. Redness is computed inside the board's
bounding box, not the whole frame.

Both eyes running together: ~29 fps of detection each against a 30 fps stream.

## Tests

```bash
native/.venv/Scripts/python -m pytest native/tests -q
```

Covers orientation (permutation, point mapping, and the flip-then-rotate order
against the CSS contract), the laser line (subpixel accuracy against a known
line, redness vs bright neutral pixels, mask confinement, outlier rejection,
determinism), the MJPEG demultiplexer across hostile chunk boundaries, and
config parsing — including that phone intrinsics never stand in for an eye's
own. The threads and widgets are checked by running the app.
