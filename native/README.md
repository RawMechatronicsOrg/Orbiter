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

**No board pose yet.** `model.camera_fx/fy/cx/cy` and `camera_distortion` were
solved for the *phone* on the `camera_url` path — a different lens at a
different resolution from either camera of this pair. Feeding them to
`solvePnP` yields a plausible pose that is simply wrong. Until the pair has its
own calibration the panel says `pose needs per-eye intrinsics` rather than
showing a number. Per-eye intrinsics and `stereoCalibrate` are the next piece
of work, and this workbench is where the corners they need are already visible.

**The laser fit is not yet a calibration.** It produces the per-frame input —
subpixel stripe points on the board, and the line they fit — but turning those
into the camera↔laser geometry needs per-eye intrinsics first, for the same
reason board pose does. See "Laser" below.

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

## Measured on this machine

Live 1280×720 frames from the pair, isolated:

```
imdecode COLOR     2.2 ms   (1.4 ms grayscale-only, + 0.2 ms for the gray view)
charuco detect    13.1 ms   (6.4 ms at half scale; 2.8 ms with no board in view)
redness           0.7 ms    (board bbox; 3.1 ms over the full frame)
laser line         ~8 ms    (redness + centroids + RANSAC + fit)
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
