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

**The laser detector needs an actual laser.** It is off by default and its
threshold defaults to 230. A line laser saturates the sensor where it lands, so
that is the signal. Measured on a lit workbench frame from this rig with no
laser powered: a floor of 60 reports a line in all 1280 columns and 180 still
reports 974 — a low threshold does not find a faint laser, it finds the scene
and reports a confident subpixel centroid for it.

**OpenCV version skew.** The server image pins `opencv-python-headless>=4.7`
and resolves to 4.x; this venv resolved 5.0. Both work — the aruco API and a
build-board/detect round trip were checked on 5.0 — but detection results are
not guaranteed bit-identical between the two.

## Measured on this machine

Live 1280×720 frames from the pair, isolated:

```
imdecode GRAY      2.2 ms
charuco detect    13.1 ms   (6.4 ms at half scale; 2.8 ms with no board in view)
laser centroid     3–4 ms   (full frame, vectorised numpy)
```

Both eyes running together: ~28 fps of detection each against a 30 fps stream.

## Tests

```bash
native/.venv/Scripts/python -m pytest native/tests -q
```

Covers orientation (permutation, point mapping, and the flip-then-rotate order
against the CSS contract), the laser centroid, the MJPEG demultiplexer across
hostile chunk boundaries, and config parsing — including that phone intrinsics
never stand in for an eye's own. The threads and widgets are checked by running
the app.
