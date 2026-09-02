"""Adapter onto the server's calibration numerics.

Everything ChArUco-shaped in this app goes through here, so there is exactly
one place that knows how to reach into the sibling `orbiter-server` package.

Why import it rather than reimplement: `orbiter_server/calibration.py` already
carries the detection and pose code with its hard-won details — notably the
flat-board planar-PnP ambiguity, where `solvePnP` flips between two solutions
and shows up as a ~10-30° rotation error with only a modest translation error.
Writing a second copy of that here would eventually disagree with the one the
calibration actually uses, and the disagreement would be invisible until a
solve came out wrong.

The functions used are pure — they take a board and intrinsics as arguments and
never read the server's global model — so calling them from a desktop app is
sound. Importing the module does pull the server's world (httpx, scipy,
pytransform3d, pydantic-settings), which is why `orbiter-server` must be
installed alongside this package.
"""

from __future__ import annotations

from typing import Any

import numpy as np

_IMPORT_HINT = (
    "orbiter-native needs the sibling orbiter-server package for its "
    "calibration numerics. Install both editable from the repo root:\n"
    "    pip install -e ./server -e ./native"
)

try:
    # Importing the package is what puts its directory on sys.path — the
    # modules inside use bare-name imports (`import config`, `from geom.rig
    # import ...`) and its __init__ exists to make that work. So this import
    # is load-bearing, not decorative; do not "clean it up".
    import orbiter_server  # noqa: F401

    from calibration import (  # type: ignore[import-not-found]
        BoardSpec,
        Intrinsics,
        _build_board,
        detect_board,
        estimate_board_pose,
        estimate_board_pose_disambiguated,
    )
except ImportError as exc:  # pragma: no cover - environment problem, not logic
    raise ImportError(f"{exc}\n\n{_IMPORT_HINT}") from exc


__all__ = [
    "BoardSpec",
    "Intrinsics",
    "build_board",
    "charuco_detect",
    "estimate_pose",
    "board_spec_from_config",
    "intrinsics_from_eye",
]


def build_board(spec: BoardSpec):
    """Construct the `cv2.aruco.CharucoBoard` for a spec."""
    return _build_board(spec)


def charuco_detect(gray: np.ndarray, board):
    """(corners, ids) or (None, None). Wraps `calibration.detect_board`,
    which already returns (None, None) below four corners."""
    return detect_board(gray, board)


#: OpenCV's ChArUco frame has its origin at a corner, y running DOWN the
#: printed face and z pointing INTO the board. Measured, not assumed: on a
#: straight-on synthetic view the board's z came back as (0.003, 0.04, 0.999)
#: in camera coordinates — along the viewing direction, away from the camera.
#: "Above the board" was therefore behind it, and the scan volume kept
#: nothing but the board's own surface noise. This rotation (180° about x)
#: makes z point OUT of the printed face, toward the camera, with y up.
_FACE_OUT = np.diag([1.0, -1.0, -1.0])


def _facing(board, R: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Re-express an OpenCV board pose in the centred, face-out frame.

    A point p' there is p = FACE_OUT·p' + centre in OpenCV's frame, so
    R' = R·FACE_OUT and t' = t + R·centre.
    """
    sx, sy = board.getChessboardSize()
    sq = float(board.getSquareLength()) * 1000.0     # the board is built in metres
    centre = np.array([sx * sq / 2.0, sy * sq / 2.0, 0.0])
    return R @ _FACE_OUT, np.asarray(t, float).ravel() + R @ centre


def estimate_pose(corners, ids, board, intrinsics: Intrinsics,
                  R_predicted=None) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Board→camera `(R, t_mm, ambiguity_deg)`, or None.

    The board frame handed out here is NOT OpenCV's raw one: its origin is the
    board's centre and its z axis points out of the printed face, toward the
    camera, so "above the board" is +z and a volume centred on the board is
    centred on the frame. See `_FACE_OUT` for why. Every consumer in this app
    — the scan volume, the laser-plane collector, the overlay — gets the pose
    from here and nowhere else, so the convention lives in one place.

    A flat board admits TWO poses that reproject almost identically, and a bare
    `solvePnP` flips between them — a ~10-30 degree rotation error with only a
    modest translation error. On this rig that would flip the scanning volume.

    `R_predicted` breaks the tie. The caller supplies the previous frame's
    rotation: at 30 fps the board barely moves between frames, so the last pose
    is a good prior for the next. With no prior — the first frame after the
    board appears — this falls back to the plain solve, whose answer is
    IPPE's best-reprojection candidate. `ambiguity_deg` reports how far apart
    the two candidates were, so a pose that depended heavily on the prior is
    visible rather than implied.
    """
    # solvePnP's DLT needs six correspondences and raises below that. A board
    # seen by its edge gives four or five, every frame; raising there was
    # logged as a detector error and blanked the eye's view as "offline".
    if corners is None or len(corners) < 6:
        return None
    if R_predicted is not None and not _is_rotation(R_predicted):
        R_predicted = None                  # a poisoned prior is not a prior
    if R_predicted is None:
        out = estimate_board_pose(corners, ids, board, intrinsics)
        if out is None:
            return None
        R, t, ambiguity = out[0], out[1], 0.0
    else:
        # The prior is in this app's frame; the server's solver wants its own.
        out = estimate_board_pose_disambiguated(corners, ids, board, intrinsics,
                                                R_predicted @ _FACE_OUT)
        if out is None:
            return None
        R, t, ambiguity = out
    R, t = _facing(board, R, t)
    if not _is_rotation(R) or not np.isfinite(t).all():
        return None                         # not a pose; do not hand it out
    return R, t, ambiguity


def _is_rotation(R) -> bool:
    """Is this a rotation matrix, and not a hole where one should be?

    A degenerate corner set can come back from solvePnP as NaN. Handed out,
    it puts a NaN pose into the calibration sets; kept by the caller as the
    next frame's prior, it raises inside the disambiguating solve — and then
    every frame after it, because the caller keeps the prior it was given and
    never sees a pose again. Both ends are checked here: nothing that is not
    a rotation is returned, and nothing that is not a rotation is believed.
    """
    R = np.asarray(R, dtype=float)
    if R.shape != (3, 3) or not np.isfinite(R).all():
        return False
    return abs(float(np.linalg.det(R)) - 1.0) < 1e-3


def board_spec_from_config(cfg: dict[str, Any]) -> BoardSpec | None:
    """Build a `BoardSpec` from a `GET /config` payload.

    Returns None when the board params are absent or unusable, so the caller
    can show "no board configured" instead of detecting against a bogus board.
    """
    try:
        return BoardSpec(
            squares_x=int(cfg["charuco_squares_x"]),
            squares_y=int(cfg["charuco_squares_y"]),
            square_length_mm=float(cfg["charuco_square_length_mm"]),
            marker_length_mm=float(cfg["charuco_marker_length_mm"]),
            aruco_dict_id=int(cfg["aruco_dict_id"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def intrinsics_from_eye(
    eye: dict[str, Any] | None, frame_wh: tuple[int, int] | None = None
) -> Intrinsics | None:
    """Per-eye intrinsics, when the eye's config carries usable ones.

    IMPORTANT: this deliberately does NOT fall back to the model's
    `camera_fx/fy/cx/cy` + `camera_distortion`. Those were solved for the
    PHONE on the `camera_url` path — a different lens at a different
    resolution from either camera of this pair. Feeding them to `solvePnP`
    would produce a plausible-looking pose that is simply wrong, and wrong
    quietly. Until the pair has its own calibration, no pose is the honest
    answer, and the UI says so.

    `frame_wh` guards the same failure one step further on. A camera matrix is
    only valid at the resolution it was solved at, and camserver can be
    reconfigured between runs; 1280x720 intrinsics applied to a 1080p frame
    put the principal point in the wrong place and scale the focal length by
    two thirds, which again yields a plausible, wrong pose. When the size does
    not match, this returns None rather than rescaling — rescaling would be a
    guess about the sensor's crop-vs-scale behaviour that nobody has verified.
    """
    if not eye:
        return None
    k = eye.get("intrinsics")
    if not isinstance(k, dict):
        return None
    try:
        stored_wh = (int(k["width"]), int(k["height"]))
        if frame_wh is not None and tuple(frame_wh) != stored_wh:
            return None
        dist = tuple(float(v) for v in k.get("dist") or (0.0,) * 5)
        return Intrinsics(
            fx=float(k["fx"]), fy=float(k["fy"]),
            cx=float(k["cx"]), cy=float(k["cy"]),
            dist=dist,
        )
    except (KeyError, TypeError, ValueError):
        return None
