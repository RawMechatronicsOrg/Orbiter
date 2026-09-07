"""COLMAP's sparse model: the five text files we write, and the two binary
files `image_undistorter` writes back.

The scanner already knows where every photo was taken from — the board pose it
solves and smooths for each frame pair — so COLMAP is never asked to recover
poses. It is handed a finished model and told to do the parts it is good at:
dense depth from many views, a mesh, an atlas. That makes this module a
translator, and everything that can go wrong here is a convention. Four are
worth naming.

**The pose is world→camera, and ours already is.** `images.txt` wants the
rotation and translation that carry a world point into the camera's frame. Our
stored pose is board→camera (`scanworker.PoseFix`), and the board *is* the
world: `xyz_cam = R xyz_board + t`, which `scanworker.py:665` runs backwards as
`xyz_board = Rᵀ(xyz_cam − t)`. So `(R, t)` is written through unchanged — no
inversion, no transpose. Inverting here would put every camera on the wrong
side of the board, and the model would still look plausible.

**The quaternion is Hamilton, scalar FIRST.** `Rotation.as_quat()` returns
scalar-LAST `(x, y, z, w)`. That reorder is the classic bug of every COLMAP
converter ever written, so it lives in exactly one function, `quat_wxyz`, and
has a test of its own.

**Millimetres.** COLMAP is scale-agnostic: the cloud, the poses and the seed
points are in millimetres on our side and stay that way on its.

**Tracks and POINTS2D are two views of one list.** A `Point3D`'s track names
`(IMAGE_ID, POINT2D_IDX)`, and that image's `points2d[idx]` names the point
back. The caller builds both in a single pass over one list of observations,
which is why no `POINT3D_ID` is ever `-1` and no index can shift — a point
dropped after its observations were written would otherwise leave a dangling
entry behind it.

What comes *back* is binary. `image_undistorter --output_type COLMAP` writes
`cameras.bin` / `images.bin`, not text, and this rig has TWO cameras with
different undistorted intrinsics and different undistorted sizes. So the
readers here resolve every camera in the file, and the caller picks each
image's `newK` through that image's own `camera_id`. Reading only the first
camera — all a single-camera pipeline ever needs — would warp every right-eye
mask it touched.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

#: PINHOLE's model id in a binary `cameras.bin`, and its four parameters
#: (fx, fy, cx, cy). Undistortion removes distortion, so this is the only model
#: `image_undistorter` writes. Anything else is refused rather than guessed at:
#: a model whose parameter count we do not know cannot be skipped over, and a
#: wrong count would mis-parse every camera after it instead of failing.
_PINHOLE_MODEL_ID = 1
_PINHOLE_PARAMS = 4

#: Bytes per POINTS2D entry in `images.bin`: X and Y as doubles, POINT3D_ID as
#: uint64. They are skipped wholesale — the observations that matter are the
#: ones we wrote going in.
_POINT2D_BYTES = 24


@dataclass
class EyeCamera:
    """One eye as a COLMAP camera: the intrinsics as solved, at the raw sensor
    size they were solved at."""

    #: 1 for the left eye, 2 for the right — the `CAMERA_ID` every image cites,
    #: and the `RIG_ID` of that camera's own single-sensor rig.
    camera_id: int
    side: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    #: OpenCV's five, `(k1, k2, p1, p2, k3)`, as `Eye.intrinsics_raw` stores them.
    dist: tuple[float, float, float, float, float] = (0.0,) * 5
    #: The distinct raw sizes this side's selected photos carry. Empty means
    #: nothing to check; anything other than exactly `(width, height)` is a
    #: refusal — see `write_cameras`.
    photo_wh: tuple[tuple[int, int], ...] = ()


@dataclass
class ImageRecord:
    """One photo: where it was taken from, which eye took it, and the seed
    points it observes."""

    image_id: int
    #: The file name COLMAP looks for under its `--image_path`, e.g.
    #: `left_0001.jpg`. `read_images_bin` speaks the same names.
    name: str
    camera_id: int
    #: Board→camera, i.e. world→camera. A right photo's is the composed pose
    #: (`stereo.compose_right_pose`), not the left's.
    R: np.ndarray = field(default_factory=lambda: np.eye(3))
    #: Millimetres.
    t: np.ndarray = field(default_factory=lambda: np.zeros(3))
    #: `(X, Y, POINT3D_ID)` per observation, in the order the tracks index.
    points2d: list[tuple[float, float, int]] = field(default_factory=list)


@dataclass
class Point3D:
    """One seed point, in the board frame, with the images that observe it."""

    point_id: int
    #: `(x, y, z)` in millimetres.
    xyz: np.ndarray = field(default_factory=lambda: np.zeros(3))
    #: The cloud's own colour, or mid-grey when the cloud is uncoloured.
    rgb: tuple[int, int, int] = (128, 128, 128)
    #: A placeholder. COLMAP reads it; the dense stage does not gate on it.
    error: float = 1.0
    #: `(IMAGE_ID, POINT2D_IDX)` per observation, length two or more.
    track: list[tuple[int, int]] = field(default_factory=list)


def quat_wxyz(R) -> tuple[float, float, float, float]:
    """A rotation matrix as a Hamilton quaternion, scalar FIRST — `(w, x, y, z)`,
    the order `images.txt` and `frames.txt` are read in.

    `scipy.spatial.transform.Rotation.as_quat()` returns scalar-LAST
    `(x, y, z, w)`. Writing that straight out gives every camera a
    plausible-looking wrong orientation, which is the easiest way there is to
    produce a model that loads, reconstructs, and is nonsense.

    The sign is canonicalised to `w >= 0`: q and −q are the same rotation, and a
    written line should not depend on which of the two the solver handed over.
    """
    x, y, z, w = Rotation.from_matrix(np.asarray(R, float)).as_quat()
    if w < 0.0:
        w, x, y, z = -w, -x, -y, -z
    return float(w), float(x), float(y), float(z)


def _g(v: float) -> str:
    """One number, the way a COLMAP text model writes them. `%.10g` keeps a
    double's meaningful digits without printing all seventeen, and is what the
    sibling converter emits."""
    return f"{float(v):.10g}"


def _pose_fields(R, t) -> str:
    """`QW QX QY QZ TX TY TZ` for one world→camera pose.

    Shared by `images.txt` and `frames.txt` on purpose: a single-sensor rig's
    `RIG_FROM_WORLD` *is* the image's own pose, and going through one formatter
    means the two files cannot drift apart whatever either writer does later.
    """
    w, x, y, z = quat_wxyz(R)
    tx, ty, tz = (float(v) for v in np.asarray(t, float).ravel())
    return " ".join(_g(v) for v in (w, x, y, z, tx, ty, tz))


def _open(path: str | Path):
    """A text model file, opened so a Windows host writes the same bytes a Linux
    one would. COLMAP's reader takes either line ending, but a model that
    changes shape with the machine that wrote it is a golden test nobody can
    keep."""
    return Path(path).open("w", encoding="utf-8", newline="\n")


def write_cameras(path: str | Path, eyes) -> None:
    """`cameras.txt`: one line per eye, at the raw sensor size its intrinsics
    were solved at.

    OPENCV carries `fx fy cx cy k1 k2 p1 p2`, which is exactly what this rig
    has: `intrinsics.py:467` solves with `cv2.CALIB_FIX_K3`, so k3 is zero and
    OPENCV is the honest model. A rig calibrated elsewhere could carry a k3, and
    FULL_OPENCV is written for that one — the guard is there so such a
    calibration is never silently truncated to eight parameters, and it is
    unreachable from this app's own solve.

    Refuses by name when a side's photos are not the size its intrinsics were
    solved at. A camera matrix is only valid at its own resolution — the same
    guard `Eye.intrinsics_for` applies live — and the consequence here is a
    whole model built around a principal point in the wrong place.
    """
    lines = []
    for eye in eyes:
        sizes = {(int(w), int(h)) for w, h in eye.photo_wh}
        solved = (int(eye.width), int(eye.height))
        if sizes and sizes != {solved}:
            seen = ", ".join(f"{w}x{h}" for w, h in sorted(sizes))
            raise ValueError(
                f"{eye.side}: intrinsics were solved at {solved[0]}x{solved[1]} "
                f"but its photos are {seen} — a camera matrix is only valid at "
                "the resolution it was solved at")
        k1, k2, p1, p2, k3 = (float(v) for v in eye.dist)
        if k3 == 0.0:
            model = "OPENCV"
            params = (eye.fx, eye.fy, eye.cx, eye.cy, k1, k2, p1, p2)
        else:
            # FULL_OPENCV's twelve: the eight above, then k3 and the rational
            # model's k4/k5/k6, for which OpenCV's five-parameter vector has no
            # room and which are therefore zero.
            model = "FULL_OPENCV"
            params = (eye.fx, eye.fy, eye.cx, eye.cy, k1, k2, p1, p2, k3, 0.0, 0.0, 0.0)
        lines.append(f"{int(eye.camera_id)} {model} {solved[0]} {solved[1]} "
                     + " ".join(_g(v) for v in params))
    with _open(path) as fh:
        fh.write("# Camera list with one line of data per camera:\n")
        fh.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        fh.write(f"# Number of cameras: {len(lines)}\n")
        for line in lines:
            fh.write(line + "\n")


def write_images(path: str | Path, images) -> None:
    """`images.txt`: two lines per image — the pose, then its POINTS2D.

    `(R, t)` goes in as stored, world→camera; the module docstring says why no
    inversion belongs here. `CAMERA_ID` is the eye's own, so a right photo is
    read through the right eye's intrinsics and its pose is the composed one —
    one pose written to both photos of a pair is wrong by the whole stereo
    baseline.
    """
    images = list(images)
    n_obs = sum(len(im.points2d) for im in images)
    mean_obs = n_obs / len(images) if images else 0.0
    with _open(path) as fh:
        fh.write("# Image list with two lines of data per image:\n")
        fh.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        fh.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        fh.write(f"# Number of images: {len(images)}, "
                 f"mean observations per image: {_g(mean_obs)}\n")
        for im in images:
            fh.write(f"{int(im.image_id)} {_pose_fields(im.R, im.t)} "
                     f"{int(im.camera_id)} {im.name}\n")
            fh.write(" ".join(f"{_g(x)} {_g(y)} {int(pid)}"
                              for x, y, pid in im.points2d) + "\n")


def write_points3d(path: str | Path, points) -> None:
    """`points3D.txt`: the seed cloud with the tracks that make it useful.

    The tracks are the load-bearing part. `patch_match_stereo` reads each
    image's depth range from the 3D points that image observes, and `__auto__`
    resolves source images by shared observations — a model with points but no
    tracks gives it neither.
    """
    points = list(points)
    n_track = sum(len(p.track) for p in points)
    mean_track = n_track / len(points) if points else 0.0
    with _open(path) as fh:
        fh.write("# 3D point list with one line of data per point:\n")
        fh.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, "
                 "TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        fh.write(f"# Number of points: {len(points)}, "
                 f"mean track length: {_g(mean_track)}\n")
        for p in points:
            x, y, z = (float(v) for v in np.asarray(p.xyz, float).ravel())
            r, g, b = (int(v) for v in p.rgb)
            track = " ".join(f"{int(image_id)} {int(idx)}" for image_id, idx in p.track)
            fh.write(f"{int(p.point_id)} {_g(x)} {_g(y)} {_g(z)} {r} {g} {b} "
                     f"{_g(p.error)} {track}\n")


def write_rigs(path: str | Path, eyes) -> None:
    """`rigs.txt`: one rig per camera, each holding that camera alone.

    A COLMAP 4.x text model carries rigs and frames, and these columns are the
    sibling converter's (`sfm_priors_to_colmap.py:194-206`). A single-sensor rig
    needs no sensor pose, so the optional `HAS_POSE` block is absent and
    `RIG_ID` is simply the camera's own id.
    """
    eyes = list(eyes)
    with _open(path) as fh:
        fh.write("# Rig list with one line of data per rig:\n")
        fh.write("#   RIG_ID, NUM_SENSORS, REF_SENSOR_TYPE, REF_SENSOR_ID, "
                 "SENSORS[] as (SENSOR_TYPE, SENSOR_ID, HAS_POSE, "
                 "[QW, QX, QY, QZ, TX, TY, TZ])\n")
        fh.write(f"# Number of rigs: {len(eyes)}\n")
        for eye in eyes:
            fh.write(f"{int(eye.camera_id)} 1 CAMERA {int(eye.camera_id)}\n")


def write_frames(path: str | Path, images) -> None:
    """`frames.txt`: one frame per image, columns from the sibling converter
    (`sfm_priors_to_colmap.py:207-227`).

    `FRAME_ID` is the image's id and `RIG_ID` its camera's, because each camera
    is a rig of its own. That rig's reference sensor is the camera and carries
    no offset, so `RIG_FROM_WORLD` is the image's own pose — written through
    `_pose_fields`, the formatter `images.txt` uses, so the two files agree byte
    for byte.
    """
    images = list(images)
    with _open(path) as fh:
        fh.write("# Frame list with one line of data per frame:\n")
        fh.write("#   FRAME_ID, RIG_ID, RIG_FROM_WORLD[QW, QX, QY, QZ, TX, TY, TZ], "
                 "NUM_DATA_IDS, DATA_IDS[] as (SENSOR_TYPE, SENSOR_ID, DATA_ID)\n")
        fh.write(f"# Number of frames: {len(images)}\n")
        for im in images:
            fh.write(f"{int(im.image_id)} {int(im.camera_id)} "
                     f"{_pose_fields(im.R, im.t)} "
                     f"1 CAMERA {int(im.camera_id)} {int(im.image_id)}\n")


def read_pinhole_bin(path: str | Path) -> dict[int, tuple[np.ndarray, int, int]]:
    """Every camera of a binary `cameras.bin`, as `{camera_id: (K, width, height)}`.

    This is the undistorted camera `image_undistorter` chose — its own `newK`
    and its own output size — and there is one per eye. The reference reads only
    the first (`pipeline.py:1430-1445`) because it runs `single_camera=1`; we
    have two, and using the left's `newK` on a right image warps every mask it
    touches, so every entry is resolved and the caller looks its image up by
    `camera_id`.

    The layout is a `uint64` camera count, then per camera a `uint32` id, an
    `int32` model, `uint64` width and height, and one `double` per model
    parameter.
    """
    path = Path(path)
    out: dict[int, tuple[np.ndarray, int, int]] = {}
    with path.open("rb") as fh:
        (n_cameras,) = struct.unpack("<Q", fh.read(8))
        for _ in range(n_cameras):
            camera_id, model_id = struct.unpack("<Ii", fh.read(8))
            width, height = struct.unpack("<QQ", fh.read(16))
            if model_id != _PINHOLE_MODEL_ID:
                raise ValueError(
                    f"{path.name}: camera {camera_id} has model id {model_id}, "
                    "not the PINHOLE image_undistorter writes")
            fx, fy, cx, cy = struct.unpack(f"<{_PINHOLE_PARAMS}d",
                                           fh.read(8 * _PINHOLE_PARAMS))
            out[int(camera_id)] = (
                np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], float),
                int(width), int(height))
    return out


def read_images_bin(path: str | Path) -> dict[str, int]:
    """The `{name: camera_id}` map of a binary `images.bin`.

    One job, deliberately. The poses in the dense workspace are the ones we put
    there — undistortion changes the intrinsics, not the pose — so
    `photos.jsonl` stays the authority on where a photo was taken from, and all
    this reader is asked is which eye's undistorted intrinsics a given file
    belongs to. It answers the `--image_list_path` post-condition too: whether
    the workspace holds exactly the images we listed.

    Per image: a `uint32` id, four `double` qvec, three `double` tvec, a
    `uint32` camera id, the name as NUL-terminated bytes, then a `uint64`
    observation count and 24 bytes per observation.
    """
    out: dict[str, int] = {}
    with Path(path).open("rb") as fh:
        (n_images,) = struct.unpack("<Q", fh.read(8))
        for _ in range(n_images):
            fh.read(4)                                   # image_id
            fh.read(56)                                  # qvec (4d) + tvec (3d)
            (camera_id,) = struct.unpack("<I", fh.read(4))
            name = bytearray()
            while (ch := fh.read(1)) not in (b"\x00", b""):
                name += ch
            (n_points2d,) = struct.unpack("<Q", fh.read(8))
            fh.read(n_points2d * _POINT2D_BYTES)
            out[name.decode("utf-8")] = int(camera_id)
    return out
