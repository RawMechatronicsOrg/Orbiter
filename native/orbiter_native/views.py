"""Which photographs COLMAP is given, which points it is seeded with, and
which of them each photograph observes.

The scan hands over two things a reconstruction cannot get anywhere else: a
cloud of confident points in the board frame, and a pose for every photograph
in that same frame. Everything in this module is one of those two turned into
a decision.

**The cloud chooses the photographs.** Projected through a photograph's own
pose and its own eye's intrinsics, the cloud says whether that photograph looks
at the subject at all, how much of the frame the subject fills and at what
distance. A photograph that fails those is not a photograph of this object, and
including it costs dense time and pollutes the atlas. Nothing here looks at
pixels: the decision is geometry, and it is made hours after the frame itself
is gone.

**The cloud is also the sparse model.** `patch_match_stereo` reads each image's
depth range from the 3D points that image observes, and `__auto__` resolves
source images by shared observations — so a model with points but no tracks
gives it neither. The seeds are a voxel subsample of the cloud, capped, and the
tracks are the same visibility test the selection used.

**POINTS2D and TRACK are two views of one list.** `tracks` walks its
observations once and appends to both sides at the same instant, which is why
no `POINT3D_ID` is ever `-1` and no index can shift. Building one and then
filtering the other is the bug this module is shaped to make impossible.

**Occlusion is not solved, and is not claimed to be.** A point counts as seen
when it is in front of the camera, inside the image, and its normal faces the
camera (`n · d < 0`). That is enough for depth ranges and for coverage; a point
behind the subject with an outward normal pointing away is excluded, a point
behind it whose normal happens to face the lens is not (§2.6).

**Coverage is measured against the selection, never against the object.** The
view buckets bin the photographs that were *selected* — which is to say, the
ones that were captured. A direction nobody ever photographed is not a failure
of the clean pass, and the gate must not report it as one.

**The sources COLMAP compares are ours; the list of images is not.**
The last thing here rewrites the `patch-match.cfg` `image_undistorter`
wrote. Its image lines are the authoritative record of what COLMAP
registered, so they are kept exactly and only the source lines are
replaced — with sources bucketed by angle, because the nearest view is
the one baseline that cannot triangulate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from .colmapio import EyeCamera, Point3D
from .photos import (
    POSE_CONVENTION,
    SCHEMA,
    SIDES,
    BoardSnapshot,
    Extrinsics,
    EyeSnapshot,
    RigSnapshot,
    camera_centre,
    view_direction,
)
from .scan import ScanVolume

#: `cameras.txt`'s ids, fixed: the left eye is camera 1 and the right is
#: camera 2. Every image line, every rig and every frame cites one of these
#: two, so the mapping is written down once.
CAMERA_ID = {"left": 1, "right": 2}

#: The two reconstruction modes. `texture_set` needs to know which one it is
#: in because the same fallback set is labelled differently in each: in
#: texture-only the workspace is undistorted from the verbatim
#: `colmap/images/`, in dense from the inpainted `colmap/images_clean/`.
MODES = ("texture-only", "dense")
FALLBACK_SOURCE = {"texture-only": "raw", "dense": "inpainted"}

#: A point sitting on the camera's own plane projects to infinity. The same
#: epsilon `stereo.StereoRig.project_right` uses, for the same reason.
_MIN_DEPTH_MM = 1e-6


@dataclass(frozen=True)
class SelectParams:
    """Every threshold the selection, the buckets and the texture gate use.

    One dataclass so a number cannot be passed to one place and not another,
    and so `session.json` can carry the lot verbatim and a run can be told
    apart from another run by reading one file.
    """

    #: Seed points a photograph must see. Below this it is looking past the
    #: subject, or at its silhouette edge-on.
    min_visible: int = 200
    #: Fraction of the frame the subject must fill, measured as occupied
    #: cells of the coverage grid. A photograph of a speck across the room
    #: costs a dense pair and contributes nothing.
    min_covered: float = 0.25
    #: The working range, millimetres. Nearer than this the subject is
    #: clipped; farther and it is a few pixels across.
    depth_mm: tuple[float, float] = (150.0, 500.0)
    #: The frame is divided into this many cells for `covered`. 32 x 18 keeps
    #: the aspect of a 16:9 sensor, so a cell is square-ish and "a quarter of
    #: the frame" means the same thing horizontally and vertically.
    grid: tuple[int, int] = (32, 18)
    #: Two photographs are the same photograph when they are BOTH within this
    #: angle of each other AND this close together. Either one alone is a new
    #: view worth having — the same rule `photos.CapturePolicy` applies live,
    #: applied again here because the live gate ran against what had been
    #: taken, not against what was finally selected.
    novelty_deg: float = 8.0
    novelty_mm: float = 20.0
    #: Photographs at most. Dense time is roughly linear in this and the
    #: atlas stops improving long before it.
    cap: int = 150
    #: View-direction buckets about the volume axis: 24 azimuth bins of 15
    #: degrees, three elevation bands of 60. Deliberately coarser than the
    #: novelty rule, so "covered" means "something looks from roughly there".
    buckets: tuple[int, int] = (24, 3)
    #: The clean-only atlas is used when there are at least this many clean
    #: photographs AND they cover at least this fraction of the buckets the
    #: selection covers. Below either, the whole set is textured from and the
    #: run is degraded — a complete atlas beats a stripe-free holey one.
    clean_min: int = 12
    clean_coverage: float = 0.80
    #: Seed points at most. Unbounded, a million voxels at tens of tracks
    #: each is a multi-gigabyte ASCII file `image_undistorter` has to parse
    #: and rewrite; capped, PatchMatch's depth ranges are unaffected, because
    #: they need coverage rather than density.
    seed_cap: int = 30_000


# ── the session, read back ───────────────────────────────────────────────


@dataclass(frozen=True, eq=False)
class PhotoMeta:
    """One `photos.jsonl` row, with the pose rebuilt as a matrix.

    `eq=False` because the poses are arrays and a generated `__eq__` would
    raise on them rather than answer.
    """

    #: This side's running number, and the name COLMAP knows the file by.
    n: int
    side: str
    #: The camserver id, e.g. `cam2` — not the COLMAP camera id, which is
    #: `CAMERA_ID[side]`.
    camera_id: str
    #: Session-relative, as written: `photos/left_0001.jpg`.
    file: str
    #: The basename, which is the `NAME` in `images.txt`, the stem of every
    #: mask and the line in `texture_images.txt`.
    name: str
    stripe: str
    wh: tuple[int, int]
    capture_mono: float
    pair_capture_mono: float
    #: Board→camera, this eye's own: the left photograph's is the smoothed
    #: pose, the right's is that pose composed across the pair.
    R: np.ndarray
    t_mm: np.ndarray
    pose_source: str
    pose_composed: bool
    pose_rms_px: float
    pose_gap_deg: float
    pose_gap_mm: float
    pose_corners: int
    pose_smooth_mm: float
    sharpness: float
    pass_id: int
    laser_on: bool
    stripe_pixels: int
    stripe_shifted: bool
    kept_points: int

    @property
    def colmap_camera_id(self) -> int:
        """1 for a left photograph, 2 for a right one."""
        return CAMERA_ID[self.side]

    @property
    def centre_mm(self) -> np.ndarray:
        """Where this camera stood, in the board frame."""
        return camera_centre(self.R, self.t_mm)

    @property
    def direction(self) -> np.ndarray:
        """Where it was looking, in the board frame."""
        return view_direction(self.R)


@dataclass
class SessionInfo:
    """`session.json`, as the reconstruction needs it: both eyes' intrinsics
    at the size they were solved at, the pair's geometry, the board and the
    scan volume."""

    session_id: str
    #: The session directory itself, so a caller holding only this can find
    #: `photos/`, `colmap/` and the rest.
    path: Path
    started_utc: str
    rig: RigSnapshot
    #: The `params` block verbatim — the thresholds that decided what was
    #: photographed, carried through rather than re-derived.
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def volume(self) -> ScanVolume:
        return self.rig.volume

    @property
    def board(self) -> BoardSnapshot | None:
        return self.rig.board

    @property
    def extrinsics(self) -> Extrinsics | None:
        return self.rig.extrinsics

    def eye(self, side: str) -> EyeSnapshot | None:
        """One eye's snapshot, or None when the session opened before that
        eye had intrinsics."""
        if side not in SIDES:
            raise ValueError(f"no such eye: {side!r}")
        return self.rig.left if side == "left" else self.rig.right

    def cameras(self, selection: Selection | None = None) -> list[EyeCamera]:
        """Both eyes as `colmapio.EyeCamera`, ready for `write_cameras`.

        When a `Selection` is given, each camera carries the distinct raw
        sizes of that side's selected photographs, which is what lets
        `write_cameras` refuse a photograph taken at a resolution its
        intrinsics were never solved at. Eyes the session has no intrinsics
        for are left out rather than written as zeros.
        """
        out: list[EyeCamera] = []
        for side in SIDES:
            eye = self.eye(side)
            if eye is None:
                continue
            wh: tuple[tuple[int, int], ...] = ()
            if selection is not None:
                wh = tuple(sorted({v.photo.wh for v in selection.accepted
                                   if v.photo.side == side}))
            k1, k2, p1, p2, k3 = (list(np.asarray(eye.dist, float).ravel())
                                  + [0.0] * 5)[:5]
            out.append(EyeCamera(
                camera_id=CAMERA_ID[side], side=side,
                width=int(eye.wh[0]), height=int(eye.wh[1]),
                fx=eye.fx, fy=eye.fy, cx=eye.cx, cy=eye.cy,
                dist=(k1, k2, p1, p2, k3), photo_wh=wh))
        return out


def load_session(session_dir: str | Path) -> tuple[SessionInfo, list[PhotoMeta]]:
    """Read a session back off disk: `session.json` and every `photos.jsonl`
    row, in the order they were written.

    The manifest is the authority on where a photograph was taken from, so the
    quaternion is turned back into a matrix here and nowhere else. Both files
    are refused by name when they are not what they claim to be — a schema we
    do not know, or a pose in a convention we do not speak, is a file that
    would reconstruct into confident nonsense.

    A session with no `photos.jsonl` yields an empty list rather than an
    error: nothing was photographed, which is a fact about that session and
    not a corrupt one.
    """
    path = Path(session_dir)
    raw = json.loads((path / "session.json").read_text(encoding="utf-8"))
    schema = raw.get("schema")
    if schema != SCHEMA:
        raise ValueError(
            f"{path / 'session.json'}: schema is {schema!r}, not {SCHEMA!r}")
    info = SessionInfo(
        session_id=str(raw.get("session_id", "")),
        path=path,
        started_utc=str(raw.get("started_utc", "")),
        rig=RigSnapshot.from_dict(raw),
        params=dict(raw.get("params") or {}),
    )

    manifest = path / "photos.jsonl"
    photos: list[PhotoMeta] = []
    if manifest.exists():
        for line in manifest.read_bytes().decode("utf-8").splitlines():
            if line.strip():
                photos.append(_photo_from_row(json.loads(line), manifest))
    return info, photos


def _number(value: Any) -> float:
    """A manifest number, or NaN for the `null` the writer puts where a
    measurement was not finite — a one-eyed pose has no gap between the eyes,
    a disarmed eye has no sharpness — because JSON has no NaN of its own."""
    return float("nan") if value is None else float(value)


def _photo_from_row(row: dict[str, Any], manifest: Path) -> PhotoMeta:
    pose = row.get("pose") or {}
    if pose.get("convention") != POSE_CONVENTION:
        raise ValueError(
            f"{manifest}: photograph {row.get('file')!r} carries a "
            f"{pose.get('convention')!r} pose, not {POSE_CONVENTION!r}")
    wh = row.get("wh") or (0, 0)
    file = str(row.get("file", ""))
    return PhotoMeta(
        n=int(row.get("n", 0)),
        side=str(row.get("side", "")),
        camera_id=str(row.get("camera_id", "")),
        file=file,
        name=Path(file).name,
        stripe=str(row.get("stripe", "")),
        wh=(int(wh[0]), int(wh[1])),
        capture_mono=float(row.get("capture_mono", 0.0)),
        pair_capture_mono=float(row.get("pair_capture_mono", 0.0)),
        R=Rotation.from_quat(np.asarray(pose["q_wxyz"], float),
                             scalar_first=True).as_matrix(),
        t_mm=np.asarray(pose["t_mm"], float).ravel(),
        pose_source=str(row.get("pose_source", "")),
        pose_composed=bool(row.get("pose_composed", False)),
        pose_rms_px=_number(row.get("pose_rms_px")),
        pose_gap_deg=_number(row.get("pose_gap_deg")),
        pose_gap_mm=_number(row.get("pose_gap_mm")),
        pose_corners=int(row.get("pose_corners", 0)),
        pose_smooth_mm=_number(row.get("pose_smooth_mm")),
        sharpness=_number(row.get("sharpness")),
        pass_id=int(row.get("pass_id", 0)),
        laser_on=bool(row.get("laser_on", True)),
        stripe_pixels=int(row.get("stripe_pixels", 0)),
        stripe_shifted=bool(row.get("stripe_shifted", False)),
        kept_points=int(row.get("kept_points", 0)),
    )


# ── what one photograph sees ─────────────────────────────────────────────


def _project(eye: EyeSnapshot, p_cam: np.ndarray) -> np.ndarray:
    """(M, 2) pixels of (M, 3) points already expressed in the camera's frame.

    The five-coefficient OpenCV model the intrinsics were solved with, written
    out rather than routed through `cv2.projectPoints` — the same trade
    `stereo.StereoRig.project_right` makes, and for the same reason: this runs
    over the whole cloud once per photograph, and the loop-free form is about
    five times cheaper.
    """
    z = p_cam[:, 2]
    x = p_cam[:, 0] / z
    y = p_cam[:, 1] / z
    k1, k2, p1, p2, k3 = (list(np.asarray(eye.dist, float).ravel()) + [0.0] * 5)[:5]
    r2 = x * x + y * y
    radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
    xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    return np.stack([eye.fx * xd + eye.cx, eye.fy * yd + eye.cy], axis=1)


def _seen(eye: EyeSnapshot, R: np.ndarray, t_mm: np.ndarray,
          xyz: np.ndarray, normals: np.ndarray
          ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Which of `xyz` this photograph sees, where they land and how far away.

    Returns the indices into `xyz`, their (M, 2) pixels and their (M,) depths
    in millimetres. Three tests, in the order that makes the expensive one
    smallest: in front of the camera, facing it, then inside the image.

    The facing test is `n · (p − C) < 0` in the board frame, which is the same
    number as `n_cam · p_cam` and cheaper to reach — the pose is a rotation, so
    the dot product is unchanged by it. It is the whole of this module's
    occlusion handling, and §2.6 says plainly what that buys and what it does
    not.
    """
    xyz = np.asarray(xyz, float).reshape(-1, 3)
    normals = np.asarray(normals, float).reshape(-1, 3)
    if len(normals) != len(xyz):
        raise ValueError(
            f"{len(xyz)} points but {len(normals)} normals — the front-facing "
            "test needs one normal per point")
    empty_i, empty_f = np.empty(0, np.int64), np.empty((0, 2))
    if not len(xyz):
        return empty_i, empty_f, np.empty(0)

    R = np.asarray(R, float).reshape(3, 3)
    p_cam = xyz @ R.T + np.asarray(t_mm, float).reshape(1, 3)
    ahead = p_cam[:, 2] > _MIN_DEPTH_MM
    facing = np.einsum("ij,ij->i", normals, xyz - camera_centre(R, t_mm)) < 0.0
    idx = np.flatnonzero(ahead & facing)
    if not len(idx):
        return empty_i, empty_f, np.empty(0)

    uv = _project(eye, p_cam[idx])
    w, h = int(eye.wh[0]), int(eye.wh[1])
    inside = ((uv[:, 0] >= 0.0) & (uv[:, 0] < w)
              & (uv[:, 1] >= 0.0) & (uv[:, 1] < h))
    idx = idx[inside]
    return idx, uv[inside], p_cam[idx, 2]


def _covered(uv: np.ndarray, wh: tuple[int, int],
             grid: tuple[int, int]) -> float:
    """The fraction of the coverage grid's cells that hold at least one
    projected point. Occupancy, not density: a photograph filled by one dense
    corner of the subject is not a photograph of the subject."""
    gw, gh = int(grid[0]), int(grid[1])
    if not len(uv):
        return 0.0
    cx = np.clip((uv[:, 0] / max(wh[0], 1) * gw).astype(np.int64), 0, gw - 1)
    cy = np.clip((uv[:, 1] / max(wh[1], 1) * gh).astype(np.int64), 0, gh - 1)
    return float(len(np.unique(cy * gw + cx))) / float(gw * gh)


# ── the selection ────────────────────────────────────────────────────────


@dataclass
class PhotoView:
    """One photograph, measured against the cloud — and either accepted or
    dropped with a reason."""

    photo: PhotoMeta
    #: Seed points this photograph sees, by the test in `_seen`.
    n_visible: int = 0
    #: Occupied cells of the coverage grid, as a fraction of all of them.
    covered: float = 0.0
    #: The median depth of what it sees, millimetres. NaN when it sees
    #: nothing.
    median_depth_mm: float = float("nan")
    #: `sharpness × covered` — the greedy order. Sharpness alone would
    #: prefer a crisp photograph of the table's edge.
    score: float = 0.0
    #: Assigned in `images.txt` order once the greedy pass is done; 0 on a
    #: photograph that was dropped.
    image_id: int = 0
    #: Why this photograph is not in the model. None on an accepted one, and
    #: never None on a dropped one — a photograph that vanished without a
    #: sentence is the failure mode this field exists to prevent.
    dropped_because: str | None = None

    @property
    def accepted(self) -> bool:
        return self.dropped_because is None


@dataclass
class Selection:
    """The photographs the model is built from, and every photograph that
    was not, with its reason."""

    session: SessionInfo
    params: SelectParams
    #: Every photograph offered, measured, in manifest order.
    views: list[PhotoView] = field(default_factory=list)
    #: The accepted ones in `images.txt` order — sorted by side and running
    #: number, which is also the order their image ids were assigned in.
    accepted: list[PhotoView] = field(default_factory=list)

    @property
    def dropped(self) -> list[PhotoView]:
        return [v for v in self.views if not v.accepted]

    @property
    def clean(self) -> list[PhotoView]:
        """The accepted photographs taken with the laser switched off."""
        return [v for v in self.accepted if not v.photo.laser_on]

    @property
    def names(self) -> list[str]:
        """The accepted photographs' file names, as `images.txt` writes
        them."""
        return [v.photo.name for v in self.accepted]

    @property
    def counts(self) -> dict[str, int]:
        """`session.json`'s `selected` block."""
        out = {side: sum(1 for v in self.accepted if v.photo.side == side)
               for side in SIDES}
        out["clean"] = len(self.clean)
        out["dropped"] = len(self.dropped)
        return out

    def by_image_id(self) -> dict[int, PhotoView]:
        return {v.image_id: v for v in self.accepted}


def select(session: SessionInfo, photos: list[PhotoMeta],
           cloud_xyz: np.ndarray, cloud_normals: np.ndarray,
           params: SelectParams | None = None) -> Selection:
    """Choose the photographs the reconstruction is built from.

    Three gates and then a greedy pass. The gates ask whether a photograph is
    of the subject at all: it must see at least `min_visible` seed points,
    they must occupy at least `min_covered` of the frame, and their median
    depth must sit inside the working range. Pose quality is not re-checked —
    the still gate, the corner floor and the eye-agreement bound were applied
    at capture, and a photograph that failed them was never written.

    The greedy pass then takes the best photographs that are not each other.
    Order is clean photographs first — so that where a clean and a laser
    photograph stand in the same place the clean one is the one that survives
    — and within that by `sharpness × covered` descending. A candidate is
    accepted unless some already-accepted photograph is BOTH within
    `novelty_deg` of its viewing direction AND within `novelty_mm` of its
    camera centre; either difference alone is a view worth having. The cap
    bounds dense time, which is roughly linear in the count.

    `cloud_normals` are the PCA normals of `cloud_xyz`, oriented outward
    (`scan.orient_normals`) — the front-facing test is meaningless without the
    orientation, because PCA gives a normal's line and not its direction.
    """
    params = params or SelectParams()
    lo, hi = params.depth_mm
    gw, gh = params.grid
    views = [PhotoView(photo=p) for p in photos]

    for v in views:
        eye = session.eye(v.photo.side)
        if eye is None:
            v.dropped_because = (
                f"the session has no intrinsics for the {v.photo.side} eye")
            continue
        idx, uv, depth = _seen(eye, v.photo.R, v.photo.t_mm,
                               cloud_xyz, cloud_normals)
        v.n_visible = int(len(idx))
        # Against the eye's own size, which is the frame `_seen` clipped to.
        v.covered = _covered(uv, (int(eye.wh[0]), int(eye.wh[1])), params.grid)
        v.median_depth_mm = float(np.median(depth)) if len(depth) else float("nan")
        v.score = float(v.photo.sharpness) * v.covered

        if v.n_visible < params.min_visible:
            v.dropped_because = (
                f"sees {v.n_visible} seed points, fewer than {params.min_visible}")
        elif v.covered < params.min_covered:
            v.dropped_because = (
                f"the subject fills {v.covered:.0%} of the frame "
                f"({int(round(v.covered * gw * gh))} of {gw * gh} cells), "
                f"under {params.min_covered:.0%}")
        elif not (lo <= v.median_depth_mm <= hi):
            v.dropped_because = (
                f"median depth {v.median_depth_mm:.0f} mm is outside "
                f"{lo:.0f}-{hi:.0f} mm")

    # Clean first, then the best score. A photograph whose eye never measured
    # sharpness carries NaN (`photos.EyePhoto.sharpness`), and NaN compares
    # false against everything — so it is pushed to the back deliberately
    # rather than landing wherever the sort happens to leave it.
    candidates = sorted(
        (v for v in views if v.dropped_because is None),   # survived the gates
        key=lambda v: (v.photo.laser_on,
                       -(v.score if np.isfinite(v.score) else -np.inf),
                       v.photo.side, v.photo.n))

    accepted: list[PhotoView] = []
    for v in candidates:
        if len(accepted) >= params.cap:
            v.dropped_because = (
                f"the selection was full at {params.cap} photographs")
            continue
        twin = _twin(v, accepted, params)
        if twin is not None:
            other, deg, mm = twin
            v.dropped_because = (
                f"the same view as {other.photo.name}: {deg:.1f} deg and "
                f"{mm:.1f} mm apart, under {params.novelty_deg:.0f} deg and "
                f"{params.novelty_mm:.0f} mm")
            continue
        accepted.append(v)

    accepted.sort(key=lambda v: (v.photo.side, v.photo.n))
    for image_id, v in enumerate(accepted, start=1):
        v.image_id = image_id
    return Selection(session=session, params=params, views=views,
                     accepted=accepted)


def _twin(cand: PhotoView, accepted: list[PhotoView],
          params: SelectParams) -> tuple[PhotoView, float, float] | None:
    """The first accepted photograph that is the same view as `cand`, with
    how far apart the two stand. None when `cand` is new."""
    d = cand.photo.direction
    c = cand.photo.centre_mm
    for other in accepted:
        cos = float(np.clip(np.dot(d, other.photo.direction), -1.0, 1.0))
        deg = float(np.degrees(np.arccos(cos)))
        mm = float(np.linalg.norm(c - other.photo.centre_mm))
        if deg < params.novelty_deg and mm < params.novelty_mm:
            return other, deg, mm
    return None


# ── view buckets, and the texture set they gate ──────────────────────────


def buckets(session: SessionInfo,
            selection: Selection) -> dict[tuple[int, int], list[int]]:
    """The selected photographs, binned by viewing direction about the volume
    axis: `{(azimuth bin, elevation band): [image_id, ...]}`.

    It takes the `Selection` and not the raw photograph list on purpose.
    Coverage here is measured **relative to what was selected — that is, to
    what was captured — not to the object**. A direction nobody photographed
    is simply not a bucket; counting it as uncovered would make the clean gate
    fail for a reason the clean pass could never fix.

    Azimuth is the direction's angle about the axis, in `params.buckets[0]`
    equal bins over the full turn; elevation is its angle out of the board
    plane, in `params.buckets[1]` equal bands over [-90, +90] degrees. The
    axis is the `ScanVolume`'s, which in the board frame is +z: the cylinder
    stands on the board (`scan.py:86-111`). That is why `session` is in the
    signature — it is where the volume lives — and it is checked against the
    selection's own, because buckets measured in one session's frame say
    nothing about another's.

    The bins are deliberately coarser than the 8 deg / 20 mm novelty rule:
    "covered" is meant to mean "something looks from roughly there", not
    "exactly one thing does".
    """
    if session.session_id != selection.session.session_id:
        raise ValueError(
            f"the selection came from session "
            f"{selection.session.session_id!r}, not {session.session_id!r}")
    n_az, n_el = selection.params.buckets
    out: dict[tuple[int, int], list[int]] = {}
    for v in selection.accepted:
        out.setdefault(_bucket_of(v.photo.direction, n_az, n_el),
                       []).append(v.image_id)
    return out


def _bucket_of(direction: np.ndarray, n_az: int, n_el: int) -> tuple[int, int]:
    """One viewing direction's bucket. A direction exactly along the axis has
    no azimuth; it lands in bin 0, which is what `arctan2(0, 0)` gives and is
    as good an answer as any."""
    d = np.asarray(direction, float).ravel()
    d = d / max(float(np.linalg.norm(d)), 1e-12)
    az = float(np.degrees(np.arctan2(d[1], d[0]))) % 360.0
    el = float(np.degrees(np.arcsin(np.clip(d[2], -1.0, 1.0))))
    ia = min(int(az // (360.0 / n_az)), n_az - 1)
    ie = min(int((el + 90.0) // (180.0 / n_el)), n_el - 1)
    return ia, ie


def texture_set(session: SessionInfo, selection: Selection,
                mode: str) -> tuple[list[str], str, str | None]:
    """Which photographs `mesh_texturer`'s workspace is undistorted from, what
    to call that set, and what to warn about — `(names, source, warning)`.

    The clean photographs are used alone when there are at least `clean_min`
    of them **and** they cover at least `clean_coverage` of the buckets the
    selection covers — again, of the selection, not of the object. Below
    either gate the whole selection is textured from instead, and the run is
    degraded rather than failed: a complete atlas with stripes in it is more
    use than a stripe-free one full of holes.

    It is all-or-nothing by design. Admitting the laser photographs whose
    buckets no clean photograph covers would fill exactly those holes, but
    `mesh_texturer` has no per-image preference, so an admitted laser
    photograph can win vertices a clean one also sees and leak stripes
    somewhere nobody predicted. The remedy for a degraded run — a top-up clean
    pass and another texture-only run — costs minutes.

    `mode` is what makes the label honest. The same fallback set is `"raw"` in
    texture-only, where the workspace is undistorted from the verbatim
    `colmap/images/`, and `"inpainted"` in dense, where it comes from
    `colmap/images_clean/`. The function cannot know which without being told.
    """
    if mode not in MODES:
        raise ValueError(f"no such mode: {mode!r} — one of {MODES}")

    covered = buckets(session, selection)
    clean_ids = {v.image_id for v in selection.clean}
    clean_covered = {key for key, ids in covered.items()
                     if clean_ids.intersection(ids)}
    n_clean = len(selection.clean)
    coverage = (len(clean_covered) / len(covered)) if covered else 0.0

    if n_clean >= selection.params.clean_min \
            and coverage >= selection.params.clean_coverage:
        return [v.photo.name for v in selection.clean], "clean", None

    source = FALLBACK_SOURCE[mode]
    missing = set(covered) - clean_covered
    warning = (
        f"only {n_clean} clean photos and {coverage:.0%} bucket coverage "
        f"({len(clean_covered)} of {len(covered)} buckets"
        + (f"; missing {_name_buckets(missing, selection.params.buckets)}"
           if missing else "")
        + f") — texturing from the full set, texture_source={source}")
    return selection.names, source, warning


def _name_buckets(missing: set[tuple[int, int]],
                  grid: tuple[int, int]) -> str:
    """The uncovered buckets as something an operator can act on — `cameras
    looking toward az 120-180 at all elevations` rather than a list of index
    pairs.

    **The arcs are directions cameras look along, not places they stand.** A
    bucket is `PhotoMeta.direction`'s bin, so a photograph taken from az 0 is
    in the az-180 bucket and an operator filling a named hole walks to the
    opposite azimuth — and, for a named elevation band, to the opposite side of
    the board plane. Naming the sense is the whole point of the phrase: read as
    a standing place, every one of these hints sends the camera exactly the
    wrong way round the turntable.

    Runs of neighbouring azimuth bins that miss the same elevation bands are
    collapsed, because that is how a coverage hole actually looks: one arc of
    the turntable nobody walked round with the laser off.
    """
    n_az, n_el = int(grid[0]), int(grid[1])
    az_span, el_span = 360.0 / n_az, 180.0 / n_el
    by_az: dict[int, frozenset[int]] = {}
    for ia in range(n_az):
        els = frozenset(ie for az, ie in missing if az == ia)
        if els:
            by_az[ia] = els

    parts: list[str] = []
    run_start: int | None = None
    for ia in range(n_az + 1):
        els = by_az.get(ia)
        if run_start is not None and els != by_az.get(run_start):
            lo = run_start * az_span
            hi = ia * az_span
            bands = by_az[run_start]
            where = "all elevations" if len(bands) == n_el else "el " + ", ".join(
                f"{-90.0 + ie * el_span:g}..{-90.0 + (ie + 1) * el_span:g}"
                for ie in sorted(bands))
            parts.append(f"az {lo:g}-{hi:g} at {where}")
            run_start = None
        if els is not None and run_start is None:
            run_start = ia
    return "cameras looking toward " + "; ".join(parts)


def image_list_text(names: list[str]) -> str:
    """`texture_images.txt`: one image name per line, exactly as `images.txt`
    writes them, with a trailing newline.

    `--image_list_path` and the cfg rewrite both match by string, so there is
    no normalisation here and there must not be one anywhere else.
    """
    return "".join(f"{name}\n" for name in names)


def write_image_list(path: str | Path, names: list[str]) -> None:
    """`image_list_text` on disk, with LF endings on every host — COLMAP takes
    either, but a file that changes shape with the machine that wrote it
    cannot be compared against a golden one."""
    with Path(path).open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(image_list_text(names))


# ── the seed cloud, and the tracks into it ───────────────────────────────


def seed_points(cloud_xyz: np.ndarray, cloud_rgb: np.ndarray | None,
                cloud_normals: np.ndarray | None, cap: int = 30_000
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A uniform voxel subsample of the cloud, at most `cap` points, as
    `(xyz, rgb, normals)`.

    **The method is a voxel grid, and the representative is the point nearest
    its voxel's centre.** A voxel size is found by bisection — doubled until
    the occupied voxels fit under the cap, then halved back as far as it can
    go while still fitting — so the sample is as large as the cap allows and
    is spread through the cloud's volume rather than through its point
    ordering. Taking the nearest point to each voxel centre, rather than the
    first one that landed there, makes the result independent of the order the
    scan happened to produce.

    Uniformity is the point, not thrift. `patch_match_stereo` reads each
    image's depth range from the points that image observes, so the seeds must
    reach every part of the object; a prefix, a stride or a random draw
    weighted by density would leave the sparsely-scanned regions with no
    range at all.

    Colours come through when the cloud has them and are mid-grey when it does
    not; normals come through when they are given and are zero when they are
    not, so a caller that needs them can tell the difference.
    """
    xyz = np.asarray(cloud_xyz, float).reshape(-1, 3)
    rgb = (np.full((len(xyz), 3), 128, np.uint8) if cloud_rgb is None
           else np.asarray(cloud_rgb, np.uint8).reshape(-1, 3))
    normals = (np.zeros((len(xyz), 3)) if cloud_normals is None
               else np.asarray(cloud_normals, float).reshape(-1, 3))
    cap = int(cap)
    if len(xyz) <= cap or cap <= 0:
        return xyz, rgb, normals

    lo = xyz.min(axis=0)
    extent = np.maximum(xyz.max(axis=0) - lo, 1e-9)
    # A first guess from the bounding box: if the points filled it evenly,
    # this size would give about `cap` occupied voxels. Real clouds are
    # surfaces, so it always undershoots — hence the doubling.
    size = float(np.cbrt(float(np.prod(extent)) / cap))
    for _ in range(64):
        if _n_voxels(xyz, lo, size) <= cap:
            break
        size *= 2.0
    else:
        raise ValueError("the cloud does not fit any voxel size — is it finite?")

    small, big = size / 2.0, size
    for _ in range(24):                 # ~1e-7 of the starting size
        mid = 0.5 * (small + big)
        if _n_voxels(xyz, lo, mid) <= cap:
            big = mid
        else:
            small = mid

    keys, offset = _voxel_keys(xyz, lo, big)
    # Sort by voxel, then by distance to that voxel's centre, and take the
    # first of each run: one point per voxel, the most central one.
    d2 = np.einsum("ij,ij->i", offset - 0.5, offset - 0.5)
    order = np.lexsort((d2, keys))
    first = np.ones(len(order), bool)
    first[1:] = keys[order][1:] != keys[order][:-1]
    keep = np.sort(order[first])
    return xyz[keep], rgb[keep], normals[keep]


def _voxel_keys(xyz: np.ndarray, lo: np.ndarray,
                size: float) -> tuple[np.ndarray, np.ndarray]:
    """Each point's voxel as one int64 key, and where it sits inside that
    voxel as a fraction of the voxel's side.

    The key is packed rather than compared row by row: `np.unique` over an
    (N, 3) array sorts rows and costs several times what one int64 sort does,
    and this runs once per bisection step.
    """
    grid = (xyz - lo) / size
    cell = np.floor(grid).astype(np.int64)
    frac = grid - cell
    dims = cell.max(axis=0) - cell.min(axis=0) + 1
    if float(dims[0]) * float(dims[1]) * float(dims[2]) >= 2.0 ** 62:
        # Too fine to pack — and far too fine to fit any cap, which is the
        # answer the caller is bisecting towards anyway, so every point is
        # given a voxel of its own.
        return np.arange(len(xyz), dtype=np.int64), frac
    cell -= cell.min(axis=0)
    keys = (cell[:, 0] * dims[1] + cell[:, 1]) * dims[2] + cell[:, 2]
    return keys, frac


def _n_voxels(xyz: np.ndarray, lo: np.ndarray, size: float) -> int:
    return int(len(np.unique(_voxel_keys(xyz, lo, size)[0])))


def tracks(selection: Selection, seeds: np.ndarray, seed_normals: np.ndarray,
           seed_rgb: np.ndarray | None = None
           ) -> tuple[list[Point3D], dict[int, list[tuple[float, float, int]]]]:
    """The seed points with their tracks, and every image's POINTS2D — built
    in one pass over one list of observations.

    Returns `(points3d, points2d_per_image)`, keyed by image id. Every
    `(IMAGE_ID, POINT2D_IDX)` in a track indexes the entry that names the
    point back, because the two are appended at the same instant: the
    observation is written into the image's list and its position in that list
    is written into the track, in the same statement. That is what makes a
    dangling observation impossible. Generating POINTS2D first and filtering
    the tracks afterwards — or the other way round — is how every converter
    that ends up writing `POINT3D_ID = -1` gets there.

    Only points observed by two or more images are written. A single
    observation constrains nothing: COLMAP's depth ranges come from the points
    an image shares with others, and `__auto__` resolves source images by
    exactly those shared observations.

    Visibility is `_seen`'s — in front, facing, inside the image — and it runs
    over both eyes, each through its own intrinsics. Colours default to
    mid-grey, which is what `points3D.txt` carries for an uncoloured cloud.
    """
    xyz = np.asarray(seeds, float).reshape(-1, 3)
    rgb = (np.full((len(xyz), 3), 128, np.uint8) if seed_rgb is None
           else np.asarray(seed_rgb, np.uint8).reshape(-1, 3))

    # observations[j] is every image that sees seed j, with where it landed.
    observations: list[list[tuple[int, float, float]]] = [[] for _ in range(len(xyz))]
    points2d: dict[int, list[tuple[float, float, int]]] = {}
    for view in selection.accepted:
        points2d[view.image_id] = []
        eye = selection.session.eye(view.photo.side)
        if eye is None:                 # `select` cannot accept such a photo
            continue
        idx, uv, _ = _seen(eye, view.photo.R, view.photo.t_mm, xyz, seed_normals)
        for j, (u, v) in zip(idx.tolist(), uv.tolist()):
            observations[j].append((view.image_id, u, v))

    points3d: list[Point3D] = []
    for j, obs in enumerate(observations):
        if len(obs) < 2:
            continue
        point_id = len(points3d) + 1
        track: list[tuple[int, int]] = []
        for image_id, u, v in obs:
            seen_by = points2d[image_id]
            track.append((image_id, len(seen_by)))
            seen_by.append((u, v, point_id))
        points3d.append(Point3D(point_id=point_id, xyz=xyz[j],
                                rgb=tuple(int(c) for c in rgb[j]),
                                error=1.0, track=track))
    return points3d, points2d


# ── the cfg the undistorter wrote, rewritten ─────────────────────────────


#: The source line `image_undistorter` writes under every image it
#: registered: `__auto__` plus `--num_patch_match_src_images`, whose default
#: is 20. The rewrite never writes it — an image we have no geometry for keeps
#: its own line, whatever number that line carries — it is here because it is
#: the shape of the file, and what a caller comparing or logging one reads it
#: as.
AUTO_SOURCES = "__auto__, 20"

#: A cfg line that is blank or starts with this is not a name and not a
#: source list: COLMAP's own reader drops both before it pairs the rest. We
#: pair around them and pass them through untouched.
CFG_COMMENT = "#"


@dataclass(frozen=True)
class CfgParams:
    """The angle buckets a reference image draws its sources from, and the
    two numbers the depth-range fallback is decided by.

    One dataclass for the reason `SelectParams` is one: a threshold cannot
    then be passed to one place and not another, and `session.json` carries
    the lot verbatim.
    """

    #: `(low, high, count)` per bucket, degrees, half-open `[low, high)` so a
    #: candidate at exactly 15 degrees belongs to one bucket and not to two.
    #: Sources are bucketed rather than sorted by proximity because
    #: nearest-first composed with the 8 degree / 20 mm novelty gate produces
    #: the *minimum* baseline every time — 20 mm at 300 mm is 3.8 degrees,
    #: which gives a depth sigma around a millimetre, worse than the laser
    #: cloud the dense pass exists to improve on.
    buckets: tuple[tuple[float, float, int], ...] = (
        (8.0, 15.0, 2), (15.0, 30.0, 3), (30.0, 45.0, 3))
    #: No source closer in viewing direction than this, ever — including the
    #: ones drawn in to fill a short bucket. It is the floor the buckets
    #: start at, written down separately because the fill has to honour it
    #: too, and it is what replaced the old `min_baseline_mm` knob.
    min_angle_deg: float = 8.0
    #: Seed points an image must observe for its own track-derived depth
    #: range to mean anything.
    min_observations: int = 50
    #: The fraction of the selection that may be thinner than that before a
    #: global range is passed instead. Strictly more than this, so one thin
    #: image in ten is not an emergency.
    thin_frac: float = 0.10
    #: How far past the volume's own extent the fallback range is opened, as
    #: a fraction of that extent. The poses are good to a millimetre and the
    #: cylinder is already generous, but a range that clips the subject costs
    #: the surface it clipped.
    pad: float = 0.20

    @property
    def n_sources(self) -> int:
        """Sources per reference image — the buckets' counts, added up. There
        is no second knob for the total: it is what the buckets ask for."""
        return sum(int(count) for _, _, count in self.buckets)


@dataclass(frozen=True)
class CfgReport:
    """The rewritten cfg, and what became of every name involved."""

    #: The whole file, ready to write over `dense/stereo/patch-match.cfg`.
    text: str
    #: Images whose source line was replaced with a bucketed list.
    rewritten: tuple[str, ...]
    #: Registered images whose source line was left exactly as it was —
    #: either the selection has no geometry for them, or it has no candidate
    #: far enough away to offer. `__auto__` is a worse answer than ours and a
    #: far better one than an empty line, which aborts the stage.
    kept_auto: tuple[str, ...]
    #: Selected images the undistorter never registered. They are dropped,
    #: not added: naming a frame COLMAP did not register aborts
    #: `patch_match_stereo` before it computes anything.
    missing: tuple[str, ...]


@dataclass(frozen=True)
class DepthRange:
    """The global depth range to force on PatchMatch when the tracks are too
    thin to give it a per-image one, and the sentence that says why."""

    depth_min_mm: float
    depth_max_mm: float
    #: What the operator is told, naming the seed cap first — thin tracks
    #: usually mean the cap is too low for this object rather than that the
    #: range needs forcing.
    warning: str


def plan_patch_match_cfg(cfg_text: str, selection: Selection,
                         params: CfgParams = CfgParams()) -> CfgReport:
    """Rewrite the source lines of the cfg `image_undistorter` wrote, and say
    what became of every name in it.

    **The cfg comes from the undistorter, not from us.** Its image lines are
    the authoritative list of frames COLMAP actually registered, and a frame
    that failed to register, named here, aborts the whole
    `patch_match_stereo` stage before it computes a single depth map. So
    every image line is kept verbatim and in order, only source lines are
    touched, and a selected photograph the undistorter did not register is
    *dropped* — named in the report rather than written into the file.

    An image the selection has no geometry for keeps its own source line,
    which is `__auto__, 20`: we cannot rank sources for a photograph whose
    pose we do not hold, and COLMAP's covisibility fallback is exactly what
    that line asks for.

    **Sources are bucketed by angle, not sorted by proximity.** Each
    reference image takes `count` sources from each `(low, high, count)`
    bucket of `params.buckets`, nearest first within a bucket; a bucket that
    cannot be filled leaves its places to the nearest remaining candidates at
    least `params.min_angle_deg` away, whatever bucket those came from.
    Nearest-first alone would hand PatchMatch the smallest baseline available
    every time, which is the one geometry that cannot triangulate.
    """
    lines = cfg_text.splitlines()
    pairs = _cfg_pairs(lines)
    registered = [lines[name_at].strip() for name_at, _ in pairs]

    by_name = {v.photo.name: v for v in selection.accepted}
    dirs = {name: by_name[name].photo.direction
            for name in registered if name in by_name}

    replaced: dict[int, str] = {}
    rewritten: list[str] = []
    kept_auto: list[str] = []
    for name_at, source_at in pairs:
        name = lines[name_at].strip()
        sources = (_sources_for(name, dirs, params)
                   if source_at is not None and name in dirs else [])
        (rewritten if sources else kept_auto).append(name)
        if sources:
            replaced[source_at] = ", ".join(sources)

    out = [replaced.get(i, line) for i, line in enumerate(lines)]
    known = set(registered)
    return CfgReport(
        text="".join(f"{line}\n" for line in out),
        rewritten=tuple(rewritten),
        kept_auto=tuple(kept_auto),
        missing=tuple(v.photo.name for v in selection.accepted
                      if v.photo.name not in known),
    )


def _cfg_pairs(lines: list[str]) -> list[tuple[int, int | None]]:
    """`(image line, source line)` index pairs, skipping the blank and
    comment lines COLMAP's own reader skips.

    The last pair's source index is None when the file ends on a name, which
    is the one way this format can be broken without looking broken. Such a
    name keeps the nothing it has: this rewrites source lines, it does not
    invent them, and a truncated cfg should fail COLMAP's own reader rather
    than be quietly patched into something that runs.
    """
    pairs: list[tuple[int, int | None]] = []
    name_at: int | None = None
    for i, line in enumerate(lines):
        text = line.strip()
        if not text or text.startswith(CFG_COMMENT):
            continue
        if name_at is None:
            name_at = i
        else:
            pairs.append((name_at, i))
            name_at = None
    if name_at is not None:
        pairs.append((name_at, None))
    return pairs


def _sources_for(name: str, dirs: dict[str, np.ndarray],
                 params: CfgParams) -> list[str]:
    """This reference image's sources: the buckets in order, then whatever
    places the short buckets left, given to the nearest remaining candidates.

    Empty when nothing is far enough away to be a source at all, which is the
    caller's signal to leave that image's own `__auto__` line alone.
    """
    d = dirs[name]
    ranked = sorted((_view_angle_deg(d, other_d), other)
                    for other, other_d in dirs.items() if other != name)

    picked: list[str] = []
    taken: set[str] = set()
    for lo, hi, count in params.buckets:
        wanted = int(count)
        for deg, other in ranked:
            if wanted <= 0:
                break
            if other in taken or not (lo <= deg < hi):
                continue
            picked.append(other)
            taken.add(other)
            wanted -= 1
    for deg, other in ranked:
        if len(picked) >= params.n_sources:
            break
        if other in taken or deg < params.min_angle_deg:
            continue
        picked.append(other)
        taken.add(other)
    return picked


def _view_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """The angle between two viewing directions, degrees — the same number
    the novelty rule compares against 8 degrees, and the axis the buckets are
    laid out along."""
    cos = float(np.clip(np.dot(np.asarray(a, float).ravel(),
                               np.asarray(b, float).ravel()), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


def depth_range_fallback(selection: Selection, counts: dict[int, int],
                         volume: ScanVolume,
                         params: CfgParams = CfgParams()) -> DepthRange | None:
    """The global `(depth_min, depth_max)` to force on PatchMatch, or None to
    leave the ranges to the tracks — which is the default and the better
    answer.

    COLMAP derives each image's depth range from the 3D points that image
    observes, one range per image, and no single global pair can beat that.
    So this returns None unless the tracks are demonstrably too thin to do
    the job: more than `params.thin_frac` of the selected images observing
    fewer than `params.min_observations` seed points. An image the counts do
    not mention observes none, which is as thin as it gets.

    When it does fire, the range is the scan volume seen from the selection:
    for every accepted photograph, how far along its optical axis the
    cylinder starts and ends; the nearest start and the farthest end of all
    of them, opened by `params.pad` at both ends. That is a bound rather than
    a measurement, which is why the warning names the seed cap first.
    """
    accepted = selection.accepted
    if not accepted:
        return None
    thin = sum(1 for v in accepted
               if counts.get(v.image_id, 0) < params.min_observations)
    frac = thin / len(accepted)
    if frac <= params.thin_frac:
        return None

    near, far = float("inf"), float("-inf")
    for v in accepted:
        lo, hi = _volume_depths(volume, v.photo.centre_mm, v.photo.direction)
        near, far = min(near, lo), max(far, hi)
    span = far - near
    # A depth range has to start in front of the camera: a photograph taken
    # from inside the volume would otherwise ask PatchMatch for a negative
    # near plane.
    depth_min = max(near - params.pad * span, _MIN_DEPTH_MM)
    depth_max = far + params.pad * span

    warning = (
        f"{frac:.0%} of the selected images ({thin} of {len(accepted)}) "
        f"observe fewer than {params.min_observations} seed points — raising "
        f"the {selection.params.seed_cap:,}-point seed cap is the first thing "
        f"to try; passing an explicit depth range "
        f"{depth_min:.0f}-{depth_max:.0f} mm is the fallback, and it is "
        f"unverified that it overrides COLMAP's own per-image ranges")
    return DepthRange(depth_min_mm=depth_min, depth_max_mm=depth_max,
                      warning=warning)


def _volume_depths(volume: ScanVolume, centre_mm: np.ndarray,
                   direction: np.ndarray) -> tuple[float, float]:
    """How far along one camera's optical axis the scan volume starts and
    ends, millimetres.

    Depth along the axis is linear in the point, so its extremes over the
    cylinder sit on the rims: the farthest point is whichever end cap the
    axis tilts towards, displaced by the full radius across the axis, and the
    nearest is the other cap displaced the other way. Written out rather than
    sampled, because a sample of a cylinder is a bound that is quietly too
    tight.
    """
    d = np.asarray(direction, float).ravel()
    d = d / max(float(np.linalg.norm(d)), 1e-12)
    # How much of the disc's radius lies across the axis, and where the two
    # end caps land along it.
    lateral = float(volume.radius_mm) * float(np.hypot(d[0], d[1]))
    caps = (d[2] * float(volume.floor_mm), d[2] * float(volume.height_mm))
    base = -float(d @ np.asarray(centre_mm, float).ravel())
    return base + min(caps) - lateral, base + max(caps) + lateral
