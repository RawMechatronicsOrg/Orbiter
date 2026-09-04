"""Rolling shutter: the sensor's readout time, measured from the board in
motion, and the per-row correction it makes possible.

**The effect.** These cameras read their sensor row by row: row `y` of a frame
is exposed `y · T / H` after row 0, with `T` the readout time of the whole
frame. Everything the scan measures on a frame — the stripe's pixels, the
board's corners — is therefore a set of observations at slightly different
instants, and while the board and the subject move relative to the camera
those instants matter. A point at the bottom of the frame is transformed into
the board's frame with a pose measured at the corners' rows, a readout apart,
and lands where the board WAS. At a hand turning the subject at 20-50 mm/s and
a readout of ~20 ms that is 0.4-1 mm, the same order as the scan's noise; with
the rig in the hand it dominates.

**Why correct points, not pixels.** The laser plane is rigid to the camera, so
a stripe pixel's 3-D position in the camera frame is exact for its own
instant; only the camera→board transform is at the wrong time. And the board's
pose track is a metric 6-DoF motion estimate at 30 Hz: two consecutive poses
give a twist, the twist gives the pose at any instant in between, and each
point is taken into the board's frame through the pose at its own row's time
(`to_board`). A few hundred float operations per frame on a few thousand
points, no warping of 2M-pixel images, no resampling of a sub-pixel centroid.

**Measuring T.** From the board itself, moving (`solve_readout`): every corner
of every frame is an observation at its own row's time. With a pose per frame
at the corners' mean row, and each frame's velocity from its neighbours' poses
over camserver's capture clock, the corners of a moving board must fit the
pose slid to each corner's row — and the amount they must be slid by is T.
The poses and T are solved together, over all frames. What makes T observable
is motion the corners cannot be fitted without: a board translated sideways
during the readout is merely sheared, and a shear is nearly a homography — a
tilted board explains most of it. A board turned or tilted quickly bends its
own straight lines, and no pose explains that. So the instruction is to twist
and tilt the board, not slide it; the solve reports the image-space skew the
motion produced and refuses when there was too little to measure by.

`seconds` is signed: positive when row 0 is read first, negative for a sensor
mounted the other way up. It belongs to one sensor mode, so like the intrinsics
it carries the frame size it was measured at and is refused at another.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

from .cvcore import estimate_pose

#: The board's own-plane object points are re-expressed in `cvcore.estimate_pose`'s
#: frame — centred, z out of the printed face — so that poses solved here mean
#: the same thing as the ones the scan runs on.
_FACE_OUT = np.diag([1.0, -1.0, -1.0])

#: Refuse a solve whose motion moved the corners by less than this over one
#: readout: the estimate would be noise, however small its reported sigma.
MIN_SKEW_PX = 3.0
#: Longer than a frame at 20 fps, or negligible: not a readout time.
MAX_READOUT_S = 0.06
MIN_READOUT_S = 0.0005
#: Fewer frames than this and the velocities have nothing to average over.
MIN_VIEWS = 20
#: Frames further apart than this are not neighbours: a velocity taken
#: across such a gap is not the velocity inside either frame. The frames
#: come in bursts of brisk motion, and each burst is solved as its own run.
MAX_RUN_GAP_S = 0.08
#: A run shorter than this has no velocity to speak of.
MIN_RUN = 3


@dataclass(frozen=True)
class Readout:
    """One sensor mode's readout time, with what it was measured from."""

    seconds: float
    width: int
    height: int
    sigma_s: float = float("nan")
    #: Corner displacement over one readout in the frame that moved fastest —
    #: how much the solve had to measure by.
    skew_px: float = float("nan")
    rms_px: float = float("nan")
    views: int = 0

    def row_offset(self, rows) -> np.ndarray:
        """Seconds after the frame's capture instant at which `rows` were read."""
        return np.asarray(rows, np.float64) * (self.seconds / self.height)

    def as_config(self) -> dict[str, Any]:
        return {
            "seconds": float(self.seconds), "width": self.width, "height": self.height,
            "sigma_s": float(self.sigma_s), "skew_px": float(self.skew_px),
            "rms_px": float(self.rms_px), "views": int(self.views),
        }

    @staticmethod
    def from_config(d: Any, frame_wh: tuple[int, int] | None) -> "Readout | None":
        """The stored figure, usable at `frame_wh`, or None — a readout time
        belongs to a sensor mode, and camserver can be reconfigured."""
        if not isinstance(d, dict):
            return None
        try:
            r = Readout(
                seconds=float(d["seconds"]), width=int(d["width"]), height=int(d["height"]),
                sigma_s=float(d.get("sigma_s", float("nan"))),
                skew_px=float(d.get("skew_px", float("nan"))),
                rms_px=float(d.get("rms_px", float("nan"))), views=int(d.get("views", 0)),
            )
        except (KeyError, TypeError, ValueError):
            return None
        if frame_wh is not None and (int(frame_wh[0]), int(frame_wh[1])) != (r.width, r.height):
            return None
        if not np.isfinite(r.seconds) or r.height <= 0:
            return None
        return r


# ── motion between two poses ──────────────────────────────────────────────


def twist(R0: np.ndarray, t0: np.ndarray, time0: float,
          R1: np.ndarray, t1: np.ndarray, time1: float) -> tuple[np.ndarray, np.ndarray]:
    """The constant twist taking board→camera pose (R0, t0) at `time0` to
    (R1, t1) at `time1`: (omega, v) per second, in the camera frame. With it,
    `pose_at(R1, t1, omega, v, dt)` is the pose `dt` seconds after `time1`."""
    dt = float(time1 - time0)
    if dt <= 0:
        raise ValueError("the second pose must be later than the first")
    d_rot = Rotation.from_matrix(np.asarray(R1, float) @ np.asarray(R0, float).T)
    omega = d_rot.as_rotvec() / dt
    v = (np.asarray(t1, float).ravel() - np.asarray(t0, float).ravel()) / dt
    return omega, v


def poses_at(R: np.ndarray, t: np.ndarray, omega: np.ndarray, v: np.ndarray,
             dts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The pose slid along the twist by each of `dts` seconds: (M, 3, 3) and
    (M, 3). Rotation about the camera frame's axes, translation linear —
    exact for a constant twist over the few milliseconds a readout lasts."""
    dts = np.asarray(dts, np.float64).ravel()
    rots = Rotation.from_rotvec(np.outer(dts, omega)).as_matrix()      # (M, 3, 3)
    R_m = rots @ np.asarray(R, float)
    t_m = np.asarray(t, float).ravel()[None, :] + np.outer(dts, v)
    return R_m, t_m


def to_board(xyz_cam: np.ndarray, dts: np.ndarray, R: np.ndarray, t: np.ndarray,
             omega: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Camera-frame points into the board's frame, each through the pose at
    its own instant: `dts` seconds after the instant (R, t) holds for."""
    xyz_cam = np.asarray(xyz_cam, np.float64).reshape(-1, 3)
    if not len(xyz_cam):
        return np.empty((0, 3))
    R_m, t_m = poses_at(R, t, omega, v, dts)
    # X_board = R_m^T (X_cam - t_m), per point.
    return np.einsum("mji,mj->mi", R_m, xyz_cam - t_m)


@dataclass(frozen=True)
class Motion:
    """The board's motion around one frame, for the scan: the pose the
    frame's corners gave holds at the instant of `ref_row`; (omega, v)
    slide it to any other row's instant."""

    omega: np.ndarray
    v: np.ndarray
    readout: Readout
    ref_row: float

    @staticmethod
    def between(prev_R, prev_t, prev_capture: float, prev_row: float,
                R, t, capture: float, row: float, readout: Readout) -> "Motion | None":
        """From the previous frame's pose to this one's, each at its own
        corners' instant. None when the two are not in order."""
        t0 = prev_capture + float(readout.row_offset(prev_row))
        t1 = capture + float(readout.row_offset(row))
        if not t1 > t0:
            return None
        omega, v = twist(prev_R, prev_t, t0, R, t, t1)
        return Motion(omega=omega, v=v, readout=readout, ref_row=float(row))

    def to_board(self, xyz_cam: np.ndarray, rows: np.ndarray,
                 R: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Camera-frame points read at `rows` into the board's frame, each
        through the pose at its own row's instant."""
        dts = self.readout.row_offset(np.asarray(rows, np.float64) - self.ref_row)
        return to_board(xyz_cam, dts, R, t, self.omega, self.v)

    @property
    def speed_mm_s(self) -> float:
        return float(np.linalg.norm(self.v))

    @property
    def spin_deg_s(self) -> float:
        return float(np.degrees(np.linalg.norm(self.omega)))


# ── measuring the readout ─────────────────────────────────────────────────


@dataclass
class MotionView:
    """One frame's corners for the readout solve, in ORIGINAL pixels."""

    corners: np.ndarray            # (N, 1, 2) or (N, 2) float
    ids: np.ndarray                # (N, 1) or (N,) int
    capture_mono: float
    wh: tuple[int, int]


class MotionCollector:
    """Frames of the board in brisk motion, per eye, for the readout solve.
    Bounded: past a few hundred frames the solve has all the motion it can
    use, and the oldest go first."""

    MAX_FRAMES = 300

    def __init__(self) -> None:
        self._views: dict[str, list[MotionView]] = {"left": [], "right": []}

    def add(self, side: str, view: MotionView) -> None:
        views = self._views[side]
        views.append(view)
        if len(views) > self.MAX_FRAMES:
            del views[: len(views) - self.MAX_FRAMES]

    def views(self, side: str) -> list[MotionView]:
        return self._views[side]

    def count(self, side: str) -> int:
        return len(self._views[side])

    @property
    def total(self) -> int:
        return sum(len(v) for v in self._views.values())

    def clear(self) -> None:
        for v in self._views.values():
            v.clear()


@dataclass
class _Prepared:
    obj: np.ndarray                # (N, 3) board points, cvcore frame, mm
    img: np.ndarray                # (N, 2) observed pixels
    rows: np.ndarray               # (N,) their rows
    row_ref: float                 # mean row: the pose's reference row
    capture: float
    rvec: np.ndarray               # initial pose, cvcore frame
    tvec: np.ndarray


def _board_points(board) -> tuple[np.ndarray, np.ndarray]:
    """All chessboard corners in cvcore's centred face-out frame, mm, and
    their IDs (the index is the ChArUco corner id)."""
    pts = np.asarray(board.getChessboardCorners(), np.float64).reshape(-1, 3) * 1000.0
    sx, sy = board.getChessboardSize()
    sq = float(board.getSquareLength()) * 1000.0
    centre = np.array([sx * sq / 2.0, sy * sq / 2.0, 0.0])
    return (pts - centre) @ _FACE_OUT.T, np.arange(len(pts))


def _project(obj_cam: np.ndarray, k) -> np.ndarray:
    """Pinhole + OpenCV's Brown distortion, for (N, 3) camera-frame points."""
    x = obj_cam[:, 0] / obj_cam[:, 2]
    y = obj_cam[:, 1] / obj_cam[:, 2]
    d = np.zeros(5)
    dd = np.asarray(k.D, np.float64).ravel()
    d[: min(5, len(dd))] = dd[:5]
    k1, k2, p1, p2, k3 = d
    r2 = x * x + y * y
    radial = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))
    xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    K = np.asarray(k.K, np.float64)
    return np.column_stack([K[0, 0] * xd + K[0, 2], K[1, 1] * yd + K[1, 2]])


def _prepare(views: list[MotionView], board, k) -> list[_Prepared]:
    board_pts, _ = _board_points(board)
    out: list[_Prepared] = []
    R_prev = None
    for v in views:
        corners = np.asarray(v.corners, np.float32).reshape(-1, 1, 2)
        ids = np.asarray(v.ids, np.int32).reshape(-1, 1)
        if len(corners) < 6:
            continue
        pose = estimate_pose(corners, ids, board, k, R_prev)
        if pose is None:
            continue
        R, t, _ = pose
        R_prev = R
        rvec = Rotation.from_matrix(R).as_rotvec()
        img = corners.reshape(-1, 2).astype(np.float64)
        out.append(_Prepared(
            obj=board_pts[ids.ravel()], img=img, rows=img[:, 1].copy(),
            row_ref=float(img[:, 1].mean()), capture=float(v.capture_mono),
            rvec=rvec, tvec=np.asarray(t, float).ravel(),
        ))
    return out


class _Problem:
    """Residuals of every corner of every frame against the poses slid to the
    corners' rows, as a function of (poses..., T)."""

    def __init__(self, frames: list[_Prepared], k, height: int) -> None:
        self.frames = frames
        self.k = k
        self.height = float(height)
        self.n = len(frames)
        # Neighbours only within a run: across a gap the difference of two
        # poses says nothing about the motion inside either frame.
        caps = np.array([f.capture for f in frames])
        self.run = np.concatenate([[0], np.cumsum(np.diff(caps) > MAX_RUN_GAP_S)])
        self.counts = np.array([len(f.img) for f in frames])
        self.offsets = np.concatenate([[0], np.cumsum(self.counts)])

    def x0(self, readout_s: float) -> np.ndarray:
        return np.concatenate([np.concatenate([f.rvec, f.tvec]) for f in self.frames]
                              + [np.array([readout_s])])

    def unpack(self, x: np.ndarray):
        poses = x[:-1].reshape(self.n, 6)
        return poses[:, :3], poses[:, 3:], float(x[-1])

    def velocities(self, rvecs, tvecs, T):
        """Each frame's twist from its neighbours, over the reference instants."""
        rots = Rotation.from_rotvec(rvecs)
        times = np.array([f.capture + f.row_ref * T / self.height for f in self.frames])
        omega = np.zeros((self.n, 3))
        vel = np.zeros((self.n, 3))
        for i in range(self.n):
            a = i - 1 if i > 0 and self.run[i - 1] == self.run[i] else i
            b = i + 1 if i + 1 < self.n and self.run[i + 1] == self.run[i] else i
            dt = times[b] - times[a]
            if dt <= 0:
                continue
            omega[i] = (rots[b] * rots[a].inv()).as_rotvec() / dt
            vel[i] = (tvecs[b] - tvecs[a]) / dt
        return omega, vel

    def residuals(self, x: np.ndarray) -> np.ndarray:
        rvecs, tvecs, T = self.unpack(x)
        omega, vel = self.velocities(rvecs, tvecs, T)
        out = np.empty((self.offsets[-1], 2))
        for i, f in enumerate(self.frames):
            dts = (f.rows - f.row_ref) * (T / self.height)
            R_m, t_m = poses_at(Rotation.from_rotvec(rvecs[i]).as_matrix(), tvecs[i],
                                omega[i], vel[i], dts)
            cam = np.einsum("mij,mj->mi", R_m, f.obj) + t_m
            out[self.offsets[i]: self.offsets[i + 1]] = _project(cam, self.k) - f.img
        return out.ravel()

    def sparsity(self):
        """A frame's residuals depend on its own pose, its neighbours' (through
        the velocity) and T."""
        m = int(self.offsets[-1]) * 2
        n = 6 * self.n + 1
        s = lil_matrix((m, n), dtype=int)
        for i in range(self.n):
            r0, r1 = self.offsets[i] * 2, self.offsets[i + 1] * 2
            for j in (i - 1, i, i + 1):
                if 0 <= j < self.n:
                    s[r0:r1, 6 * j: 6 * j + 6] = 1
            s[r0:r1, n - 1] = 1
        return s

    def skew_px(self, x: np.ndarray) -> float:
        """The largest corner displacement over one readout among the frames:
        what the motion gave the solve to measure T by."""
        rvecs, tvecs, T = self.unpack(x)
        omega, vel = self.velocities(rvecs, tvecs, T)
        worst = 0.0
        for i, f in enumerate(self.frames):
            R = Rotation.from_rotvec(rvecs[i]).as_matrix()
            R_m, t_m = poses_at(R, tvecs[i], omega[i], vel[i], np.array([-T / 2, T / 2]))
            a = _project(f.obj @ R_m[0].T + t_m[0], self.k)
            b = _project(f.obj @ R_m[1].T + t_m[1], self.k)
            worst = max(worst, float(np.linalg.norm(b - a, axis=1).mean()))
        return worst


def _in_runs(frames: list[_Prepared]) -> list[_Prepared]:
    """Drop frames that have no neighbour within `MAX_RUN_GAP_S` on either
    side to make a run of at least `MIN_RUN`. Frames are sorted by time."""
    if not frames:
        return frames
    out: list[_Prepared] = []
    run: list[_Prepared] = [frames[0]]
    for prev, cur in zip(frames, frames[1:]):
        if cur.capture - prev.capture > MAX_RUN_GAP_S:
            if len(run) >= MIN_RUN:
                out.extend(run)
            run = []
        run.append(cur)
    if len(run) >= MIN_RUN:
        out.extend(run)
    return out


def solve_readout(views: list[MotionView], board, k,
                  initial_s: float = 0.02) -> tuple[Readout | None, str]:
    """T for one eye from its corners under motion. `(Readout, "")`, or
    `(None, why)` when the frames cannot support the solve."""
    if not views:
        return None, "no frames"
    wh = views[0].wh
    frames = _prepare(views, board, k)
    frames.sort(key=lambda f: f.capture)
    frames = _in_runs(frames)
    if len(frames) < MIN_VIEWS:
        return None, f"only {len(frames)} usable frames in bursts of motion, need {MIN_VIEWS}"
    problem = _Problem(frames, k, wh[1])
    x0 = problem.x0(initial_s)
    rms_before = float(np.sqrt(np.mean(problem.residuals(problem.x0(0.0)) ** 2)))
    try:
        fit = least_squares(problem.residuals, x0, jac_sparsity=problem.sparsity(),
                            x_scale="jac", method="trf", max_nfev=200)
    except (ValueError, np.linalg.LinAlgError) as exc:
        return None, f"solve failed: {exc}"
    T = float(fit.x[-1])
    rms = float(np.sqrt(np.mean(fit.fun ** 2)))
    skew = problem.skew_px(fit.x)
    if skew < MIN_SKEW_PX:
        return None, (f"too little motion: corners moved {skew:.1f} px over a readout, "
                      f"need {MIN_SKEW_PX:.0f} — twist and tilt the board faster")
    # Sigma of T from the fit's curvature, scaled by the residual variance.
    sigma = float("nan")
    try:
        jac = fit.jac.toarray() if hasattr(fit.jac, "toarray") else np.asarray(fit.jac)
        jtj = jac.T @ jac
        dof = max(jac.shape[0] - jac.shape[1], 1)
        cov = np.linalg.inv(jtj) * (float(np.sum(fit.fun ** 2)) / dof)
        sigma = float(np.sqrt(max(cov[-1, -1], 0.0)))
    except np.linalg.LinAlgError:
        pass
    if not (MIN_READOUT_S <= abs(T) <= MAX_READOUT_S):
        return None, f"implausible readout {T * 1000:.2f} ms (rms {rms:.2f} px, skew {skew:.1f} px)"
    return Readout(seconds=T, width=int(wh[0]), height=int(wh[1]), sigma_s=sigma,
                   skew_px=skew, rms_px=rms, views=len(frames)), f"rms {rms_before:.2f} → {rms:.2f} px"
