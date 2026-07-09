"""Geometric SAM2 prompt from calibrated rig poses.

Reads ``sfm_priors.json`` (schema ``orbiter.sfm_priors.v1`` — per-image
world->camera pose + shared intrinsics), projects the turntable working
volume (a cylinder around the rotation axis) into each frame, and derives:

  * a **box prompt** — bbox of the in-frame projected volume points;
  * **positive points** — the volume axis (object) + disc rim (turntable);
  * **negative points** — a ring at ~2R (guaranteed room) + frame corners;
  * a **silhouette mask** — convex hull of the projected volume, used as the
    geometric fallback and as a clamp against SAM2 grabbing the room.

When a frame has no usable geometry (missing from the priors, or the volume
projects entirely off-frame) a heuristic centre-box prompt is returned and
flagged, so the masking never hard-fails on an uncalibrated scan.

World frame: origin at the rig axes intersection, +Z up, units mm — the same
frame the priors' world->camera transforms expect.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from mask_pipeline import dilate_disk, erode_disk

# Rim discretization per level and levels along the cylinder height.
_RIM_POINTS = 24
_LEVELS = (0.0, 0.5, 1.0)
# Box padding as a fraction of min(W, H).
_BOX_PAD_FRAC = 0.02
# Negative ring radius multiplier (×R — "definitely the room"); the
# constructor's ``neg_radius_mult`` overrides it.
_NEG_RADIUS_MULT = 2.0
# Minimum in-frame volume points for a geometry prompt to be trusted.
_MIN_INFRAME_PTS = 8
# Probe grid density (N×N over the projected bbox); the constructor's
# ``grid_steps`` overrides it.
_GRID_STEPS = 8


@dataclass
class Prompt:
    """SAM2 prompt + geometric fallback for one frame."""

    kind: str                       # "geometry" | "heuristic"
    box_xyxy: np.ndarray            # (4,) float32
    pos_xy: np.ndarray              # (N,2) float32, label 1
    neg_xy: np.ndarray              # (M,2) float32, label 0
    silhouette: np.ndarray | None   # HxW uint8 0/255, None for heuristic
    extra_pos_xy: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 2), np.float32)
    )                               # assembly top-up points (turntable disc)
    grid_xy: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 2), np.float32)
    )                               # probe grid inside the silhouette
    # Deterministic geometry stamps — regions KNOWN to rotate with the table
    # (projected disc / nominal board), unioned into the final mask so SAM2
    # never gets a vote on them. None when the geometry isn't available.
    stamp: np.ndarray | None = None         # HxW uint8 0/255 (disc ∪ board)
    disc_mask: np.ndarray | None = None     # HxW bool — for recall metrics


class PriorsError(ValueError):
    """Bad/unsupported priors file."""


def _volume_points(
    center_xy: tuple[float, float], radius: float, base: float, height: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(all_pts, axis_pts, table_pts) in world mm.

    ``axis_pts`` sit LOW on the rotation axis — the object body is just above
    the table, while the volume height is a generous upper bound; a positive
    point near the volume top would land in the room behind the object and
    flip the whole prompt. ``table_pts`` lie ON the turntable disc (z = base):
    centre, mid-radius ring and the rim — the part of the assembly SAM2 tends
    to drop when prompted on the object alone.
    """
    cx, cy = center_xy
    axis = np.array(
        [[cx, cy, base + height * f] for f in (0.08, 0.2, 0.35)],
        dtype=np.float64,
    )
    table_rows = [[cx, cy, base]]
    for r_f in (0.45, 0.85):
        for k in range(0, _RIM_POINTS, 3):
            a = 2.0 * math.pi * k / _RIM_POINTS
            table_rows.append([
                cx + radius * r_f * math.cos(a),
                cy + radius * r_f * math.sin(a),
                base,
            ])
    table = np.asarray(table_rows, dtype=np.float64)

    # Hull/bbox support: full rim at several heights + top disc — NOT used as
    # positives, only to bound the silhouette.
    hull_rows = []
    for f in _LEVELS:
        z = base + height * f
        for k in range(_RIM_POINTS):
            a = 2.0 * math.pi * k / _RIM_POINTS
            hull_rows.append([cx + radius * math.cos(a), cy + radius * math.sin(a), z])
    for r_f in (0.33, 0.66):
        for k in range(0, _RIM_POINTS, 3):
            a = 2.0 * math.pi * k / _RIM_POINTS
            hull_rows.append([
                cx + radius * r_f * math.cos(a),
                cy + radius * r_f * math.sin(a),
                base + height,
            ])
    hull = np.asarray(hull_rows, dtype=np.float64)
    return np.vstack([axis, table, hull]), axis, table


def _negative_ring(
    center_xy: tuple[float, float], radius: float, base: float, height: float,
    mult: float = _NEG_RADIUS_MULT,
) -> np.ndarray:
    cx, cy = center_xy
    rows = []
    r = radius * mult
    for f in (0.0, 0.5, 1.0):
        z = base + height * f
        for k in range(0, _RIM_POINTS, 2):
            a = 2.0 * math.pi * k / _RIM_POINTS
            rows.append([cx + r * math.cos(a), cy + r * math.sin(a), z])
    return np.asarray(rows, dtype=np.float64)


class GeomPrompter:
    """Projects the turntable volume into frames listed in a priors file."""

    def __init__(
        self,
        poses_path: Path,
        *,
        volume_radius_mm: float,
        volume_height_mm: float,
        volume_base_mm: float,
        volume_center_xy_mm: tuple[float, float] | None = None,
        grid_steps: int = _GRID_STEPS,
        neg_radius_mult: float = _NEG_RADIUS_MULT,
        disc_radius_mm: float | None = None,
    ) -> None:
        self.grid_steps = max(1, int(grid_steps))
        try:
            priors = json.loads(poses_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PriorsError(f"cannot read priors {poses_path}: {exc}") from exc
        if priors.get("schema") != "orbiter.sfm_priors.v1":
            raise PriorsError(
                f"unsupported priors schema {priors.get('schema')!r} in {poses_path}"
            )

        # Calibrated rig geometry for the deterministic stamps (optional
        # block, written by sfm_export when the rig is calibrated).
        turntable = priors.get("turntable") or {}
        axis = turntable.get("axis_xy_mm") or [0.0, 0.0]
        if volume_center_xy_mm is None:
            # Default the volume axis to the calibrated turntable axis — the
            # disc centre — rather than a blind world origin.
            volume_center_xy_mm = (float(axis[0]), float(axis[1]))
        self.volume_center_xy_mm = volume_center_xy_mm
        self.disc_rim = (
            self._disc_rim_points(
                (float(axis[0]), float(axis[1])), float(disc_radius_mm),
                volume_base_mm,
            )
            if disc_radius_mm and disc_radius_mm > 0 else None
        )
        self.board_outline = self._board_outline_points(turntable.get("board"))

        intr = priors.get("camera_intrinsics") or {}
        try:
            fx, fy = float(intr["fx"]), float(intr["fy"])
            cx, cy = float(intr["cx"]), float(intr["cy"])
        except (KeyError, TypeError, ValueError) as exc:
            raise PriorsError(f"bad camera_intrinsics in {poses_path}: {exc}") from exc
        self.K = np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64,
        )
        self.dist = np.array(
            [float(intr.get(k, 0.0)) for k in ("k1", "k2", "p1", "p2")],
            dtype=np.float64,
        )

        self.poses: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for img in priors.get("images", []):
            try:
                name = str(img["file"]).replace("\\", "/")
                q = np.array(
                    [float(img["qw"]), float(img["qx"]),
                     float(img["qy"]), float(img["qz"])], dtype=np.float64,
                )
                t = np.array(
                    [float(img["tx"]), float(img["ty"]), float(img["tz"])],
                    dtype=np.float64,
                )
            except (KeyError, TypeError, ValueError):
                continue
            n = float(np.linalg.norm(q))
            if n == 0.0:
                continue
            self.poses[name] = (q / n, t)

        self.vol_all, self.vol_axis, self.vol_table = _volume_points(
            volume_center_xy_mm, volume_radius_mm, volume_base_mm, volume_height_mm,
        )
        self.vol_neg = _negative_ring(
            volume_center_xy_mm, volume_radius_mm, volume_base_mm,
            volume_height_mm, neg_radius_mult,
        )

    # Physical disc slab thickness (mm) — the rim is stamped at the top
    # surface AND this far below it, so the disc's side band (rotating MDF
    # edge, useful texture) is covered too.
    _DISC_THICKNESS_MM = 15.0

    @classmethod
    def _disc_rim_points(
        cls, center_xy: tuple[float, float], radius: float, base: float,
    ) -> np.ndarray:
        """World points on the physical disc rim — top surface (z = base)
        plus a ring one slab-thickness below. The disc is centred on the
        turntable axis by construction."""
        cx, cy = center_xy
        n = 2 * _RIM_POINTS
        return np.asarray([
            [cx + radius * math.cos(2.0 * math.pi * k / n),
             cy + radius * math.sin(2.0 * math.pi * k / n),
             z]
            for z in (base, base - cls._DISC_THICKNESS_MM)
            for k in range(n)
        ], dtype=np.float64)

    @staticmethod
    def _board_outline_points(board: dict | None) -> np.ndarray | None:
        """World points along the board's perimeter, from its NOMINAL
        calibrated pose (``calib_board_world``) — per-frame detection is
        deliberately not used: during an object scan the board is often
        occluded by the subject.

        Edges are sampled (not just corners) so the filled hull follows lens
        distortion. Returns None when the priors carry no board pose."""
        if not board:
            return None
        try:
            R = cv2.Rodrigues(np.asarray(board["rvec"], np.float64))[0]
            t = np.asarray(board["t"], np.float64)
            w = float(board["width_mm"])
            h = float(board["height_mm"])
        except (KeyError, TypeError, ValueError, cv2.error):
            return None
        per_edge = 6
        pts: list[list[float]] = []
        for i in range(per_edge):
            f = i / per_edge
            pts += [[w * f, 0.0, 0.0], [w, h * f, 0.0],
                    [w * (1.0 - f), h, 0.0], [0.0, h * (1.0 - f), 0.0]]
        local = np.asarray(pts, dtype=np.float64)
        return local @ R.T + t

    def _project_front(
        self, pts_world: np.ndarray, q: np.ndarray, t: np.ndarray,
    ) -> np.ndarray:
        """Project world points keeping every in-front point, INCLUDING ones
        outside the frame — a stamp polygon must not shrink when the disc or
        board crosses the frame edge (cv2.fillConvexPoly clips for us)."""
        R = self._quat_to_R(q)
        cam = pts_world @ R.T + t
        in_front = cam[:, 2] > 1.0
        if not np.any(in_front):
            return np.zeros((0, 2), np.float64)
        rvec, _ = cv2.Rodrigues(R)
        uv, _ = cv2.projectPoints(
            pts_world[in_front].reshape(-1, 1, 3), rvec, t.reshape(3, 1),
            self.K, self.dist,
        )
        return uv.reshape(-1, 2)

    def _stamp_masks(
        self, q: np.ndarray, t: np.ndarray, w: int, h: int,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """(stamp, disc_mask) for one frame — filled hulls of the projected
        disc rim and nominal board outline. Coordinates are clamped to a
        sane range so distortion blow-ups far outside the frame can't wrap
        int32."""
        lim_lo, lim_hi = -2.0 * max(w, h), 3.0 * max(w, h)

        def _fill(pts_world: np.ndarray | None) -> np.ndarray | None:
            if pts_world is None or pts_world.shape[0] < 3:
                return None
            uv = self._project_front(pts_world, q, t)
            if uv.shape[0] < 3:
                return None
            uv = np.clip(uv, lim_lo, lim_hi)
            m = np.zeros((h, w), np.uint8)
            cv2.fillConvexPoly(m, cv2.convexHull(uv.astype(np.int32)), 255)
            return m

        disc = _fill(self.disc_rim)
        board = _fill(self.board_outline)
        if disc is None and board is None:
            return None, None
        stamp = np.zeros((h, w), np.uint8)
        if disc is not None:
            stamp |= disc
        if board is not None:
            stamp |= board
        return stamp, (disc > 0 if disc is not None else None)

    @staticmethod
    def _quat_to_R(q_wxyz: np.ndarray) -> np.ndarray:
        w, x, y, z = q_wxyz
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ], dtype=np.float64)

    def _project(
        self, pts_world: np.ndarray, q: np.ndarray, t: np.ndarray,
        w: int, h: int,
    ) -> np.ndarray:
        """Project world points; return (N,2) of in-frame, in-front points."""
        R = self._quat_to_R(q)
        cam = pts_world @ R.T + t
        in_front = cam[:, 2] > 1.0
        if not np.any(in_front):
            return np.zeros((0, 2), np.float64)
        rvec, _ = cv2.Rodrigues(R)
        uv, _ = cv2.projectPoints(
            pts_world[in_front].reshape(-1, 1, 3), rvec, t.reshape(3, 1),
            self.K, self.dist,
        )
        uv = uv.reshape(-1, 2)
        ok = (
            (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        )
        return uv[ok]

    @staticmethod
    def heuristic(w: int, h: int) -> Prompt:
        """Centre-box prompt for frames without usable geometry (§5.4)."""
        bw, bh = 0.6 * w, 0.6 * h
        box = np.array(
            [(w - bw) / 2, (h - bh) / 2, (w + bw) / 2, (h + bh) / 2],
            dtype=np.float32,
        )
        return Prompt(
            kind="heuristic",
            box_xyxy=box,
            pos_xy=np.array([[w / 2, h / 2]], dtype=np.float32),
            neg_xy=np.array(
                [[w * 0.02, h * 0.02], [w * 0.98, h * 0.02],
                 [w * 0.02, h * 0.98], [w * 0.98, h * 0.98]], dtype=np.float32,
            ),
            silhouette=None,
        )

    def prompt_for(self, rel_name: str, w: int, h: int) -> Prompt:
        """Prompt for one frame (``rel_name`` relative to ``--images``, posix)."""
        key = f"photos/{rel_name}".replace("\\", "/")
        pose = self.poses.get(key) or self.poses.get(rel_name.replace("\\", "/"))
        if pose is None:
            return self.heuristic(w, h)
        q, t = pose

        uv_all = self._project(self.vol_all, q, t, w, h)
        if uv_all.shape[0] < _MIN_INFRAME_PTS:
            return self.heuristic(w, h)
        uv_axis = self._project(self.vol_axis, q, t, w, h)
        uv_table = self._project(self.vol_table, q, t, w, h)
        uv_neg = self._project(self.vol_neg, q, t, w, h)

        # Deterministic stamps (projected disc + nominal board). Their
        # in-frame points also join the hull support so the silhouette (and
        # the clamp derived from it) always covers the stamped regions —
        # board corners can stick out past the volume cylinder.
        stamp, disc_mask = self._stamp_masks(q, t, w, h)
        for extra in (self.disc_rim, self.board_outline):
            if extra is not None:
                uv_extra = self._project(extra, q, t, w, h)
                if uv_extra.shape[0]:
                    uv_all = np.vstack([uv_all, uv_extra])

        pad = _BOX_PAD_FRAC * min(w, h)
        box = np.array([
            max(0.0, uv_all[:, 0].min() - pad),
            max(0.0, uv_all[:, 1].min() - pad),
            min(w - 1.0, uv_all[:, 0].max() + pad),
            min(h - 1.0, uv_all[:, 1].max() + pad),
        ], dtype=np.float32)

        # Positives: low axis points (the object body) + the disc centre.
        # Table points go into ``extra_pos_xy`` for the dedicated table query
        # — mixing them into the box query makes SAM2 merge object and table
        # into one sloppy segment instead of two crisp ones.
        pos = np.vstack([uv_axis, uv_table[:1]]) if uv_table.size else uv_axis
        table_extra = (
            uv_table[:: max(1, uv_table.shape[0] // 6)][:6]
            if uv_table.size else np.zeros((0, 2))
        )
        if pos.shape[0] == 0:
            pos = uv_all[:1]

        # Silhouette = filled convex hull of every in-frame volume point.
        sil = np.zeros((h, w), np.uint8)
        hull = cv2.convexHull(uv_all.astype(np.int32))
        cv2.fillConvexPoly(sil, hull, 255)

        # Negatives: the 2R ring + frame corners, kept when they fall OUTSIDE
        # the (slightly grown) silhouette. The box is useless as the room
        # criterion — a close-up volume's bbox can span the whole frame, which
        # would drop every negative and let SAM2 segment the entire scene.
        margin = max(1, int(0.02 * min(w, h)))
        sil_grown = dilate_disk(sil, margin)

        def _is_room(pt: np.ndarray) -> bool:
            xi, yi = int(round(pt[0])), int(round(pt[1]))
            return (0 <= xi < w and 0 <= yi < h
                    and sil_grown[yi, xi] == 0)

        negs = [p for p in uv_neg if _is_room(p)]
        for corner in ((w * 0.02, h * 0.02), (w * 0.98, h * 0.02),
                       (w * 0.02, h * 0.98), (w * 0.98, h * 0.98)):
            c = np.asarray(corner, dtype=np.float64)
            if _is_room(c):
                negs.append(c)
        neg = (np.asarray(negs, dtype=np.float32) if negs
               else np.zeros((0, 2), np.float32))
        # Cap the prompt size — dozens of points only slow the decoder down.
        if neg.shape[0] > 12:
            neg = neg[:: neg.shape[0] // 12][:12]

        # Probe grid: regular samples inside the (eroded) silhouette. Each one
        # becomes an independent single-point SAM2 query; masks mostly inside
        # the silhouette are unioned into the assembly. Erosion keeps probes
        # off the hull edge, where they'd land on the room half the time.
        erode = max(1, int(0.04 * min(w, h)))
        sil_core = erode_disk(sil, erode)
        gx0, gy0, gx1, gy1 = (int(box[0]), int(box[1]),
                              int(box[2]), int(box[3]))
        grid: list[list[float]] = []
        steps = self.grid_steps
        for iy in range(steps):
            for ix in range(steps):
                px = gx0 + (gx1 - gx0) * (ix + 0.5) / steps
                py = gy0 + (gy1 - gy0) * (iy + 0.5) / steps
                if sil_core[int(py), int(px)]:
                    grid.append([px, py])

        return Prompt(
            kind="geometry",
            box_xyxy=box,
            pos_xy=pos.astype(np.float32),
            neg_xy=neg,
            silhouette=sil,
            extra_pos_xy=np.asarray(table_extra, dtype=np.float32).reshape(-1, 2),
            grid_xy=np.asarray(grid, dtype=np.float32).reshape(-1, 2),
            stamp=stamp,
            disc_mask=disc_mask,
        )
