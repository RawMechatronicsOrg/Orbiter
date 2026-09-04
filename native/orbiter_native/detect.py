"""ChArUco detection per frame, with the corners tracked between detections.

NOT reimplemented here. `orbiter_server.calibration` already owns detection,
including the flat-board pose ambiguity work, and those functions turned out
to be pure — they take a board and intrinsics as arguments and never touch the
global model — so they are imported and called directly. Duplicating that
numerics would be the one genuinely expensive mistake available in this
module: a second copy would drift from the one the calibration actually runs,
and the drift would be invisible until a solve came out wrong.

**Why track at all.** A full ChArUco pass is the most expensive thing this app
does per frame: measured at 1080p on this machine, 29 ms with the board in view
and 18-22 ms with no board at all, single-threaded — against a 33 ms frame
interval, twice over for two eyes. At 30 fps the board barely moves between
frames, so its corners can be followed instead of found: pyramidal KLT from
the previous frame, on a crop around the last corners, then `cornerSubPix` on
the new frame so every corner is re-localised on the actual saddle and
nothing accumulates. That costs ~0.7 ms. A full pass still runs every
`redetect_every` frames to refresh the IDs and pick up corners that came into
view, so the average is a few ms rather than thirty.

**What keeps a tracked corner honest.** Two things. The board is flat, so one
homography from its own plane must explain every corner in the image: each
tracked frame fits that homography with RANSAC and drops the corners that miss
it — a hand across the board takes the corners under it out rather than
dragging them along. But a checkerboard shifted by a whole square lands its
saddles on saddles, and the homography then fits perfectly with every ID
wrong; on a uniform frame KLT even reports corners that never moved. Only the
markers can tell, so the track reads a few of them through the homography it
implies, and they must decode to the IDs the board says are there. Measured
on a synthetic frame: 32 of 32 markers decode in place, none after a shift of
one square in any direction, at 0.04 ms per marker. Tracked corners against a
fresh detection of the same frame: 0.15 px median, 0.66 px worst, which is the
disagreement between two sub-pixel refiners, not drift.

The laser stripe lives in `laser.py`, not here: it needs colour rather than
luminance and is fitted against the board region, so it consumes this module's
output instead of sitting beside it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from .cvcore import build_board, charuco_detect, estimate_pose

#: Termination for KLT and the sub-pixel refinement alike.
_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
#: Pixels per cell of the canonical patch a marker is read from, and the share
#: of a marker's correction capacity a read may use — OpenCV's own default.
_CELL_PX = 8
_CORRECTION_RATE = 0.6


@dataclass(frozen=True)
class TrackParams:
    """Tuning for following the corners between full detections."""

    #: Frames between full detections while a track holds. Tracking costs about
    #: a fortieth of a detection, so this sets the average cost per frame, and
    #: also how long a corner that came into view waits to be picked up.
    #: 1 (or 0) makes every frame a full pass.
    redetect_every: int = 10
    #: KLT window and pyramid depth. The reach is about win/2 * 2**levels px
    #: of motion between frames — 80 px here, well beyond a hand at 30 fps.
    klt_win: int = 21
    klt_levels: int = 3
    #: The crop KLT runs on: the last corners' bounding box plus this. It bounds
    #: what the tracker can follow; motion past it fails the track and the
    #: frame falls through to a full detection. Measured at 1080p: 0.66 ms on
    #: the crop against 3.2 ms on the whole frame, same corners to the pixel.
    roi_margin_px: int = 64
    #: Half-window for `cornerSubPix` on the new frame, which is what makes a
    #: tracked corner as good as a detected one: the saddle is re-found from
    #: the pixels, KLT only says where to look.
    subpix_half: int = 5
    #: RANSAC tolerance for the board-plane homography every tracked corner must
    #: fit. A corner on the wrong saddle misses by a square — tens of pixels.
    homography_tol_px: float = 2.0
    #: Markers read through the tracked homography, the ones nearest the
    #: tracked corners. None may decode to a different ID than the board puts
    #: there, and at least one must decode at all — a hand can hide one, a
    #: shift of a whole square hides them all or shows the wrong ones.
    verify_markers: int = 3
    #: Fewer surviving corners than this ends the track: `solvePnP` wants six
    #: anyway, and a homography over four cannot reject anything.
    min_corners: int = 6
    #: Losing more than this share of the last detection's corners brings the
    #: next full pass forward to the next frame.
    min_keep_frac: float = 0.6


@dataclass
class BoardHit:
    """ChArUco detection on one frame, in ORIGINAL (pre-orientation) pixels."""

    corners: np.ndarray | None = None       # (N, 1, 2) float32
    ids: np.ndarray | None = None           # (N, 1) int32
    #: Board→camera pose, only when per-eye intrinsics were supplied. Board
    #: frame: origin at the board's centre, z out of the printed face toward
    #: the camera — `cvcore.estimate_pose` explains why it is not OpenCV's. The
    #: model's intrinsics belong to the phone, not to this pair, so this stays
    #: None until the pair itself is calibrated — see `cvcore.intrinsics_from_eye`.
    R: np.ndarray | None = None
    t: np.ndarray | None = None
    #: Geodesic angle between the two planar-PnP candidates. Large means the
    #: view was genuinely ambiguous and the temporal prior did the deciding.
    ambiguity_deg: float = 0.0
    #: True when the corners were followed from the previous frame rather than
    #: found by a full ChArUco pass. Same IDs, same sub-pixel refinement.
    tracked: bool = False
    ms: float = 0.0

    @property
    def count(self) -> int:
        return 0 if self.corners is None else int(len(self.corners))

    def coverage(self, w: int, h: int) -> float:
        """Fraction of the frame area spanned by the detected corners' bbox.

        A calibration sweep wants corners spread across the frame, not clustered
        in the middle; this is the cheap proxy the overlay shows for that.
        """
        if self.corners is None or self.count < 4:
            return 0.0
        pts = self.corners.reshape(-1, 2)
        span = pts.max(axis=0) - pts.min(axis=0)
        return float((span[0] * span[1]) / max(w * h, 1))


class BoardDetector:
    """Holds the built `cv2.aruco.CharucoBoard` and the corner track.

    Rebuilding the board object every frame would be pure waste; the spec only
    changes when the operator edits the board params in the web UI, which
    `set_spec` handles. One instance per detector thread: the track is the
    previous frame's corners and pixels, and belongs to one eye.
    """

    def __init__(self, spec=None, track: TrackParams = TrackParams()) -> None:
        self._spec = None
        self._board = None
        self._track = track
        #: Last accepted rotation, used as the prior that breaks the flat-board
        #: pose ambiguity on the next frame.
        self._last_R = None
        # The track: where the corners were in the frame they were last seen.
        self._prev_gray: np.ndarray | None = None
        self._corners: np.ndarray | None = None
        self._ids: np.ndarray | None = None
        self._plane: np.ndarray | None = None      # (N, 2) board-plane coords
        self._since_detect = 0
        self._n_detected = 0
        # The board's markers, for reading through the track's homography:
        # each one's four plane corners, its ID, and the canonical patch.
        self._marker_obj: np.ndarray | None = None  # (M, 4, 2)
        self._marker_ids: np.ndarray | None = None  # (M,)
        self._dictionary = None
        self._cells = 0
        self._patch_dst: np.ndarray | None = None
        self.set_spec(spec)

    def set_spec(self, spec) -> None:
        """Swap the board spec. No-op when unchanged, so it is safe to call on
        every config poll."""
        if spec is None:
            self._spec, self._board = None, None
            self._forget()
            return
        if spec == self._spec:
            return
        self._spec = spec
        self._board = build_board(spec)
        self._last_R = None
        self._forget()
        board = self._board
        self._marker_obj = np.asarray(board.getObjPoints(), np.float64)[:, :, :2]
        self._marker_ids = np.asarray(board.getIds()).ravel()
        self._dictionary = board.getDictionary()
        self._cells = int(self._dictionary.markerSize) + 2     # bits plus the border
        edge = float(self._cells * _CELL_PX)
        self._patch_dst = np.float32([[0, 0], [edge, 0], [edge, edge], [0, edge]])

    @property
    def ready(self) -> bool:
        return self._board is not None

    @property
    def board(self):
        return self._board

    def detect(self, gray: np.ndarray, intrinsics=None) -> BoardHit:
        """Corners on a GRAYSCALE frame, tracked or detected. Pose only when
        intrinsics are given."""
        if self._board is None:
            return BoardHit()
        t0 = time.perf_counter()
        p = self._track
        have_track = self._corners is not None
        # Counted so that full passes land every `redetect_every` frames.
        due = self._since_detect + 1 >= p.redetect_every

        corners = ids = None
        tracked = False
        if have_track and not due:
            corners, ids = self._follow(gray)
            tracked = corners is not None
        if corners is None:
            corners, ids = charuco_detect(gray, self._board)
            if corners is None and have_track and due:
                # A refresh that found nothing — blur, a hand over the markers.
                # The corners may still be there to follow; the homography
                # decides, not the marker decoder.
                corners, ids = self._follow(gray)
                tracked = corners is not None
            # A full pass ran; it starts the count to the next one whether it
            # found the board or not. Retrying every frame would cost exactly
            # what tracking saves, at the moment the frames are hardest.
            self._since_detect = 0
            if not tracked:
                self._n_detected = 0 if corners is None else len(corners)
        else:
            self._since_detect += 1
        if tracked and len(corners) < p.min_keep_frac * self._n_detected:
            self._since_detect = p.redetect_every     # next frame: full pass

        hit = BoardHit(corners=corners, ids=ids, tracked=tracked)
        if corners is not None and intrinsics is not None:
            pose = estimate_pose(corners, ids, self._board, intrinsics,
                                 self._last_R)
            if pose is not None:
                hit.R, hit.t, hit.ambiguity_deg = pose
                self._last_R = hit.R
        elif corners is None:
            # The board left the view; the old rotation is no longer a prior
            # for whatever comes back, and reusing it could lock in the wrong
            # twin for the rest of the session.
            self._last_R = None
        self._remember(gray, corners, ids)
        hit.ms = (time.perf_counter() - t0) * 1000.0
        return hit

    # ── the track ─────────────────────────────────────────────────────────

    def forget(self) -> None:
        """Drop the track and the pose prior.

        A track kept while the detector is not being called would thaw
        against a frame from whenever it last ran — minutes, in scan mode,
        where the right eye skips the board entirely.
        """
        self._forget()
        self._last_R = None

    def _forget(self) -> None:
        self._prev_gray = self._corners = self._ids = self._plane = None
        self._since_detect = 0
        self._n_detected = 0

    def _remember(self, gray: np.ndarray, corners, ids) -> None:
        """Keep this frame and its corners as the start of the next follow."""
        if corners is None or ids is None or len(corners) < self._track.min_corners:
            self._forget()
            return
        obj, _ = self._board.matchImagePoints(corners, ids)
        if obj is None or len(obj) != len(corners):
            self._forget()
            return
        self._prev_gray = gray
        self._corners = np.ascontiguousarray(corners, dtype=np.float32).reshape(-1, 1, 2)
        self._ids = np.asarray(ids, dtype=np.int32).reshape(-1, 1)
        self._plane = np.ascontiguousarray(obj.reshape(-1, 3)[:, :2], dtype=np.float64)

    def _follow(self, gray: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
        """The previous corners found again in `gray`, or (None, None)."""
        p = self._track
        prev, pts = self._prev_gray, self._corners
        if prev is None or prev.shape != gray.shape:
            return None, None                    # the stream was reconfigured
        h, w = gray.shape
        xy = pts.reshape(-1, 2)
        lo = np.maximum(np.floor(xy.min(axis=0)).astype(int) - p.roi_margin_px, 0)
        hi = np.minimum(np.ceil(xy.max(axis=0)).astype(int) + p.roi_margin_px + 1, (w, h))
        if (hi - lo).min() <= p.klt_win:
            return None, None
        off = lo.astype(np.float32)
        win = (p.klt_win, p.klt_win)
        nxt, status, _err = cv2.calcOpticalFlowPyrLK(
            prev[lo[1]:hi[1], lo[0]:hi[0]], gray[lo[1]:hi[1], lo[0]:hi[0]],
            pts - off, None, winSize=win, maxLevel=p.klt_levels, criteria=_CRITERIA)
        if nxt is None:
            return None, None
        nxt = nxt + off
        keep = status.ravel() > 0
        # The refinement window must fit inside the image.
        m = p.subpix_half + 2
        nx, ny = nxt[:, 0, 0], nxt[:, 0, 1]
        keep &= (nx >= m) & (nx < w - m) & (ny >= m) & (ny < h - m)
        if keep.sum() < p.min_corners:
            return None, None
        nxt = np.ascontiguousarray(nxt[keep])
        ids, plane = self._ids[keep], self._plane[keep]
        cv2.cornerSubPix(gray, nxt, (p.subpix_half, p.subpix_half), (-1, -1), _CRITERIA)
        # The board is flat: one homography from its plane explains every
        # corner. Whatever it does not explain is on the wrong saddle, or
        # under a hand, and goes.
        hom, inl = cv2.findHomography(plane, nxt.reshape(-1, 2).astype(np.float64),
                                      cv2.RANSAC, p.homography_tol_px)
        if hom is None or inl is None:
            return None, None
        inl = inl.ravel() > 0
        if inl.sum() < p.min_corners:
            return None, None
        if not self._markers_agree(gray, hom, plane[inl]):
            return None, None
        return nxt[inl], ids[inl]

    def _markers_agree(self, gray: np.ndarray, hom: np.ndarray,
                       plane: np.ndarray) -> bool:
        """Read the markers nearest the tracked corners through `hom`. False if
        any reads as a marker the board does not put there, or none reads."""
        centre = plane.mean(axis=0)
        order = np.argsort(np.linalg.norm(self._marker_obj.mean(axis=1) - centre, axis=1))
        seen = 0
        for m in order[: self._track.verify_markers]:
            got = self._read_marker(gray, hom, self._marker_obj[m])
            if got is None:
                continue
            if got != int(self._marker_ids[m]):
                return False
            seen += 1
        return seen > 0

    def _read_marker(self, gray: np.ndarray, hom: np.ndarray,
                     corners_plane: np.ndarray) -> int | None:
        """The marker ID at `corners_plane` seen through `hom`, or None when
        nothing valid is there — a black square, a hand, the image edge."""
        h, w = gray.shape
        px = cv2.perspectiveTransform(corners_plane.reshape(1, 4, 2), hom).reshape(4, 2)
        if (not np.isfinite(px).all() or (px < 0).any()
                or (px[:, 0] >= w).any() or (px[:, 1] >= h).any()):
            return None
        warp = cv2.getPerspectiveTransform(px.astype(np.float32), self._patch_dst)
        edge = self._cells * _CELL_PX
        patch = cv2.warpPerspective(gray, warp, (edge, edge), flags=cv2.INTER_LINEAR)
        _, bw = cv2.threshold(patch, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        cells = self._cells
        bits = bw.reshape(cells, _CELL_PX, cells, _CELL_PX).mean(axis=(1, 3)) > 127
        if bits[0].any() or bits[-1].any() or bits[:, 0].any() or bits[:, -1].any():
            return None                                  # no black border
        ok, idx, _rotation = self._dictionary.identify(
            bits[1:-1, 1:-1].astype(np.uint8), _CORRECTION_RATE)
        return int(idx) if ok else None


