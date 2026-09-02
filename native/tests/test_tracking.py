"""The corner track between full ChArUco passes, on synthetic frames.

A rendered board is warped into a frame with a per-frame homography — a slow
drift and turn, the way a hand moves it — so every frame also has a fresh
detection to compare the track against. What is checked: tracked corners agree
with a detection of the same frame to well under a pixel with the IDs intact,
a full pass returns on schedule, motion the tracker cannot vouch for falls
back to detection instead of yielding wrong corners, the track ends when the
board leaves, and it holds through blur that defeats the marker decoder.
"""

from __future__ import annotations

import numpy as np
import cv2
import pytest

from orbiter_native.cvcore import BoardSpec, Intrinsics, build_board, charuco_detect
from orbiter_native.detect import BoardDetector, TrackParams

W, H = 1280, 720
SPEC = BoardSpec(squares_x=8, squares_y=8, square_length_mm=36.0,
                 marker_length_mm=26.64, aruco_dict_id=5)


@pytest.fixture(scope="module")
def board():
    return build_board(SPEC)


@pytest.fixture(scope="module")
def tile(board):
    return board.generateImage((800, 800), marginSize=40, borderBits=1)


def render(tile, k: int, dx: float = 0.0, blur: float = 0.8) -> np.ndarray:
    """Frame `k` of a drift: 3 px right and 2 px down per frame, turning 0.3°,
    seen with mild perspective. `dx` shifts this one frame on top of that."""
    ang = np.radians(0.3 * k)
    c, s = np.cos(ang), np.sin(ang)
    quad = np.float32([[0, 0], [400, 24], [384, 416], [16, 384]]) - 200
    quad = quad @ np.array([[c, -s], [s, c]]).T + [560 + 3 * k + dx, 360 + 2 * k]
    src = np.float32([[0, 0], [800, 0], [800, 800], [0, 800]])
    hom = cv2.getPerspectiveTransform(src, quad.astype(np.float32))
    img = cv2.warpPerspective(tile, hom, (W, H), borderMode=cv2.BORDER_CONSTANT,
                              borderValue=140)
    if blur:
        img = cv2.GaussianBlur(img, (0, 0), blur)
    rng = np.random.default_rng(k)
    noisy = img.astype(np.float32) + rng.normal(0, 3, img.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def by_id(corners, ids) -> dict[int, np.ndarray]:
    return {int(i): c for i, c in zip(ids.ravel(), corners.reshape(-1, 2))}


def _agree(hit, fresh: dict[int, np.ndarray], max_px: float) -> None:
    """Tracked corners share their IDs with a fresh detection of the same
    frame and sit where it puts them."""
    got = by_id(hit.corners, hit.ids)
    common = set(got) & set(fresh)
    assert len(common) >= 0.9 * len(got), (len(common), len(got))
    err = np.array([np.linalg.norm(got[i] - fresh[i]) for i in common])
    assert np.median(err) < 0.3, np.median(err)
    assert err.max() < max_px, err.max()


def test_tracks_between_full_passes(board, tile) -> None:
    det = BoardDetector(SPEC)
    first = det.detect(render(tile, 0))
    assert not first.tracked and first.count >= 30
    for k in range(1, 8):
        g = render(tile, k)
        hit = det.detect(g)
        assert hit.tracked, k
        assert hit.count >= 0.9 * first.count
        _agree(hit, by_id(*charuco_detect(g, board)), max_px=1.0)


def test_full_pass_returns_on_schedule(tile) -> None:
    det = BoardDetector(SPEC, TrackParams(redetect_every=4))
    flags = [det.detect(render(tile, k)).tracked for k in range(10)]
    assert flags == [False, True, True, True] * 2 + [False, True]


def test_every_frame_is_a_full_pass_when_asked(tile) -> None:
    det = BoardDetector(SPEC, TrackParams(redetect_every=1))
    assert not any(det.detect(render(tile, k)).tracked for k in range(4))


def test_motion_past_the_reach_falls_back_to_detection(board, tile) -> None:
    det = BoardDetector(SPEC)
    det.detect(render(tile, 0))
    assert det.detect(render(tile, 1)).tracked
    g = render(tile, 2, dx=300.0)
    hit = det.detect(g)
    assert not hit.tracked
    _agree(hit, by_id(*charuco_detect(g, board)), max_px=0.01)


def test_a_whole_square_jump_is_not_mistaken_for_its_neighbour(board, tile) -> None:
    """A checkerboard shifted by one square lands its saddles on saddles, and
    a homography over the corners cannot tell. The step guard must: the frame
    either comes from a full pass or carries the right IDs."""
    det = BoardDetector(SPEC)
    first = det.detect(render(tile, 0))
    xy = first.corners.reshape(-1, 2)
    d = np.linalg.norm(xy[:, None] - xy[None, :], axis=2)
    np.fill_diagonal(d, np.inf)
    square_px = float(np.median(d.min(axis=1)))
    g = render(tile, 1, dx=square_px)
    hit = det.detect(g)
    fresh = by_id(*charuco_detect(g, board))
    if hit.tracked:
        _agree(hit, fresh, max_px=1.0)
    else:
        _agree(hit, fresh, max_px=0.01)


def test_the_track_ends_when_the_board_leaves(tile) -> None:
    det = BoardDetector(SPEC)
    assert det.detect(render(tile, 0)).count
    assert det.detect(render(tile, 1)).tracked
    gone = det.detect(np.full((H, W), 140, np.uint8))
    assert gone.count == 0 and not gone.tracked
    back = det.detect(render(tile, 3))
    assert back.count and not back.tracked


def test_the_track_does_not_outlive_the_markers(board, tile) -> None:
    """Blur that defeats the marker decoder leaves saddles KLT can still
    follow — with no way to vouch for their IDs. The track must end there
    rather than carry on unverified."""
    det = BoardDetector(SPEC)
    det.detect(render(tile, 0))
    blurred = render(tile, 1, blur=5.0)
    corners, _ = charuco_detect(blurred, board)
    assert corners is None, "precondition: the marker decoder must fail here"
    hit = det.detect(blurred)
    assert not hit.tracked and hit.count == 0


def test_a_hidden_marker_does_not_end_the_track(board, tile) -> None:
    """A hand over one marker reads as nothing, not as a wrong ID: the other
    markers still vouch for the track."""
    det = BoardDetector(SPEC)
    first = det.detect(render(tile, 0))
    g = render(tile, 1)
    # Paint over the marker nearest the tracked corners' centroid — the first
    # one the check reads.
    obj, _ = board.matchImagePoints(first.corners, first.ids)
    centre = obj.reshape(-1, 3)[:, :2].mean(axis=0)
    markers = np.asarray(board.getObjPoints(), np.float64)[:, :, :2]
    m = int(np.argmin(np.linalg.norm(markers.mean(axis=1) - centre, axis=1)))
    plane = obj.reshape(-1, 3)[:, :2].astype(np.float64)
    hom, _ = cv2.findHomography(plane, first.corners.reshape(-1, 2).astype(np.float64), 0)
    px = cv2.perspectiveTransform(markers[m].reshape(1, 4, 2), hom).reshape(4, 2)
    cv2.fillConvexPoly(g, np.rint(px).astype(np.int32), 120)
    hit = det.detect(g)
    assert hit.tracked
    _agree(hit, by_id(*charuco_detect(render(tile, 1), board)), max_px=1.0)


def test_pose_from_tracked_corners_matches_detection(tile) -> None:
    k = Intrinsics(fx=1000.0, fy=1000.0, cx=W / 2, cy=H / 2, dist=(0.0,) * 5)
    tracking = BoardDetector(SPEC)
    fresh = BoardDetector(SPEC, TrackParams(redetect_every=1))
    tracking.detect(render(tile, 0), k)
    fresh.detect(render(tile, 0), k)
    for j in range(1, 5):
        g = render(tile, j)
        a, b = tracking.detect(g, k), fresh.detect(g, k)
        assert a.tracked and not b.tracked
        assert a.t is not None and b.t is not None
        assert np.linalg.norm(a.t - b.t) < 1.0, (a.t, b.t)
        angle = np.degrees(np.arccos(np.clip((np.trace(a.R.T @ b.R) - 1) / 2, -1, 1)))
        assert angle < 0.3, angle
