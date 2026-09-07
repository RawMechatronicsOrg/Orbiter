"""The stripe mask and the inpaint that goes with it.

Every frame here is a real JPEG: the scene is drawn in floating point, encoded
with `cv2.imencode` at the same quality the camera uses and handed to `build`
as bytes, exactly as `recon.clean_images` will hand it the camera's own file.
That is deliberate rather than fussy — 4:2:0 chroma subsampling smears a red
line sideways and softens its crest before any detector sees it, and a test
that fed `build` a float array would be measuring a stripe this chain never
gets to look at.

The stripe is the live one: FWHM 12 px, so a Gaussian σ of 5.1 px. A bright
one saturates all three channels along its middle and reads WHITE there, which
puts a hole in the redness score right where the stripe is strongest; that
hole is part of what the dilation exists to close, so it is drawn rather than
smoothed away.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from orbiter_native import laser, stripemask
from orbiter_native.laser import StripePixels
from orbiter_native.stripemask import MASK_IGNORE, MASK_USE, MaskParams

#: The live stripe, and the Gaussian σ that matches it.
FWHM_PX = 12.0
SIGMA_PX = FWHM_PX / 2.3548

#: `laser.LaserParams.redness_min` — the level the SCAN needs before it sees a
#: stripe at all. Duplicated rather than imported so the dim test states the
#: number it is beating in the file that beats it.
SCAN_REDNESS_MIN = 45

#: The dim stripe's peak redness: under the scan's threshold, over the mask's.
DIM_PEAK = 20.0

#: The frame the rig actually delivers, and the size the plan's ≤ 8 % budget
#: is quoted against.
FULL_W, FULL_H = 1920, 1080

#: A stripe is "visible" where it added at least this many levels of red. Below
#: it the line is under the sensor's own noise and no mask is owed it.
VISIBLE_LEVELS = 10.0


def _visible_px(peak: float) -> float:
    """How far either side of the centre the stripe is still visible."""
    return SIGMA_PX * float(np.sqrt(2.0 * np.log(peak / VISIBLE_LEVELS)))


def _scene(w: int, h: int, rng, warm: float = 6.0) -> np.ndarray:
    """The rig's scene, near enough: a black-and-white board on a warmer bench.

    The bench is warm because a real one is — it is the surface `laser.py` says
    starts to qualify as stripe below a redness of about 30 — but it stays well
    under `MaskParams.redness_min` so that the mask's low threshold has
    something to be low ABOUT. The board is a checkerboard because its hard
    black/white edges are the strongest structure in the frame and the ridge
    term has to ignore them.
    """
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = 90 + 18 * np.sin(xx / 97.0) + 14 * np.sin(yy / 61.0)
    bgr = np.empty((h, w, 3), np.float32)
    bgr[..., 0] = base - warm
    bgr[..., 1] = base - warm
    bgr[..., 2] = base
    square = (((xx // 110).astype(int) + (yy // 110).astype(int)) % 2).astype(np.float32)
    inside = np.zeros((h, w), bool)
    inside[h // 6:h - h // 6, w // 6:w - w // 6] = True
    for c in range(3):
        bgr[..., c] = np.where(inside, 45 + 165 * square, bgr[..., c])
    return np.clip(bgr + rng.normal(0.0, 1.5, bgr.shape), 0, 255).astype(np.uint8)


def _stripe(bgr: np.ndarray, p0, p1, peak: float, *, core: float = 1.0):
    """Draw one laser stripe from `p0` to `p1`; return it and its geometry.

    `core` is how much of the peak also lands in green and blue along the very
    middle — 1.0 for a stripe bright enough to clip all three channels and read
    white, 0.0 for a dim one that only ever reddens what it touches.
    """
    h, w = bgr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    (ax, ay), (bx, by) = p0, p1
    dx, dy = bx - ax, by - ay
    length = float(np.hypot(dx, dy))
    across = np.abs((yy - ay) * dx - (xx - ax) * dy) / length
    along = ((xx - ax) * dx + (yy - ay) * dy) / (length * length)
    on = (along >= 0.0) & (along <= 1.0)
    g = np.exp(-0.5 * (across / SIGMA_PX) ** 2) * on
    out = bgr.astype(np.float32)
    out[..., 2] += peak * g
    if core:
        white = core * peak * g ** 8
        out[..., 0] += white
        out[..., 1] += white
    return np.clip(out, 0, 255).astype(np.uint8), across, on


def _jpeg(bgr: np.ndarray) -> bytes:
    """The frame as the camera would hand it over."""
    return stripemask.encode_jpeg(bgr)


def _frame(w: int, h: int, p0, p1, peak: float = 200.0, *, core: float = 1.0,
           seed: int = 7, warm: float = 6.0):
    """A JPEG of one stripe on the scene, plus the pixels it is drawn on."""
    rng = np.random.default_rng(seed)
    lit, across, on = _stripe(_scene(w, h, rng, warm), p0, p1, peak, core=core)
    return _jpeg(lit), on & (across <= _visible_px(peak))


def _detected(jpeg_bytes: bytes) -> StripePixels:
    """What the SCAN's own detector makes of the frame — the sidecar's term."""
    return laser.find_stripe_pixels(
        cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR))


def test_mask_covers_the_whole_synthetic_stripe():
    """Nothing visible of the stripe survives into the pixels COLMAP reads."""
    jpeg_bytes, stripe = _frame(960, 720, (120.0, 140.0), (830.0, 590.0))
    mask, _ = stripemask.build(jpeg_bytes, _detected(jpeg_bytes), None)

    covered = float((mask[stripe] == MASK_IGNORE).mean())
    assert covered >= 0.99, f"{covered:.4f} of the stripe is masked"


@pytest.mark.parametrize("p0, p1", [
    ((0.0, 0.0), (FULL_W - 1.0, FULL_H - 1.0)),
    ((FULL_W / 6 + 20.0, FULL_H / 6 + 20.0), (FULL_W * 5 / 6 - 20.0, FULL_H * 5 / 6 - 20.0)),
], ids=["full-diagonal", "across-the-board"])
def test_mask_keeps_most_of_the_frame(p0, p1):
    """A greedy mask is still a mask: the object has to survive it.

    The worst realistic case is the corner-to-corner stripe, 2203 px of it at
    45°, which is where a square structuring element would cost √2 more than a
    round one. The second case is the stripe lying entirely across the board,
    which a reviewer will ask about because board pixels are in `StripePixels`
    by construction — it is masked the same way and costs no more.
    """
    jpeg_bytes, stripe = _frame(FULL_W, FULL_H, p0, p1)
    mask, _ = stripemask.build(jpeg_bytes, _detected(jpeg_bytes), None)

    masked = float((mask == MASK_IGNORE).mean())
    assert masked <= 0.08, f"{masked:.4f} of the frame is masked"
    assert float((mask[stripe] == MASK_IGNORE).mean()) >= 0.99


def test_mask_polarity_is_zero_means_ignore():
    """0 on the stripe, 255 off it — COLMAP's convention, and the opposite of
    what `cv2.inpaint` wants, which is why `build` owns the inversion."""
    jpeg_bytes, stripe = _frame(960, 720, (120.0, 140.0), (830.0, 590.0))
    mask, _ = stripemask.build(jpeg_bytes, _detected(jpeg_bytes), None)

    assert mask.dtype == np.uint8 and mask.shape == (720, 960)
    assert set(np.unique(mask).tolist()) <= {MASK_IGNORE, MASK_USE}
    assert mask[stripe].max() == MASK_IGNORE
    assert mask[0, 0] == MASK_USE and mask[-1, 0] == MASK_USE


def test_mask_catches_a_dim_stripe_the_scan_detector_misses():
    """The recall case, and the reason the mask runs its own detection pass.

    A stripe peaking at 20 levels of redness is invisible to the scan — its
    threshold is 45 — so the sidecar for this frame carries no pixels at all.
    The mask still has to cover it, because PatchMatch does not care how faint
    a false correspondence was.
    """
    assert DIM_PEAK < SCAN_REDNESS_MIN
    jpeg_bytes, stripe = _frame(960, 720, (60.0, 150.0), (900.0, 570.0),
                                peak=DIM_PEAK, core=0.0)
    detected = _detected(jpeg_bytes)
    assert not detected.ok, f"the scan was supposed to miss this: {detected.count} px"

    mask, _ = stripemask.build(jpeg_bytes, detected, None)
    covered = float((mask[stripe] == MASK_IGNORE).mean())
    assert covered >= 0.99, f"{covered:.4f} of the dim stripe is masked"


def test_mask_includes_the_projected_kept_points():
    """Where the scan MEASURED the stripe is masked, not only where a detector
    can still see it — the sidecar's kept points say so in millimetres.

    The frame carries no stripe at all, so the projection is the only term with
    anything to contribute and the assertion cannot be satisfied by accident.
    """
    rng = np.random.default_rng(11)
    jpeg_bytes = _jpeg(_scene(960, 720, rng))
    K = np.array([[900.0, 0.0, 480.0], [0.0, 900.0, 360.0], [0.0, 0.0, 1.0]])
    dist = np.array([-0.09, 0.02, 0.0, 0.0, 0.0])
    R = np.eye(3)
    t_mm = np.array([0.0, 0.0, 400.0])
    xyz = np.array([[0.0, 0.0, 0.0], [12.0, -8.0, 25.0], [-30.0, 20.0, -15.0],
                    [0.0, 0.0, -500.0]])

    uv = stripemask.project_points(xyz, R, t_mm, K, dist)
    assert len(uv) == 3, "the point behind the camera should have been dropped"

    without, _ = stripemask.build(jpeg_bytes, None, None)
    with_points, _ = stripemask.build(jpeg_bytes, None, uv)
    for x, y in np.rint(uv).astype(int):
        assert with_points[y, x] == MASK_IGNORE
        assert without[y, x] == MASK_USE


def test_inpaint_removes_the_red_excess():
    """After the inpaint the stripe is not merely masked, it is gone.

    Measured as redness, because that is the thing the stripe added: the mean
    over the band it occupied has to land under the median of the frame as a
    whole — the frame being mostly bench, which carries a few levels of real
    warmth that the neutral board the stripe crosses does not.
    """
    jpeg_bytes, stripe = _frame(960, 720, (200.0, 200.0), (760.0, 520.0))
    before = laser.redness(cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR))
    _, clean = stripemask.build(jpeg_bytes, _detected(jpeg_bytes), None)
    after = laser.redness(clean)

    frame_median = float(np.median(before))
    assert frame_median > 0.0, "the fixture's bench was supposed to be warm"
    assert float(after[stripe].mean()) < frame_median
    assert float(after[stripe].mean()) < 0.05 * float(before[stripe].mean())


def test_mask_is_the_same_with_and_without_the_ridge_term_on_a_clear_stripe():
    """The ridge is a belt, not the trousers.

    On a stripe the threshold pass can see perfectly well the two masks have to
    agree — if turning the ridge on moved the answer here, it would be finding
    structure that is not stripe, and the ≤ 8 % budget would be the next thing
    to go. The tolerance is there for the crest of the saturated white core,
    where the redness score dips and only the ridge has anything to say.
    """
    jpeg_bytes, stripe = _frame(960, 720, (120.0, 140.0), (830.0, 590.0))
    detected = _detected(jpeg_bytes)
    with_ridge, _ = stripemask.build(jpeg_bytes, detected, None,
                                     MaskParams(use_ridge=True))
    without, _ = stripemask.build(jpeg_bytes, detected, None,
                                  MaskParams(use_ridge=False))

    disagree = float((with_ridge != without).mean())
    assert disagree <= 0.005, f"the ridge moved {disagree:.4f} of the frame"
    assert float((without[stripe] == MASK_IGNORE).mean()) >= 0.99


def test_clean_copy_is_lossless_and_the_colmap_copy_is_jpeg(tmp_path):
    """Two copies of the same inpainted pixels, and they are not the same file.

    `clean/` is the archive and has to survive the round trip bit for bit;
    `colmap/images_clean/` is JPEG because `image_undistorter` re-encodes
    everything it copies anyway, so a lossless intermediate would buy nothing
    and cost a gigabyte.
    """
    jpeg_bytes, _ = _frame(480, 360, (40.0, 60.0), (440.0, 300.0))
    mask, clean = stripemask.build(jpeg_bytes, _detected(jpeg_bytes), None)

    png = tmp_path / "left_0007.png"
    stripemask.write_clean(png, clean)
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert np.array_equal(
        cv2.imdecode(np.frombuffer(png.read_bytes(), np.uint8), cv2.IMREAD_COLOR),
        clean)

    as_jpeg = stripemask.encode_jpeg(clean)
    assert as_jpeg[:3] == b"\xff\xd8\xff"
    round_trip = cv2.imdecode(np.frombuffer(as_jpeg, np.uint8), cv2.IMREAD_COLOR)
    assert round_trip.shape == clean.shape
    assert not np.array_equal(round_trip, clean), "q95 JPEG is not lossless"
    assert float(np.abs(round_trip.astype(int) - clean.astype(int)).mean()) < 3.0

    # The mask goes out under the photograph's WHOLE name plus .png, which is
    # what COLMAP looks a mask up by, and it survives the trip unchanged.
    mask_path = tmp_path / "left_0007.jpg.png"
    stripemask.write_mask(mask_path, mask)
    read_back = cv2.imdecode(np.frombuffer(mask_path.read_bytes(), np.uint8),
                             cv2.IMREAD_GRAYSCALE)
    assert np.array_equal(read_back, mask)
