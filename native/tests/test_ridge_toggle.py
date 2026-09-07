"""The `use_ridge` toggle: what it wires up, and what it must leave alone.

`tests/test_ridge_bench.py` decides whether the ridge detector is the better
estimator. This file is about the switch that puts it in service: the detector
carries its crest out of the frame where the score still exists
(`laser.ridge_centres`), and the scan takes each scanline's position from that
crest wherever the two estimators are looking at the same thing
(`laser.prefer_ridge`).

The claim under test is a narrow one, and it is narrow on purpose. The toggle
moves POSITIONS. Which scanlines are live, how wide their runs had to be, what
the other eye confirmed and every count the panel shows are decided by the
thresholded pixels and the veto, and none of that may notice the switch — a
second estimator that also changed the rejection arithmetic would be a second
scanner, not a second estimator.

One thing the toggle does NOT do, which the plan's own note might lead a reader
to expect: it does not recover dim stripes inside `scan_frame`. Recall lives
upstream of the estimator — a frame whose stripe never clears `redness_min` has
no pixels, so the scan refuses it before any centre is taken, and the crest of
a frame like that is never consulted. Dim recall is `stripemask`'s to collect
(story B2). What the toggle buys the scan is the saturated core.
"""

from __future__ import annotations

import numpy as np
import pytest

from orbiter_native import gpu, ridge
from orbiter_native.laser import (
    RIDGE_AGREE_PX,
    LaserParams,
    StripePixels,
    find_stripe_pixels,
    prefer_ridge,
    ridge_centres,
)
from orbiter_native.scan import ScanParams, scan_frame

from test_stereo_scan import (
    BOARD2_R,
    BOARD2_T,
    KL,
    KR,
    R_TRUE,
    T_TRUE,
    WIDE,
    _curve,
    _pixels,
    _plane,
    _rig,
)

#: Every count `scan_frame` reports. The toggle may not touch one of them.
COUNTERS = ("n_scanlines", "n_pixels", "n_confirmed", "n_rejected_unconfirmed",
            "n_rejected_blob", "n_split", "n_rejected_range", "n_rejected_jump",
            "n_rejected_offside", "n_rejected_volume", "along_x")

#: A sub-pixel step small enough that no gate downstream can notice it — at
#: 500 mm on the synthetic rig one pixel across the stripe is about 3 mm of
#: depth, so this is a third of a millimetre — and large enough to be
#: unmistakable in the output.
NUDGE_PX = 0.1

#: A crest this far from the confirmed run is another feature: past
#: `RIDGE_AGREE_PX`, which is what `prefer_ridge` exists to catch.
GLINT_PX = 12.0


def _stripe_frame(centre: float = 44.0, peak: float = 200.0, *, w: int = 96,
                  h: int = 90) -> np.ndarray:
    """A BGR frame carrying one horizontal Gaussian stripe of the live width,
    which is what `find_stripe_pixels` is given in service."""
    x = np.arange(w)
    y = np.arange(h)[:, None]
    profile = peak * np.exp(-0.5 * ((y - centre) / (12.0 / 2.3548)) ** 2)
    bgr = np.zeros((h, w, 3), np.uint8)
    bgr[:, :, 2] = np.clip(np.rint(profile * np.ones_like(x, float)), 0, 255).astype(np.uint8)
    return bgr


def _with_crest(pixels: StripePixels, scan: np.ndarray, pos: np.ndarray) -> StripePixels:
    """The same stripe pixels, told what the ridge found."""
    return StripePixels(x=pixels.x, y=pixels.y, w=pixels.w, r=pixels.r, wh=pixels.wh,
                        along_x=pixels.along_x, reason=pixels.reason,
                        crest=(np.asarray(scan, float), np.asarray(pos, float)))


# ── what the detector carries out ────────────────────────────────────────


def test_the_detector_takes_the_crest_only_when_it_is_asked() -> None:
    """The flag is the whole switch: nothing else about the pixel list moves
    with it, so a scan that turns it off is the scan that shipped before."""
    bgr = _stripe_frame()
    on = find_stripe_pixels(bgr, LaserParams(use_ridge=True))
    off = find_stripe_pixels(bgr, LaserParams(use_ridge=False))

    assert off.crest is None
    assert on.crest is not None
    assert np.array_equal(on.x, off.x) and np.array_equal(on.y, off.y)
    assert np.array_equal(on.w, off.w) and np.array_equal(on.r, off.r)
    assert on.along_x is off.along_x and on.wh == off.wh


def test_the_crest_is_in_whole_frame_coordinates_under_a_band() -> None:
    """`rows` searches a band, and the ridge runs on that band — so its rows
    are the band's and have to be carried back to the frame's. Getting this
    wrong moves every point of the scan by the band's offset, which is the
    kind of error that looks like a calibration problem."""
    bgr = _stripe_frame(centre=60.0, h=140)
    whole = find_stripe_pixels(bgr, LaserParams(use_ridge=True))
    banded = find_stripe_pixels(bgr, LaserParams(use_ridge=True), rows=(30, 110))

    assert whole.crest is not None and banded.crest is not None
    assert np.array_equal(whole.crest[0], banded.crest[0])
    # Not identical — the band mirrors at its own edges, so the response near
    # them differs — but the same stripe, found where the stripe is.
    assert np.abs(banded.crest[1] - 60.0).max() < 0.05, np.abs(banded.crest[1] - 60.0).max()
    assert np.abs(banded.crest[1] - whole.crest[1]).max() < 0.05


def test_a_band_too_thin_for_the_kernel_yields_no_crest() -> None:
    """31 taps mirror 15 px into the band. A thinner one is not a failure to
    report, it is a band with no room for a stripe: `ridge.response` refuses
    it by name and the detector must not carry that refusal into the scan."""
    assert ridge_centres(np.zeros((12, 40), np.float32), True) is None
    assert ridge_centres(np.zeros((40, 12), np.float32), True) is None
    assert ridge_centres(np.zeros((40, 40), np.float32), True) is not None


# ── what the chooser does with it ────────────────────────────────────────


def test_the_crest_is_taken_where_the_two_agree() -> None:
    scan = np.array([3.0, 4.0, 5.0])
    pos = np.array([10.0, 20.0, 30.0])
    pixels = _with_crest(StripePixels(), scan, pos + 0.3)
    assert np.allclose(prefer_ridge(scan, pos, pixels), pos + 0.3)


def test_a_crest_from_another_feature_is_refused() -> None:
    """One scanline can carry the stripe and a glint. The veto has already
    said which run is the stripe; `ridge.centres` reports the strongest crest
    of the whole band and has not been told."""
    scan = np.array([3.0, 4.0, 5.0])
    pos = np.array([10.0, 20.0, 30.0])
    moved = np.array([10.2, 20.0 + GLINT_PX, 30.0 - GLINT_PX])
    got = prefer_ridge(scan, pos, _with_crest(StripePixels(), scan, moved))
    assert got[0] == pytest.approx(10.2)
    assert got[1] == 20.0 and got[2] == 30.0
    # And the bound is where it says it is, from either side.
    edge = np.array([10.0 + RIDGE_AGREE_PX, 20.0 - RIDGE_AGREE_PX, 30.0])
    assert np.allclose(prefer_ridge(scan, pos, _with_crest(StripePixels(), scan, edge)), edge)


def test_a_scanline_the_ridge_missed_keeps_the_centroid() -> None:
    """The two estimators drop scanlines on different grounds, so neither
    list contains the other. A missing crest is not a missing point."""
    scan = np.array([3.0, 4.0, 5.0, 9.0])
    pos = np.array([10.0, 20.0, 30.0, 40.0])
    pixels = _with_crest(StripePixels(), np.array([4.0, 7.0, 9.0]),
                         np.array([20.5, 99.0, 40.5]))
    assert np.allclose(prefer_ridge(scan, pos, pixels), [10.0, 20.5, 30.0, 40.5])


def test_no_crest_leaves_the_positions_exactly_alone() -> None:
    """The off path, and the path every producer that predates this took."""
    scan, pos = np.array([1.0, 2.0]), np.array([7.5, 8.25])
    assert prefer_ridge(scan, pos, StripePixels()) is pos
    assert prefer_ridge(scan, pos, _with_crest(StripePixels(), [], [])) is pos
    assert prefer_ridge(np.empty(0), np.empty(0),
                        _with_crest(StripePixels(), scan, pos)).size == 0


# ── and what the scan does with the chooser ──────────────────────────────


def _scan_with(crest_shift: float | None):
    """One frame pair through `scan_frame`, with the left eye's crest sitting
    `crest_shift` px off the centroid the estimator would have produced —
    or with no crest at all."""
    rig, plane = _rig(), _plane()
    truth = _curve(lambda x: 500.0 + 30.0 * np.sin(x / 25.0))
    left = _pixels(KL, np.eye(3), np.zeros(3), truth)
    right = _pixels(KR, R_TRUE, T_TRUE, truth)
    base = scan_frame(rig, plane, left, right, BOARD2_R, BOARD2_T, WIDE)
    if crest_shift is None:
        return base, base
    # The estimator's own answer, moved by a known amount: the only way to
    # say what the crest did is to know what it was measured against.
    order = np.argsort(base.pixels_left[:, 0])
    crest = (base.pixels_left[order, 0], base.pixels_left[order, 1] + crest_shift)
    with_ridge = scan_frame(rig, plane, _with_crest(left, *crest), right,
                            BOARD2_R, BOARD2_T, WIDE)
    return base, with_ridge


def test_the_toggle_moves_the_position_and_leaves_every_count_alone() -> None:
    """The claim in one test: same scanlines, same counts, positions moved by
    exactly what the crest asked for."""
    base, moved = _scan_with(NUDGE_PX)
    assert base.reason is None and moved.reason is None
    for name in COUNTERS:
        assert getattr(moved, name) == getattr(base, name), name
    assert np.array_equal(moved.scanlines, base.scanlines)
    assert moved.veto_px == pytest.approx(base.veto_px)

    # Along the scanline nothing moved; across it, everything moved once.
    assert np.allclose(moved.pixels_left[:, 0], base.pixels_left[:, 0])
    assert np.allclose(moved.pixels_left[:, 1] - base.pixels_left[:, 1], NUDGE_PX)
    # And that pixel became depth: a tenth of a pixel is a third of a
    # millimetre at 500 mm on this rig, and it went the one way.
    dz = moved.points_camera[:, 2] - base.points_camera[:, 2]
    assert np.all(np.abs(dz) < 1.0) and np.abs(np.median(dz)) > 0.05, np.median(dz)


def test_a_glint_sized_crest_leaves_the_scan_exactly_as_it_was() -> None:
    """`prefer_ridge`'s bound, seen from the scan: a crest 12 px away is not
    a correction, and the frame that results is the frame with no crest at
    all — point for point."""
    base, glint = _scan_with(GLINT_PX)
    assert glint.reason is None
    for name in COUNTERS:
        assert getattr(glint, name) == getattr(base, name), name
    assert np.array_equal(glint.pixels_left, base.pixels_left)
    assert np.array_equal(glint.points_camera, base.points_camera)


# ── the GPU path carries the same number ─────────────────────────────────


@pytest.mark.skipif(not gpu.available(), reason=gpu.describe())
def test_the_gpu_path_finds_the_same_crest_as_the_cpu_one() -> None:
    """Two implementations, one estimator. The GPU path runs the ridge on the
    score tensor it has just computed and never brings the score down; the
    number it hands the scan has to be the one the reference would have."""
    import torch

    bgr = _stripe_frame(centre=44.3, h=96)
    p = LaserParams(use_ridge=True)
    cpu = find_stripe_pixels(bgr, p)
    rgb = np.ascontiguousarray(bgr[:, :, ::-1]).transpose(2, 0, 1)
    on_gpu = gpu.stripe_pixels(torch.from_numpy(np.ascontiguousarray(rgb)).cuda(), p)

    assert cpu.crest is not None and on_gpu.crest is not None
    assert np.array_equal(cpu.crest[0], on_gpu.crest[0])
    # The two score images differ by a level or two (float against uint8
    # stages, `test_gpu.py` measures it at <= 3), and a level of score is
    # worth well under a hundredth of a pixel of crest.
    assert np.abs(cpu.crest[1] - on_gpu.crest[1]).max() < 0.02

    # And under a band, where the row the ridge reports is the band's and
    # both paths have to add the same offset back.
    band = (12, 88)
    cpu_band = find_stripe_pixels(bgr, p, rows=band)
    gpu_band = gpu.stripe_pixels(torch.from_numpy(np.ascontiguousarray(rgb)).cuda(),
                                 p, rows=band)
    assert cpu_band.crest is not None and gpu_band.crest is not None
    assert np.abs(gpu_band.crest[1] - 44.3).max() < 0.05
    assert np.abs(cpu_band.crest[1] - gpu_band.crest[1]).max() < 0.02

    assert gpu.stripe_pixels(torch.from_numpy(np.ascontiguousarray(rgb)).cuda(),
                             LaserParams(use_ridge=False)).crest is None


@pytest.mark.skipif(not ridge.cuda_available(), reason=ridge.describe())
def test_the_gpu_path_leaves_the_score_on_the_card() -> None:
    """The reason the ridge runs inside the detector rather than at the call
    site. `ridge.response` returns where its input lives, so a CUDA score
    yields CUDA tensors and only `centres`' handful of numbers is downloaded."""
    import torch

    score = torch.from_numpy(_stripe_frame()[:, :, 2].astype(np.float32)).cuda()
    r, nx, ny = ridge.response(score)
    assert r.is_cuda and nx.is_cuda and ny.is_cuda
    scan, pos = ridge.centres(r, nx, ny, "x")
    assert isinstance(scan, np.ndarray) and isinstance(pos, np.ndarray)
