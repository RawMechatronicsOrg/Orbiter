"""The ridge response, and the sub-pixel centres it yields.

`ridge` answers the two questions the colour detector cannot. A saturated
stripe has a flat top, so `laser.stripe_centroids` loses its Gaussian fit and
is left with the flanks; a dim stripe never clears `redness_min` at all, so no
scanline is even offered to it. Both tests below are built to show exactly
that, and the saturated one is a direct comparison against the estimator it
would replace rather than a bar picked out of the air.

The synthetic stripe is the live one: FWHM 12 px, so a Gaussian σ of 5.1 px,
rounded to whole levels and clipped at 255 the way a real score image is.
"""

from __future__ import annotations

import numpy as np
import pytest

from orbiter_native import ridge
from orbiter_native.laser import stripe_centroids

#: The live stripe, as the plan measured it, and the Gaussian σ that matches.
FWHM_PX = 12.0
SIGMA_PX = FWHM_PX / 2.3548

#: `laser.LaserParams.redness_min` — the level below which the scan sees no
#: stripe at all. Duplicated rather than imported so the dim test states the
#: number it is beating in the file that beats it.
REDNESS_MIN = 45

#: The peak that leaves a 3 px flat top once clipped at 255: the profile is
#: at or above 255 while |d| <= 1.5, i.e. peak = 255·exp(½·(1.5/σ)²) = 266.
PLATEAU_PEAK = 255.0 * np.exp(0.5 * (1.5 / SIGMA_PX) ** 2)

#: Sensor noise in whole levels, the figure the repo already measures this
#: class of estimator at
#: (`test_centroid.py::test_fit_beats_or_matches_the_centroid_under_noise`).
SENSOR_NOISE = 4.0

gpu_only = pytest.mark.skipif(not ridge.cuda_available(), reason=ridge.describe())


def _stripe(centre: float, peak: float = 200.0, *, w: int = 96, h: int = 80,
            slope: float = 0.0, noise: float = 0.0, rng=None) -> np.ndarray:
    """One Gaussian stripe across a score image, as the score would arrive.

    Whole levels clipped to 0..255, because that is what `laser.stripe_score`
    hands over and both the rounding and the ceiling are part of what the two
    estimators have to live with.
    """
    x = np.arange(w)
    y = np.arange(h)[:, None]
    s = peak * np.exp(-0.5 * ((y - (centre + slope * x)) / SIGMA_PX) ** 2) * np.ones_like(x, float)
    if noise:
        s = s + rng.normal(0.0, noise, s.shape)
    return np.clip(np.rint(s), 0, 255).astype(np.float32)


def _laser_centres(score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`laser.stripe_centroids` over the same image, thresholded the way the
    scan thresholds it: pixels at or above `redness_min`, columns as
    scanlines. This is `laser._centroids` with the lit test spelled out."""
    ys, xs = np.nonzero(score >= REDNESS_MIN)
    if not len(ys):
        return np.empty(0), np.empty(0)
    scan, pos, *_ = stripe_centroids(xs, ys, score[ys, xs])
    return scan, pos


def _rms(v) -> float:
    return float(np.sqrt(np.mean(np.square(v))))


def test_response_peaks_on_the_ridge() -> None:
    """A crest where the line is, nothing where it is not, and a normal that
    points across it."""
    img = _stripe(40.0)
    r, nx, ny = ridge.response(img)
    assert r.shape == nx.shape == ny.shape == img.shape
    assert r.dtype == nx.dtype == ny.dtype == np.float32

    column = r[:, 48]
    assert int(np.argmax(column)) == 40, np.argmax(column)
    assert column[40] > 0.0
    # Off the line the profile is convex, so the curvature across it is
    # positive and the ridge strength is negative: not a bright line at all.
    assert column[5] < 0.01 * column[40], (column[5], column[40])

    # The stripe is horizontal, so the direction across it is straight down
    # the y axis. The eigenvector is pinned to one half-plane, so this is a
    # statement about the value and not only about the magnitude.
    assert abs(float(nx[40, 48])) < 1e-3, nx[40, 48]
    assert float(ny[40, 48]) > 0.999, ny[40, 48]
    assert np.allclose(nx ** 2 + ny ** 2, 1.0, atol=1e-5)


@gpu_only
def test_cpu_and_gpu_agree() -> None:
    """Same kernels, same border, same numbers — from either backend, and
    whether the frame arrives as an array or as a tensor already on CUDA."""
    import torch

    img = _stripe(40.3, slope=0.06)
    r, nx, ny = ridge.response(img)
    asked = ridge.response(img, use_gpu=True)
    given = ridge.response(torch.from_numpy(img).cuda())
    assert asked[0].is_cuda and given[0].is_cuda

    for got in (asked, given):
        gr, gnx, gny = (t.cpu().numpy() for t in got)
        assert np.abs(gr - r).max() <= 1e-5 * np.abs(r).max(), np.abs(gr - r).max()
        # An eigenvector is a direction, not an arrow: compare the two as
        # directions, which is the only thing either backend promises.
        assert np.abs(nx * gnx + ny * gny).min() > 1.0 - 1e-5
        # The centres agree to 4e-6 px on every scanline but the two where a
        # quantised score puts the crest exactly between two samples: the
        # response ties there, the two backends break the tie differently,
        # and the vertex found from either sample lands 0.002 px apart. That
        # is the property worth asserting — not which sample won.
        cpu = ridge.centres(r, nx, ny, "x")
        gpu = ridge.centres(gr, gnx, gny, "x")
        assert np.array_equal(cpu[0], gpu[0])
        assert np.abs(cpu[1] - gpu[1]).max() < 0.005, np.abs(cpu[1] - gpu[1]).max()

    # The override works the other way too: a frame already on the card,
    # forced onto the reference path, comes back as numpy and bit-identical
    # to the same frame that never left the host.
    back = ridge.response(torch.from_numpy(img).cuda(), use_gpu=False)
    assert isinstance(back[0], np.ndarray)
    assert np.array_equal(back[0], r)


def test_centres_recover_a_known_subpixel_offset() -> None:
    """An unsaturated stripe, placed to a fraction of a pixel and found
    there — on either scan axis, since a transposed image is the same
    problem with the roles of x and y exchanged."""
    for delta in (0.0, 0.17, 0.31, 0.5, -0.28):
        centre = 40.0 + delta
        img = _stripe(centre)
        scan, pos = ridge.centres(*ridge.response(img), "x")
        assert np.array_equal(scan, np.arange(img.shape[1])), len(scan)
        assert np.abs(pos - centre).max() < 0.05, (delta, np.abs(pos - centre).max())

        turned = np.ascontiguousarray(img.T)
        scan_y, pos_y = ridge.centres(*ridge.response(turned), "y")
        assert np.array_equal(scan_y, scan)
        assert np.abs(pos_y - centre).max() < 0.05, (delta, np.abs(pos_y - centre).max())


def test_ridge_is_no_worse_than_the_current_estimator_on_a_saturated_plateau() -> None:
    """The same frames to both estimators, and the ridge must not lose.

    The bar is the estimator in service rather than a fixed number of pixels:
    `laser.stripe_centroids` already measures 0.035-0.081 px on its own
    benchmark, so any absolute bar loose enough to be safe would be loose
    enough to pass a regression.

    Noise is here for a reason. A noiseless synthetic plateau is perfectly
    symmetric about its centre and both estimators land within 0.003 px of
    it — a comparison of float rounding, not of estimators. Saturation costs
    the fit its footing only when there is something to be unsure about, so
    the frames carry the same noise the repo's own centroid benchmark uses.
    """
    # The fixture is a 3 px flat top, asserted on the noiseless profile so it
    # cannot quietly stop being saturated: the ceiling covers |d| <= 1.5,
    # which is three or four whole samples depending on where between them
    # the centre falls.
    assert int((_stripe(40.0, PLATEAU_PEAK)[:, 0] >= 255).sum()) == 3
    assert int((_stripe(40.5, PLATEAU_PEAK)[:, 0] >= 255).sum()) == 4

    rng = np.random.default_rng(7)
    ridge_err, laser_err = [], []
    for _ in range(40):
        centre = 40.0 + rng.uniform(-0.5, 0.5)
        img = _stripe(centre, PLATEAU_PEAK, noise=SENSOR_NOISE, rng=rng)
        ridge_err.extend(ridge.centres(*ridge.response(img), "x")[1] - centre)
        laser_err.extend(_laser_centres(img)[1] - centre)

    assert len(ridge_err) == len(laser_err) == 40 * 96, (len(ridge_err), len(laser_err))
    got, bar = _rms(ridge_err), _rms(laser_err)
    assert got <= bar, f"ridge {got:.4f} px RMS against the current {bar:.4f} px"


def test_ridge_recovers_a_dim_stripe_the_threshold_misses() -> None:
    """Below `redness_min` the scan has no pixels and therefore no points.
    The ridge does not care how bright the stripe is, only that it is one."""
    for peak in (15.0, 20.0, 30.0, 40.0):
        rng = np.random.default_rng(11)
        errors, offered = [], 0
        for _ in range(20):
            centre = 40.0 + rng.uniform(-0.5, 0.5)
            img = _stripe(centre, peak, noise=1.0, rng=rng)
            assert img.max() < REDNESS_MIN, (peak, img.max())
            scan, pos = ridge.centres(*ridge.response(img), "x")
            assert len(scan) == img.shape[1], (peak, len(scan))
            errors.extend(pos - centre)
            offered += len(_laser_centres(img)[1])
        assert offered == 0, (peak, offered)
        err = np.abs(errors)
        assert np.median(err) < 0.05, (peak, np.median(err))
        assert np.percentile(err, 95) < 0.15, (peak, np.percentile(err, 95))


def test_the_response_says_why_it_cannot_run() -> None:
    """Each refusal names its own cause; the fixes are different."""
    with pytest.raises(ValueError, match="one score plane"):
        ridge.response(np.zeros((8, 8, 3), np.float32))
    # 31 taps mirror 15 px into the frame, and a 12 px frame has not got them.
    with pytest.raises(ValueError, match="mirrors 15 px"):
        ridge.response(np.zeros((12, 40), np.float32))
    with pytest.raises(ValueError, match="scanline"):
        ridge.centres(*ridge.response(_stripe(40.0)), "rows")
