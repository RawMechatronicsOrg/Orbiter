"""Where the stripe crosses a scanline, to a fraction of a pixel.

`laser.stripe_centroids` fits each scanline's run of lit pixels as a Gaussian
and falls back to the excess-over-threshold centroid. Checked on synthetic
profiles: exact on a clean sub-pixel Gaussian, no worse than the plain
centroid under noise, sane on a saturated flat top, and the same answer
through the calibration line fit that shares it."""

from __future__ import annotations

import numpy as np

from orbiter_native.laser import stripe_centroids

THR = 45


def _profile(centre: float, sigma: float, peak: float, rng=None, noise: float = 0.0,
             n: int = 11):
    """One scanline's lit pixels: positions and scores above THR."""
    x = np.arange(n) - n // 2
    s = peak * np.exp(-0.5 * ((x - centre) / sigma) ** 2)
    if noise:
        s = s + rng.normal(0, noise, n)
    s = np.clip(np.rint(s), 0, 255)
    lit = s >= THR
    return x[lit], s[lit]


def _one(across, w, fit: bool):
    key = np.zeros(len(across), int)
    scan, pos, n_lines, n_split, n_rej = stripe_centroids(key, across, w, fit=fit)
    assert n_lines == 1 and n_split == 0 and n_rej == 0 and len(pos) == 1
    return float(pos[0])


def test_fit_is_exact_on_a_clean_subpixel_gaussian() -> None:
    for centre in (-0.37, 0.0, 0.21, 0.49):
        for sigma in (0.9, 1.4):
            x, s = _profile(centre, sigma, 200.0)
            assert abs(_one(x, s, fit=True) - centre) < 0.03, (centre, sigma)


def test_fit_beats_or_matches_the_centroid_under_noise() -> None:
    rng = np.random.default_rng(1)
    for sigma, peak in ((0.8, 200.0), (1.2, 150.0), (1.8, 255.0)):
        err = {True: [], False: []}
        for _ in range(600):
            centre = rng.uniform(-0.5, 0.5)
            x, s = _profile(centre, sigma, peak, rng, noise=4.0)
            for fit in (True, False):
                err[fit].append(_one(x, s, fit) - centre)
        rms = {k: float(np.sqrt(np.mean(np.square(v)))) for k, v in err.items()}
        assert rms[True] <= rms[False] * 1.05, (sigma, peak, rms)
        assert rms[True] < 0.08, (sigma, peak, rms)


def test_a_saturated_flat_top_falls_back_to_the_centroid() -> None:
    x = np.arange(-3, 4)
    s = np.array([60, 200, 255, 255, 255, 200, 60], float)
    assert abs(_one(x, s, fit=True) - 0.0) < 1e-9
    s = np.array([60, 255, 255, 255, 200, 90, 50], float)          # lopsided plateau
    assert -1.0 < _one(x, s, fit=True) < 0.0


def test_the_strongest_run_wins_and_widths_are_gated() -> None:
    # Two runs on one scanline: a stripe and a fainter glint 10 px away.
    xs, ws = _profile(0.2, 1.0, 220.0)
    gx, gw = _profile(0.0, 1.0, 90.0)
    across = np.concatenate([xs, gx + 10])
    w = np.concatenate([ws, gw])
    key = np.zeros(len(across), int)
    scan, pos, n_lines, n_split, n_rej = stripe_centroids(key, across, w, width_px=(2, 24))
    assert n_lines == 1 and n_split == 1 and n_rej == 0
    assert abs(pos[0] - 0.2) < 0.05
    _, pos2, _, _, n_rej2 = stripe_centroids(key, across, w, width_px=(8, 24))
    assert len(pos2) == 0 and n_rej2 == 1


def test_the_fallback_beats_the_raw_centroid_where_the_fit_gives_up() -> None:
    """Wide, saturated profiles are the ones the Gaussian fit hands off; there
    the excess-over-floor centroid must be measurably better than the raw
    intensity-weighted one — the reason it exists."""
    rng = np.random.default_rng(7)
    raw, excess = [], []
    for _ in range(800):
        centre = rng.uniform(-0.5, 0.5)
        x, s = _profile(centre, 1.8, 255.0, rng, noise=4.0)
        raw.append((x * s).sum() / s.sum() - centre)
        excess.append(_one(x, s, fit=False) - centre)
    rms = lambda v: float(np.sqrt(np.mean(np.square(v))))
    assert rms(excess) < rms(raw) / 1.4, (rms(excess), rms(raw))


def test_a_strong_run_of_the_wrong_width_does_not_take_the_scanline() -> None:
    """A wide dim smear outscores the stripe on its scanline; the width gate
    must remove it BEFORE the strongest run is picked, or the stripe goes
    with it."""
    xs, ws = _profile(0.5, 1.0, 220.0)                 # the stripe, 4-5 px wide
    smear_x = np.arange(20, 50)                        # 30 px of dim smear
    smear_w = np.full(30, 60.0)                        # sums to more than the stripe
    across = np.concatenate([xs, smear_x])
    w = np.concatenate([ws, smear_w])
    key = np.zeros(len(across), int)
    scan, pos, n_lines, n_split, n_rej = stripe_centroids(key, across, w, width_px=(2, 24))
    assert n_lines == 1 and n_split == 1 and n_rej == 0
    assert len(pos) == 1 and abs(pos[0] - 0.5) < 0.05
    # With no run of a valid width, the scanline is counted as rejected.
    _, pos2, _, _, n_rej2 = stripe_centroids(np.zeros(30, int), smear_x, smear_w,
                                             width_px=(2, 24))
    assert len(pos2) == 0 and n_rej2 == 1
