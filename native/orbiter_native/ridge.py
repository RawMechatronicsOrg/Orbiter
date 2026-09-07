"""Steger's ridge response: where a bright line runs, read from shape not level.

The scan finds the stripe by colour. Every pixel whose score clears
`laser.LaserParams.redness_min` (45) is lit, and where it crosses each
scanline comes from that scanline's score profile (`laser.stripe_centroids`).
That works, and on a narrow unsaturated profile it works very well — 0.026 to
0.037 px RMS for the Gaussian fit, 0.035 to 0.081 for the centroid it falls
back to. It has two blind spots, and both are about level rather than shape.

**A saturated stripe has no profile left.** Clipped at 255 the top is flat,
the Gaussian fit has nothing to fit, and the centroid is left guessing from
the flanks. **A dim stripe has no pixels at all.** On a dark or red-absorbing
surface the stripe never reaches 45, nothing is lit, and no scanline is even
offered to the estimator — the point is not measured badly, it is missing.

A ridge detector answers both, because it asks a different question: not how
bright is this pixel, but is this pixel on a crest. Across a bright line the
second derivative is large and negative; along it, nearly zero. So smooth the
score with a Gaussian of the stripe's own width, take the Hessian of the
smoothed image analytically — through Gaussian-derivative kernels, which is
exact rather than a difference of differences — and look at its eigenvalues.
The one of larger magnitude is the curvature across the line and its
eigenvector is the line's normal; −λ is then a ridge strength that is
positive on a bright line, negative on a dark one and zero on flat ground,
whatever the absolute brightness. Saturation flattens the top and leaves the
flanks, and the flanks are exactly what a second derivative measures; a dim
stripe has the same shape as a bright one, a hundred times smaller.

There is one tuning constant, σ, and it is not free: it is the stripe's own
width. σ = FWHM / 2.355 for a Gaussian, the live stripe measures FWHM ≈ 12 px,
so σ ≈ 5 px. Kernels are cut at 3σ, which is 31 taps.

Two implementations, one set of numbers. The numpy one needs nothing but
numpy and is always importable — this module is optional to everything else,
so `stripemask` has to be able to use it on a machine with no GPU at all. The
torch one runs the same separable kernels through `F.conv2d` on CUDA and is
what a live path would use. Measured here at 1920×1080, σ = 5: 0.45 s for the
numpy reference against 2.9 ms of GPU time on the RTX 5060 Ti — 1.9 ms for
the six separable passes and 1.0 ms for the eigenvalues — which is a factor
of 150 and the reason the reference is allowed to stay a reference. Both
correlate rather than convolve, and both mirror at the border the same way
(reflect-101), so they agree to float32 rounding: on a synthetic stripe the
responses differ by 3e-7 of full scale and the centres by 3e-6 px.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

#: The stripe's width as a Gaussian σ. The live stripe measures FWHM ≈ 12 px
#: and σ = FWHM / (2 √(2 ln 2)) = FWHM / 2.355, so 5 px. Tuning this away
#: from the stripe's own width is what costs a ridge detector its accuracy:
#: too small and it follows the speckle, too large and it smears the crest.
SIGMA_PX = 5.0

#: Kernels are cut at this many σ either side. Past 3σ a Gaussian derivative
#: carries under a thousandth of its weight, and every extra tap is another
#: full pass over the frame.
KERNEL_SIGMAS = 3.0

#: How far along the normal the Taylor vertex may land before the crest is
#: judged to belong to a different pixel. Steger's own rule is half a pixel;
#: a little past that keeps an honest centre sitting just outside the sample
#: it was found from, which a crest at exactly ±0.5 px legitimately does.
MAX_VERTEX_PX = 0.6

#: The smallest |normal| across the scanline for the crossing to be defined.
#: 0.3 admits a line up to 72° off the scanline's normal; the scan axis is
#: chosen so the stripe crosses it (`laser.StripePixels.along_x`), so this
#: only ever rejects a scanline the stripe runs along rather than across.
MIN_CROSS = 0.3


# ── the torch path, bound on first use ────────────────────────────────────

_torch = None
_F = None
_device = None
_state: bool | None = None
_why = "not probed yet"


def cuda_available() -> bool:
    """True when torch is installed and sees a CUDA device. Probed once; the
    reason for a False is kept, for the log and for the tests that skip."""
    global _state
    if _state is None:
        _state = _probe()
    return _state


def describe() -> str:
    """What `cuda_available()` found, in one line."""
    cuda_available()
    return _why


def _probe() -> bool:
    """Import torch and ask whether there is a device — and stop there.

    Deliberately no `get_device_name`, no allocation, nothing that would make
    torch initialise a CUDA context: `gpu._sleep_while_waiting` has to set
    `cudaSetDeviceFlags` BEFORE any context exists, and it is the detector
    threads that pay if it cannot. This module must never be the one that
    creates the context first.
    """
    global _torch, _F, _device, _why
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:
        _why = (f"numpy reference: {exc.name or exc} not installed"
                " — pip install -e ./native[gpu]")
        return False
    if not torch.cuda.is_available():
        _why = "numpy reference: torch has no CUDA device"
        return False
    _torch, _F = torch, F
    _device = torch.device("cuda", 0)
    _why = "torch on CUDA device 0"
    return True


# ── the kernels ───────────────────────────────────────────────────────────


@lru_cache(maxsize=4)
def _kernels(sigma: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Correlation kernels for (smooth, ∂, ∂²) at scale `sigma`, as float32.

    Cached, and never written to by anything downstream: they depend on σ
    alone, and building 31 taps for every frame would cost more arithmetic
    than one row of the frame it is about to filter.

    Written for CORRELATION — `out[i] = Σ_m k[m]·img[i + m]` — because that is
    what both the shift-and-add below and `F.conv2d` actually compute; what
    torch calls a convolution is a cross-correlation. Expanding `img` about
    `i` says what each kernel has to satisfy: Σk = 0, Σm·k = 1, Σm²·k = 0 for
    a first derivative, and Σk = 0, Σm·k = 0, Σm²·k = 2 for a second.

    The sampled Gaussian and its derivatives satisfy the ones that vanish by
    symmetry exactly. The other two they only satisfy in the limit, and
    cutting at 3σ costs the second derivative about 3 % of its scale, so both
    are fixed by hand: the mean is removed from `k2` so a flat image gives
    exactly zero curvature, and each derivative kernel is scaled to be exact
    on the polynomial it differentiates. The result carries real units —
    score per pixel and score per pixel² — which is what lets a caller put a
    threshold on the response rather than on a shape-dependent number.
    """
    r = max(1, int(np.ceil(KERNEL_SIGMAS * float(sigma))))
    m = np.arange(-r, r + 1, dtype=np.float64)
    g = np.exp(-0.5 * (m / sigma) ** 2)
    g /= g.sum()
    k1 = (m / sigma ** 2) * g                       # −G′, in correlation form
    k1 /= (m * k1).sum()                            # exact on a ramp
    k2 = ((m ** 2 - sigma ** 2) / sigma ** 4) * g   # +G″
    k2 -= k2.mean()                                 # exactly DC-free
    k2 *= 2.0 / (m ** 2 * k2).sum()                 # exact on a parabola
    return (g.astype(np.float32), k1.astype(np.float32), k2.astype(np.float32))


def _check_size(shape: tuple[int, int], radius: int) -> None:
    """Mirroring needs as many pixels as it reflects, on both axes."""
    if min(shape) <= radius:
        raise ValueError(
            f"a {2 * radius + 1}-tap kernel mirrors {radius} px into a frame,"
            f" so it needs more than {radius} px each way; got {shape[1]}×{shape[0]}")


# ── the response ──────────────────────────────────────────────────────────


def response(score2d, sigma: float = SIGMA_PX, *, use_gpu: bool | None = None):
    """The ridge strength and the line's normal at every pixel.

    `score2d` is a stripe score image — `laser.stripe_score` (uint8) or
    `gpu.stripe_score` (float32 on CUDA) — or anything else shaped like one.
    Returns `(ridge, nx, ny)`, all (H, W) float32. `ridge` is −λ, the
    curvature across the line: positive on a bright line against darker
    ground, negative on a dark one, zero where there is no shape at all.
    `(nx, ny)` is λ's unit eigenvector, the direction ACROSS the line. Which
    way along that direction is a free choice — a line has no side — so the
    larger of the two components is made positive, because a detector that
    answered ±n at random would make every downstream number unreproducible.

    Where the arithmetic runs is where the result lives: hand it a CUDA
    tensor and CUDA tensors come back, hand it a numpy array and numpy arrays
    do. `use_gpu` overrides that choice in both directions — True on a numpy
    frame uploads it and returns tensors, False on a tensor downloads it —
    and True without a CUDA device is an error rather than a silent
    downgrade, because the only reason to ask is to measure the two against
    each other.
    """
    if _wants_gpu(score2d, use_gpu):
        return _eigen(*_hessian_torch(score2d, _kernels_torch(float(sigma))), _torch)
    return _eigen(*_hessian_numpy(score2d, _kernels(float(sigma))), np)


def _wants_gpu(score2d, use_gpu: bool | None) -> bool:
    """Which path runs — and, when it is the torch one, bind it. A CUDA
    tensor arriving from elsewhere is proof torch works, but it is not the
    thing that fills this module's own handles."""
    if use_gpu is None:
        use_gpu = bool(getattr(score2d, "is_cuda", False))
    if use_gpu and not cuda_available():
        raise RuntimeError(f"the torch path was asked for and is not here — {describe()}")
    return bool(use_gpu)


def _hessian_numpy(score2d, kernels):
    """(r_xx, r_xy, r_yy) of the Gaussian-smoothed score, in numpy.

    Six separable passes rather than nine: the three passes along x are
    shared, because r_xx, r_xy and r_yy differ along x only in which of the
    three kernels they want there.
    """
    k0, k1, k2 = kernels
    img = _as_numpy(score2d)                # a CUDA frame with use_gpu=False
    if img.ndim != 2:
        raise ValueError(f"the ridge response wants one score plane; got shape {img.shape}")
    img = img.astype(np.float32, copy=False)
    _check_size(img.shape, len(k0) // 2)
    row0, row1, row2 = (_corr(img, k, axis=1) for k in (k0, k1, k2))
    return (_corr(row2, k0, axis=0),        # ∂²/∂x², smoothed along y
            _corr(row1, k1, axis=0),        # ∂²/∂x∂y
            _corr(row0, k2, axis=0))        # ∂²/∂y², smoothed along x


def _corr(img: np.ndarray, k: np.ndarray, axis: int) -> np.ndarray:
    """One separable pass: correlate `img` with `k` along `axis`, mirrored.

    `np.pad(mode="reflect")` is reflect-101 — the edge sample is not repeated
    — which is precisely what `F.pad(mode="reflect")` does, so the two
    implementations agree pixel for pixel including at the border.

    A tap at a time, rather than one stacked window and a `tensordot`: 31 taps
    over a 1080p frame is 31 fused adds over 2 M float32, while the stacked
    window would want a quarter of a gigabyte of view to do it in one call.
    The reference is allowed to be slow; it is not allowed to be fat.
    """
    r = len(k) // 2
    pad = [(0, 0), (0, 0)]
    pad[axis] = (r, r)
    padded = np.pad(img, pad, mode="reflect")
    n = img.shape[axis]
    out = np.zeros_like(img)
    for m, weight in enumerate(k):
        window = slice(m, m + n)
        out += weight * (padded[window] if axis == 0 else padded[:, window])
    return out


@lru_cache(maxsize=4)
def _kernels_torch(sigma: float):
    """`_kernels` living on the device. Cached for the same reason, and one
    reason more: three host-to-device copies per frame are three
    synchronisation points in a path whose whole point is not to have any."""
    return tuple(_torch.from_numpy(k).to(_device) for k in _kernels(sigma))


def _hessian_torch(score2d, kernels):
    """(r_xx, r_xy, r_yy) through `F.conv2d` on CUDA — the same six passes."""
    torch = _torch
    k0, k1, k2 = kernels
    x = score2d if torch.is_tensor(score2d) else torch.from_numpy(
        np.ascontiguousarray(np.asarray(score2d)))
    if x.ndim != 2:
        raise ValueError(f"the ridge response wants one score plane; got shape {tuple(x.shape)}")
    x = x.to(device=_device, dtype=torch.float32)
    _check_size(tuple(x.shape), k0.numel() // 2)
    rows = [_corr_torch(x[None, None], k, axis=1) for k in (k0, k1, k2)]
    return (_corr_torch(rows[2], k0, axis=0)[0, 0],
            _corr_torch(rows[1], k1, axis=0)[0, 0],
            _corr_torch(rows[0], k2, axis=0)[0, 0])


def _corr_torch(x, k, axis: int):
    """`_corr` for a (1, 1, H, W) tensor. `F.conv2d` correlates, so the
    kernels need no flip; `F.pad`'s reflect is numpy's reflect."""
    r = k.numel() // 2
    pad = (0, 0, r, r) if axis == 0 else (r, r, 0, 0)
    shape = (1, 1, -1, 1) if axis == 0 else (1, 1, 1, -1)
    return _F.conv2d(_F.pad(x, pad, mode="reflect"), k.view(shape))


def _eigen(rxx, rxy, ryy, xp):
    """−λ and λ's unit eigenvector, for the eigenvalue of larger |λ|.

    For a symmetric 2×2 the two eigenvalues are `t/2 ± h`, with `t` the trace
    and `h = √(((rxx − ryy)/2)² + rxy²)`. Since h ≥ 0, the one of larger
    magnitude is `t/2 + h` when t ≥ 0 and `t/2 − h` when t < 0 — one signed
    term, no branch and no sort. On a bright line the curvature across it is
    strongly negative and the one along it is near zero, so t < 0, λ is the
    negative eigenvalue, and −λ comes out positive. A dark line reverses
    every sign of that sentence and lands negative, which is how the response
    tells the two apart without ever thresholding the image.

    The eigenvector solves (H − λI)v = 0, which gives two proportional
    candidates, `(rxy, λ − rxx)` and `(λ − ryy, rxy)`. They degenerate at
    different times, so the longer one is taken. Both collapse only when H is
    a multiple of the identity — a round blob rather than a line, a specular
    dot say — where there is no direction across to find at all. That case
    gets an arbitrary (1, 0), and it is `centres` that then throws it out, by
    the same test that rejects a line running along its own scanline.

    `xp` is numpy or torch: every operation here exists under both names with
    the same meaning, so there is one copy of the algebra rather than two.
    """
    half = 0.5 * (rxx + ryy)
    h = xp.sqrt((0.5 * (rxx - ryy)) ** 2 + rxy ** 2)
    lam = half + xp.where(half >= 0, h, -h)

    v1x, v1y = rxy, lam - rxx
    v2x, v2y = lam - ryy, rxy
    n1 = v1x * v1x + v1y * v1y
    n2 = v2x * v2x + v2y * v2y
    longer = n1 >= n2
    vx = xp.where(longer, v1x, v2x)
    vy = xp.where(longer, v1y, v2y)
    norm = xp.sqrt(xp.where(longer, n1, n2))
    have = norm > 0
    safe = xp.where(have, norm, 1.0)
    nx = xp.where(have, vx / safe, 1.0)
    ny = xp.where(have, vy / safe, 0.0)

    # One side of the line, always the same side: an eigenvector's sign is
    # arbitrary and would otherwise flip with rounding from pixel to pixel.
    # The rule is "the larger component points positive", not "nx points
    # positive", because the boundary of any such rule is where it flips at
    # random — and a normal of (0, ±1) is a horizontal stripe, which is most
    # of them. This boundary sits at 45°, where `laser`'s own choice of scan
    # axis is equally undecided and no stripe should be sitting anyway.
    flip = xp.where(xp.abs(nx) >= xp.abs(ny), nx, ny) < 0
    return -lam, xp.where(flip, -nx, nx), xp.where(flip, -ny, ny)


# ── the sub-pixel centres ─────────────────────────────────────────────────


def centres(ridge, nx, ny, key_axis: str, *, min_response: float = 0.0):
    """Where the ridge crosses each scanline, to a fraction of a pixel.

    `key_axis` names the image axis that indexes the scanline, the same
    decision `laser.StripePixels.along_x` records: `"x"` for a stripe running
    mostly along x, where the scanlines are columns and the answer is a row,
    and `"y"` for one running mostly along y. The return is
    `laser.stripe_centroids`'s first two values and nothing else — `(scan,
    pos)`, float64, one entry per scanline that carries a crest, ascending —
    so a call site can hold one behind a toggle against the other.

    The centre is Steger's. Take the scanline's strongest pixel, expand the
    response to second order ALONG the normal, and the vertex

        t = (r₋ − r₊) / 2·(r₋ − 2·r₀ + r₊)

    is the crest, where r∓ are the response one pixel either side along that
    normal, sampled bilinearly. The crest point is p + t·n; what the scanline
    wants is where the line through it crosses the scanline, and that is

        pos = across + t / n_across

    because carrying the crest back along the tangent (−n_y, n_x) to the fixed
    key coordinate adds t·n_key²/n_across to t·n_across, and n is a unit
    vector so the two sum to t/n_across. It reduces to `across + t` when the
    line crosses the scanline square on, which is the usual case.

    A scanline is dropped when the parabola opens upward (no crest there),
    when the vertex lands further than `MAX_VERTEX_PX` from the pixel it was
    found from, or when the line runs too nearly along the scanline to cross
    it anywhere definite (`MIN_CROSS`). `min_response` is this detector's
    `redness_min`, in curvature rather than in level: zero — the default —
    asks only that the pixel be a bright ridge at all, which is precisely the
    dim stripe the colour threshold never lit.

    Works in numpy, and downloads a CUDA response to do it. What comes back is
    a handful of numbers per scanline that the scan consumes on the CPU, and
    one copy of this geometry is worth more than the transfer it saves.
    """
    r = _as_numpy(ridge)
    if key_axis == "x":
        view, n_key, n_acr = r.T, _as_numpy(nx).T, _as_numpy(ny).T
    elif key_axis == "y":
        view, n_key, n_acr = r, _as_numpy(ny), _as_numpy(nx)
    else:
        raise ValueError(f'key_axis is the scanline\'s axis, "x" or "y"; got {key_axis!r}')

    empty = np.empty(0, np.float64)
    if view.size == 0:
        return empty, empty
    lines = np.arange(view.shape[0])
    i = view.argmax(axis=1)
    peak = view[lines, i].astype(np.float64)
    key_n = n_key[lines, i].astype(np.float64)
    acr_n = n_acr[lines, i].astype(np.float64)

    # Back to image coordinates to step along the normal: for "x" the
    # scanline IS the column, so the pixel is (x, y) = (line, argmax).
    px, py = (lines, i) if key_axis == "x" else (i, lines)
    step_x, step_y = (key_n, acr_n) if key_axis == "x" else (acr_n, key_n)
    before = _sample(r, px - step_x, py - step_y)
    after = _sample(r, px + step_x, py + step_y)

    curve = before - 2.0 * peak + after
    crest = curve < 0.0
    t = np.divide(0.5 * (before - after), curve, out=np.zeros_like(curve), where=crest)
    keep = (crest & (peak > min_response) & (np.abs(t) <= MAX_VERTEX_PX)
            & (np.abs(acr_n) >= MIN_CROSS))
    return lines[keep].astype(np.float64), i[keep] + t[keep] / acr_n[keep]


def _sample(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Bilinear samples of `img` at float (x, y), clamped to the frame.

    Clamping rather than refusing: the only samples that reach the edge come
    from a crest sitting on it, which no real stripe does, and a crest that
    did would be dropped by the vertex bound anyway.
    """
    h, w = img.shape
    x = np.clip(np.asarray(x, np.float64), 0.0, w - 1.0)
    y = np.clip(np.asarray(y, np.float64), 0.0, h - 1.0)
    x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    fx, fy = x - x0, y - y0
    top = img[y0, x0] * (1.0 - fx) + img[y0, x1] * fx
    bottom = img[y1, x0] * (1.0 - fx) + img[y1, x1] * fx
    return top * (1.0 - fy) + bottom * fy


def _as_numpy(a) -> np.ndarray:
    """A torch tensor's values as numpy, anything else through `asarray`.
    Duck-typed on purpose: a machine with no torch installed must still be
    able to call `centres` on a numpy response."""
    return a.detach().cpu().numpy() if hasattr(a, "detach") else np.asarray(a)
