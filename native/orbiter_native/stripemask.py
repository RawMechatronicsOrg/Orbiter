"""The laser stripe, taken out of the photographs COLMAP is about to read.

A photograph taken while the laser is on carries a bright red line across
whatever it was looking at, and that line does not belong to the object. It
moves between frames, so every dense matcher sees a strong, confident,
*wrong* correspondence wherever it lies, and every texture atlas gets a red
smear baked into it.

**A mask alone does not fix it, and that is why this module inpaints as well.**
`stereo_fusion` takes `--StereoFusion.mask_path` and honours it, but
`patch_match_stereo` takes no mask at all: it computes a depth map for every
pixel, and the stripe has already contaminated every NCC window that touched
it by the time fusion gets a say. Masking is deleting the damage afterwards.
So the stripe is painted out of the pixels PatchMatch reads, and the mask is
kept as the second line — because inpainting invents texture, and invented
texture can match spuriously just as happily as a stripe can.

**The mask is three questions unioned, not one.** The frame's own detection
(`laser.find_stripe_pixels`) is the truth about where the scan *measured* the
stripe, but it ran at `LaserParams.redness_min` = 45 and only inside the band
the sheet could appear in. So it is joined by the frame's kept 3D points
projected back into that eye — the places the stripe demonstrably was, said
in millimetres rather than in pixels — and by a second detection pass at a
threshold the scan could never use: redness ≥ 15, plus the ridge response for
the crests a threshold breaks up. The scan cannot afford a low threshold
because a false positive there becomes a wrong 3D point. Here a false
positive costs a few masked pixels out of two million, and a false *negative*
costs a red streak in the atlas — so the trade runs the other way and the
mask is deliberately greedy.

**Everything is in the raw sensor frame**, the same frame the photograph, the
sidecar's stripe pixels and the stored intrinsics all live in. Undistortion
comes later and belongs to `recon.undistort_masks`, which undistorts the mask
cached here rather than rebuilding it against pixels that no longer exist.

**Polarity is COLMAP's: 0 = ignore, non-zero = use.** That is a verified fact
about the fusion mask, not a guess, and it is the opposite of what
`cv2.inpaint` wants — inpaint fills where the mask is non-zero — which is
exactly the inversion `build` performs between the two.

This module decides; `recon.clean_images` writes. The three writers below are
byte-level helpers so that the naming and the encoding choices live in one
place: lossless PNG for the `clean/` archive, quality-95 JPEG for
`colmap/images_clean/`. The JPEG is not a compromise made reluctantly —
`image_undistorter` re-encodes every image it copies, so a lossless
intermediate never survives into `dense/images` anyway, and these pixels were
invented by an inpainter in the first place.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import ridge
from .laser import StripePixels, stripe_score

#: COLMAP's fusion-mask convention, spelled out because the two halves read
#: identically as bare integers and getting them the wrong way round masks
#: the object and fuses the stripe.
MASK_IGNORE = 0
MASK_USE = 255

#: What `colmap/images_clean/` is encoded at. High enough that the inpaint,
#: not the encoder, is what the dense stage sees; low enough that a session's
#: worth of copies is not another gigabyte.
JPEG_QUALITY = 95


@dataclass(frozen=True)
class MaskParams:
    """Every threshold the mask and the inpaint depend on, in one place.

    The defaults are the plan's, and they are written into `session.json` with
    the rest of the run so that a changed number is visible rather than
    archaeological.
    """

    #: The second detection pass's floor on `laser.stripe_score`, well below
    #: the scan's own 45 (`laser.LaserParams.redness_min`). The scan cannot go
    #: this low — the bench and the board's own p99 redness live up here — but
    #: a mask can, because the cost of being wrong is a masked pixel rather
    #: than a 3D point in the wrong place.
    redness_min: int = 15
    #: Passed straight to `laser.stripe_score`: the side of the opening that
    #: estimates each channel's background. The same value the scan uses, so
    #: the two passes are the same measurement at two thresholds.
    background_px: int = 15
    #: Whether the ridge response joins the threshold pass. On by default: the
    #: numpy reference in `ridge` needs nothing but numpy, so this is a knob
    #: for measurement rather than a capability test.
    use_ridge: bool = True
    #: The stripe's own width as a Gaussian σ — `ridge`'s constant, not a
    #: second opinion about it.
    ridge_sigma: float = ridge.SIGMA_PX
    #: How strong a crest has to be to count. `ridge` returns the curvature
    #: across the line in score units, which for an ideal Gaussian stripe of
    #: FWHM 12 px is amplitude / 66.8; through a q95 JPEG, whose chroma
    #: subsampling widens the stripe before the score ever sees it, it is
    #: closer to amplitude / 320. So 0.1 is a stripe peaking near 30 — inside
    #: the gap between our floor and the scan's threshold, which is the whole
    #: point of the term. It is safe this low because `stripe_score` is
    #: REDNESS: a bright white edge, however sharp, has no crest in it.
    ridge_min: float = 0.1
    #: How much wider the masked band is made, in pixels — 2 × FWHM. Applied
    #: as an elliptical element of this diameter, so the region grows by half
    #: of it in every direction, which covers both the JPEG's 8×8 blocks and
    #: its 2×2 chroma subsampling: the stripe's red bleeds a little past where
    #: the detector can still see it. A full-diagonal stripe across a 1080p
    #: frame then costs ≈ 5 % of it, against the 8 % the plan budgets.
    dilate_px: int = 24
    #: `cv2.inpaint`'s radius — how far around each masked pixel it looks for
    #: something to extend inward.
    inpaint_px: int = 3


def score_image(bgr: np.ndarray, p: MaskParams = MaskParams()) -> np.ndarray:
    """How much each pixel looks like laser stripe, as `laser` measures it.

    Public because it is the expensive half and both the threshold pass and
    the ridge want the same numbers: a caller processing a session's worth of
    photographs computes this once per frame — on the GPU if it has one — and
    hands the result to `ridge.response` itself rather than paying for it
    twice.
    """
    return stripe_score(bgr, p.background_px)


def project_points(xyz_board: np.ndarray, R: np.ndarray, t_mm: np.ndarray,
                   K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """Board-frame points as pixels in one eye, (N, 2) float32.

    `R` and `t_mm` are the photograph's stored pose, board→camera
    (`x_cam = R x_board + t`), and `K`/`dist` are that eye's intrinsics in the
    raw sensor frame — the frame the mask is built in. Both the points and the
    translation are in millimetres; a projection does not care which unit, only
    that they agree.

    **Points behind the camera are dropped rather than projected.**
    `cv2.projectPoints` divides by z whatever its sign, so a point behind the
    lens comes back mirrored through the principal point and lands somewhere
    perfectly plausible in the image. The returned array is therefore shorter
    than the input by however many such points there were, which is what the
    caller wants: it is about to rasterise these, and a phantom is worse than
    a gap.
    """
    xyz = np.asarray(xyz_board, float).reshape(-1, 3)
    if not len(xyz):
        return np.empty((0, 2), np.float32)
    R = np.asarray(R, float).reshape(3, 3)
    t = np.asarray(t_mm, float).reshape(3)
    front = (xyz @ R.T + t)[:, 2] > 0.0
    if not front.any():
        return np.empty((0, 2), np.float32)
    uv, _ = cv2.projectPoints(xyz[front], cv2.Rodrigues(R)[0], t,
                              np.asarray(K, float).reshape(3, 3),
                              np.asarray(dist, float).reshape(-1))
    return uv.reshape(-1, 2).astype(np.float32)


def build(jpeg_bytes: bytes, stripe: StripePixels | None,
          projected_points_px: np.ndarray | None,
          params: MaskParams = MaskParams()) -> tuple[np.ndarray, np.ndarray]:
    """The fusion mask and the inpainted photograph, both in the raw frame.

    `jpeg_bytes` is the camera's own file, exactly as `photos.py` stored it.
    `stripe` is that frame's `StripePixels` out of the `stripe/*.npz` sidecar —
    for a right-eye photograph these are the time-shifted ones, which is what
    `stripe_shifted` records and why the sidecar carries them at all. `None`
    for either that and `projected_points_px` simply drops that term; a
    photograph from a clean pass has neither and gets a mask built from the
    second detection pass alone.

    `projected_points_px` is the frame's `kept_xyz_board` already in pixels —
    `project_points` above is the way to get there, and it lives here so that
    the two callers in `recon` do not each invent their own.

    Returns `(mask, clean)`: `mask` is uint8, `MASK_IGNORE` on the stripe and
    `MASK_USE` everywhere else, the same size as the photograph; `clean` is
    the BGR photograph with the masked band painted over. Nothing is written —
    `recon.clean_images` owns the paths.
    """
    bgr = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("this is not a decodable JPEG")
    h, w = bgr.shape[:2]
    lit = np.zeros((h, w), np.uint8)

    if stripe is not None and stripe.count:
        # The sidecar was written against one particular frame size, and the
        # intrinsics were solved at that size too. A mismatch is a resolution
        # change between capture and reconstruct, which is exactly the case
        # `Eye.intrinsics_for` refuses live rather than silently rescaling.
        sidecar_wh = tuple(int(v) for v in stripe.wh)
        if sidecar_wh != (w, h):
            raise ValueError(f"the sidecar's stripe is {sidecar_wh} but the "
                             f"photograph is {(w, h)}")
        lit[stripe.y, stripe.x] = 255

    if projected_points_px is not None:
        uv = np.asarray(projected_points_px, float).reshape(-1, 2)
        if len(uv):
            xy = np.rint(uv).astype(np.int64)
            inside = ((xy[:, 0] >= 0) & (xy[:, 0] < w)
                      & (xy[:, 1] >= 0) & (xy[:, 1] < h))
            # Single pixels: the dilation below grows them into discs wider
            # than any reprojection error this rig can produce.
            lit[xy[inside, 1], xy[inside, 0]] = 255

    score = score_image(bgr, params)
    lit[score >= params.redness_min] = 255
    if params.use_ridge:
        crest, _, _ = ridge.response(score.astype(np.float32), params.ridge_sigma)
        lit[crest >= params.ridge_min] = 255

    # An odd disc, so the growth is the same in every direction and a diagonal
    # stripe costs no more than a horizontal one — a square element would
    # widen a 45° stripe by a factor of √2.
    side = 2 * (int(params.dilate_px) // 2) + 1
    ignore = cv2.dilate(lit, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                       (side, side)))
    # `ignore` is already the inverted mask cv2.inpaint asks for — non-zero
    # where it should paint — so the inversion happens once, here, and the two
    # conventions never have to be reconciled again downstream.
    clean = cv2.inpaint(bgr, ignore, params.inpaint_px, cv2.INPAINT_TELEA)
    mask = np.where(ignore > 0, MASK_IGNORE, MASK_USE).astype(np.uint8)
    return mask, clean


def write_mask(path: str | Path, mask: np.ndarray) -> None:
    """The mask as a PNG, at `<NAME>.png` — the photograph's full name with the
    extension appended, so `left_0007.jpg.png`. COLMAP looks a mask up by
    exactly that string, which is why no basename is ever stripped anywhere in
    this chain."""
    _write_png(path, mask)


def write_clean(path: str | Path, bgr: np.ndarray) -> None:
    """The inpainted photograph as a lossless PNG, for `clean/`.

    This copy is the archive: it is the only place the inpainted pixels exist
    without a second generation of JPEG on top of them, and it is what an
    operator opens to judge whether the inpaint did something sensible.
    """
    _write_png(path, bgr)


def encode_jpeg(bgr: np.ndarray, quality: int = JPEG_QUALITY) -> bytes:
    """The inpainted photograph as JPEG bytes, for `colmap/images_clean/`."""
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise ValueError("OpenCV would not encode this as JPEG")
    return buf.tobytes()


def _write_png(path: str | Path, img: np.ndarray) -> None:
    """Encode, then write the bytes ourselves.

    `cv2.imwrite` goes through OpenCV's own filename handling, which on
    Windows is not unicode-safe; the session root is under the user's home
    directory and that name is not ours to choose.
    """
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ValueError("OpenCV would not encode this as PNG")
    Path(path).write_bytes(buf.tobytes())
