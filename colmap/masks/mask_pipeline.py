"""Mask post-processing tail shared by the SAM2 and geometric methods.

The segmentation (SAM2 or the geometric silhouette) produces a raw boolean
foreground; this module turns it into a COLMAP-ready mask:

    raw -> keep components >= min-area (union of ALL of them — the rotating
    assembly may show as several blobs) -> fill holes -> dilate -> 0/255

The output follows COLMAP's mask convention: white (255) where features are
allowed (the rotating assembly), black (0) where they are suppressed (the
static room). COLMAP reads it from ``<mask_path>/<image_name>.png``.
"""

from __future__ import annotations

import cv2
import numpy as np


def dilate_disk(mask: np.ndarray, radius: int) -> np.ndarray:
    """Binary dilation by a disk of ``radius`` px, via a distance transform —
    O(pixels), independent of the radius.

    ``cv2.dilate`` with an ellipse structuring element is O(pixels · radius²);
    on a 12 MP frame a ~300 px kernel costs ~1.8 s per call and dominated the
    whole mask runtime. The distance transform yields the identical disk
    dilation — every pixel within ``radius`` of a set pixel — in a few ms.
    Input/output follow the 0/255 uint8 convention used here.
    """
    if radius <= 0:
        return mask.copy()
    # distanceTransform measures, per non-zero pixel, the distance to the
    # nearest zero pixel. Feed the INVERSE so the result is "distance to the
    # nearest foreground"; pixels within radius are the dilation.
    bg = (mask == 0).astype(np.uint8)
    dist = cv2.distanceTransform(bg, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return (dist <= radius).astype(np.uint8) * 255


def erode_disk(mask: np.ndarray, radius: int) -> np.ndarray:
    """Binary erosion by a disk of ``radius`` px — the distance-transform dual
    of :func:`dilate_disk` (keeps foreground farther than ``radius`` from any
    background pixel)."""
    if radius <= 0:
        return mask.copy()
    fg = (mask > 0).astype(np.uint8)
    dist = cv2.distanceTransform(fg, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return (dist > radius).astype(np.uint8) * 255


def union_components(mask: np.ndarray, min_area_px: int) -> np.ndarray:
    """Union of every connected component with area >= ``min_area_px``.

    Unlike a single-central-component pick, this keeps multi-blob foregrounds
    (object + board + table edge can disconnect at some elevations); only
    sub-threshold speckle is dropped. Returns a 0/255 mask.
    """
    n, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, 8)
    out = np.zeros_like(mask)
    for label in range(1, n):  # label 0 is the background
        if int(stats[label, cv2.CC_STAT_AREA]) >= min_area_px:
            out[labels == label] = 255
    return out


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill interior holes (specular highlights, dark cavities) so the
    foreground is solid.

    Flood the exterior background and invert: whatever the flood cannot reach
    is interior. The mask is padded with a 1px black border first so every
    border-adjacent background region stays connected to the seed — a
    foreground that spans the frame would otherwise split the background and
    get the cut-off part "filled" as a giant fake hole.
    """
    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    flood = padded.copy()
    h, w = padded.shape
    ff = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, ff, (0, 0), 255)
    holes = cv2.bitwise_not(flood)[1:-1, 1:-1]
    return cv2.bitwise_or(mask, holes)


def finalize_mask(
    raw: np.ndarray,
    *,
    min_area_frac: float,
    dilate_px: int,
) -> tuple[np.ndarray, dict]:
    """Run the post-processing tail on a raw boolean/0-255 foreground.

    Returns ``(mask, stats)`` — ``mask`` 0/255 uint8; ``stats`` carries
    ``raw_area_px``, ``final_area_px``, ``coverage`` and ``empty_after_clean``
    (True when nothing met ``min_area_frac`` — the caller fails open).
    """
    h, w = raw.shape[:2]
    m = (np.asarray(raw) > 0).astype(np.uint8) * 255
    raw_area = int(np.count_nonzero(m))

    min_area_px = int(min_area_frac * h * w)
    comp = union_components(m, min_area_px)
    filled = _fill_holes(comp)

    if dilate_px > 0:
        dk = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
        final = cv2.dilate(filled, dk)
    else:
        final = filled

    final_area = int(np.count_nonzero(final))
    stats = {
        "raw_area_px": raw_area,
        "final_area_px": final_area,
        "coverage": round(final_area / float(h * w), 4),
        "empty_after_clean": bool(np.count_nonzero(comp) == 0),
    }
    return final, stats


def make_overlay(obj_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Debug overlay: foreground in true colour, suppressed region dimmed and
    red-tinted, outline drawn in green — so an operator can eyeball mask
    quality at a glance."""
    out = obj_bgr.copy()
    bg = mask == 0
    out[bg] = (0.35 * out[bg] + np.array([0.0, 0.0, 90.0])).astype(np.uint8)
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (0, 255, 0), 2)
    return out
