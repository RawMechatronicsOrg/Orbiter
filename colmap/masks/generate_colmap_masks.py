"""generate_colmap_masks.py — COLMAP masks for a turntable shoot via SAM 2.1.

Keeps everything that ROTATES WITH THE TABLE (object + ChArUco board + the
turntable itself) and suppresses the static room. In turntable photogrammetry
the static background gives COLMAP perfectly-matching correspondences across
frames, so the solver decides the camera never moved — masking the room (not
the object!) is what prevents the degenerate "reconstructed the room"
solution. The board/table are kept deliberately: they rotate with the object
and give texture for registration.

Per frame the calibrated rig geometry provides the prompt: the turntable
working volume (a cylinder) is projected through the frame's world->camera
pose from ``sfm_priors.json`` into a box + positive/negative points; SAM 2.1
turns that into a pixel-accurate foreground. The projected silhouette also
serves as the no-ML fallback (``--method geometric``) and as a clamp so SAM2
can never leak into the room.

Output — COLMAP-ready masks plus human-checkable overlays::

    <masks>/   0001.jpg.png ...           (white = keep, black = suppressed)
    <preview>/ 0001_overlay.jpg ...       (mask painted over the frame)

Subdirectories under ``--images`` are mirrored into ``--masks``/``--preview``.
Then point COLMAP at the masks::

    colmap feature_extractor ... --ImageReader.mask_path <masks>

Return codes:
    0 — success (>=1 frame with a foreground isolated)
    1 — bad input (missing dirs/poses, unusable SAM2 setup)
    3 — produced nothing usable (every frame degenerate)
    4 — unhandled exception (traceback on stderr)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Force UTF-8 stdout — progress lines contain arrows that crash Windows cp1252.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        pass

# Sibling imports — the pipeline modules live next to this file. Resolving
# __file__ first means this works when invoked by absolute path or via a
# symlink (e.g. a wrapper on PATH), not just from the tool's own directory.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import mask_pipeline as mp  # noqa: E402
from geom_prompt import GeomPrompter, PriorsError, Prompt  # noqa: E402
from profiling import PhaseProfiler  # noqa: E402
from sam2_masker import Sam2Masker, Sam2Unavailable  # noqa: E402

# One profiler for the process (this is a single-run CLI). Phases accumulate
# across frames; ``PROF.frame()`` additionally records a per-frame breakdown
# onto each manifest row so a single slow frame is locatable.
PROF = PhaseProfiler()

_TOOL_VERSION = "2.2"
_IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
# Every tunable below is also a CLI flag — these are the defaults.
#
# Clamp growth: the geometric silhouette dilated by this fraction of
# min(W, H) is the hard "could be assembly" boundary for SAM2 output.
_CLAMP_GROW_FRAC = 0.05
# SAM2 inference resolution (long side). The model embeds at 1024 internally;
# feeding 4000-px phone frames only costs time and the masks come back
# upsampled anyway. The final mask is resized back to full resolution.
_SAM_LONG_SIDE = 1024
# A probe mask joins the assembly union when at least this fraction of it
# lies inside the grown silhouette — room segments (walls, lamp, window)
# always spill far outside the projected volume.
_CONTAINMENT_MIN = 0.8
# ... and when it is not frame-sized (a wall segment can sit fully inside a
# frame-filling silhouette)...
_PROBE_MAX_FRACTION = 0.35
# ... and when it does not touch the top edge of the frame: the camera always
# looks DOWN at the table, so the rotating assembly never extends past the
# frame top — anything that does is the room behind it.
_TOP_EDGE_BAND_FRAC = 0.01
# Probe grid density (N×N over the projected bbox) and the negative-ring
# radius multiplier — forwarded to GeomPrompter.
_GRID_STEPS = 8
_NEG_RADIUS_MULT = 2.0
# A growth-pass segment joins the assembly union when at least this fraction
# of it overlaps the dilated union — any-touch let furniture standing next to
# the disc edge chain itself in.
_CONNECT_OVERLAP_MIN = 0.10
# Physical turntable disc radius (mm) — the disc (and the nominal board) are
# stamped into the mask deterministically; they rotate with the table by
# construction, so SAM2 gets no vote on them.
_DISC_RADIUS_MM = 180.0


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _quality_summary(rows: list[dict]) -> dict:
    """Cross-frame quality numbers for the manifest.

    Coverage varies smoothly over a turntable sweep — a spike pinpoints
    exactly the frame to eyeball in the preview dir. Outliers = frames more
    than max(0.05, 3·MAD) away from the median coverage. ``sam_disc_recall``
    (pre-stamp) tracks how healthy the probe pass is on the disc itself.
    """
    cov = [(r["image"], float(r["coverage"])) for r in rows
           if not r.get("degenerate") and r.get("coverage") is not None]
    out: dict = {"coverage_median": None, "coverage_outliers": []}
    if cov:
        vals = np.asarray([c for _, c in cov], dtype=np.float64)
        med = float(np.median(vals))
        mad = float(np.median(np.abs(vals - med)))
        thr = max(0.05, 3.0 * mad)
        out["coverage_median"] = round(med, 4)
        out["coverage_outliers"] = [
            name for name, c in cov if abs(c - med) > thr
        ]
    recalls = [r["sam_disc_recall"] for r in rows
               if r.get("sam_disc_recall") is not None]
    out["mean_sam_disc_recall"] = (
        round(float(np.mean(recalls)), 4) if recalls else None
    )
    return out


def _discover_images(root: Path) -> list[tuple[Path, Path]]:
    """Sorted ``[(relpath, abspath)]`` of images under ``root`` (recursive)."""
    out: list[tuple[Path, Path]] = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in _IMAGE_EXT:
            out.append((p.relative_to(root), p))
    return out


def _clamp_hull(silhouette: np.ndarray, grow_frac: float) -> np.ndarray:
    """Dilated silhouette used to cap SAM2 output (never grab the room).

    ``grow`` is a large fraction of the (12 MP) frame, so a literal ellipse
    dilation cost ~1.8 s/call — the dominant runtime. ``mp.dilate_disk`` is the
    same disk dilation via a distance transform, O(pixels)."""
    h, w = silhouette.shape
    grow = max(1, int(grow_frac * min(h, w)))
    return mp.dilate_disk(silhouette, grow)


def _sam2_foreground(
    masker: Sam2Masker,
    bgr: np.ndarray,
    prompt: Prompt,
    tune,
) -> tuple[np.ndarray, float]:
    """SAM2 mask for one frame: the union of contained probe segments.

    SAM2 is at its best with a single positive point and at its worst with a
    frame-sized box plus a scatter of mixed points (it returns texture-noise
    masks on busy scenes). So instead of one big query, every probe-grid
    point inside the projected volume runs as an independent single-point
    query, and a probe's segment joins the assembly when it stays mostly
    inside the (grown) silhouette — room segments always spill far outside
    the projected volume. The union is the rotating assembly: object, board,
    turntable — whatever the probes hit.

    Inference runs at ``tune.sam_long_side`` (model-native 1024); the union
    is resized back to full resolution. Returned score = mean of accepted
    probes. ``tune`` is the parsed argparse namespace — every filter
    threshold is a CLI flag.
    """
    h, w = bgr.shape[:2]
    s = min(1.0, tune.sam_long_side / max(w, h))
    sw, sh = int(round(w * s)), int(round(h * s))
    small = cv2.resize(bgr, (sw, sh), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    with PROF.phase("encode"):
        masker.set_frame(rgb)

    if prompt.silhouette is None or prompt.grid_xy.shape[0] == 0:
        # Heuristic prompt — one centre query is all the geometry we have.
        with PROF.phase("decode"):
            fg_s, score = masker.query_box(
                prompt.box_xyxy * s, prompt.pos_xy * s, prompt.neg_xy * s,
            )
        fg = cv2.resize(fg_s.astype(np.uint8), (w, h),
                        interpolation=cv2.INTER_NEAREST).astype(bool)
        return fg, score

    clamp_s = cv2.resize(_clamp_hull(prompt.silhouette, tune.clamp_grow_frac),
                         (sw, sh), interpolation=cv2.INTER_NEAREST) > 0
    top_band = max(1, int(tune.top_edge_band_frac * sh))

    def _passes(hyps) -> tuple[np.ndarray, float] | None:
        # Walk the multimask hypotheses LARGEST first and keep the first one
        # that passes the filters: argmax(score) prefers a crisp *part* on
        # multi-coloured objects, leaving the rest of the object unmasked.
        for m, score in hyps:
            area = int(np.count_nonzero(m))
            if area == 0 or area > tune.probe_max_fraction * sw * sh:
                continue
            if np.any(m[:top_band, :]):
                continue  # reaches the frame top — room, not assembly
            if np.count_nonzero(m & clamp_s) / area < tune.containment_min:
                continue
            return m, score
        return None

    # Decode EVERY probe point in one batched decoder pass against the frame
    # embedding (the encoder already ran in set_frame). Seed points (on the
    # projected disc) come first, then the growth candidates (volume axis +
    # grid) in their original order, so the result list splits cleanly at
    # n_seed. This replaces dozens of per-point predict() calls — one GPU sync
    # each — with a handful of batched decodes.
    seed_pts = np.asarray(prompt.extra_pos_xy, np.float32).reshape(-1, 2)
    grow_arrs = [np.asarray(a, np.float32).reshape(-1, 2)
                 for a in (prompt.pos_xy, prompt.grid_xy)
                 if np.asarray(a).size]
    grow_pts = np.vstack(grow_arrs) if grow_arrs else np.zeros((0, 2), np.float32)
    n_seed = seed_pts.shape[0]
    all_pts = np.vstack([seed_pts, grow_pts])
    with PROF.phase("decode"):
        hyps_all = masker.query_points_batched(all_pts * s)
    seed_hyps, grow_hyps = hyps_all[:n_seed], hyps_all[n_seed:]

    # Pass 1 — seed: probes ON the projected turntable disc. These points are
    # physically the rotating table, so their segments anchor the assembly.
    union = np.zeros((sh, sw), bool)
    scores: list[float] = []
    for hyps in seed_hyps:
        hit = _passes(hyps)
        if hit is not None:
            union |= hit[0]
            scores.append(hit[1])

    # Pass 2 — growth: axis + grid probes join only when their segment
    # touches the (dilated) assembly so far. Everything on the table connects
    # to it; a lamp/shelf that merely PROJECTS inside the volume doesn't.
    # With no seed (disc fully occluded) fall back to plain containment.
    grow = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    accepted = [hit for hit in (_passes(hy) for hy in grow_hyps)
                if hit is not None]
    # Iterate: keep attaching touching segments until nothing changes, so a
    # tall object chains up from the disc through its own parts.
    pending = accepted
    for _ in range(3):
        if not pending:
            break
        seed_grown = (
            cv2.dilate(union.astype(np.uint8), grow) > 0
            if union.any() else None
        )
        still: list[tuple[np.ndarray, float]] = []
        changed = False
        for m, score in pending:
            # Overlap FRACTION, not any-touch: a lamp standing next to the
            # disc edge touches the dilated union with a sliver of itself —
            # an on-table segment overlaps it substantially.
            joined = seed_grown is None or (
                np.count_nonzero(m & seed_grown)
                >= tune.connect_overlap_min * np.count_nonzero(m)
            )
            if joined:
                union |= m
                scores.append(score)
                changed = True
            else:
                still.append((m, score))
        pending = still
        if not changed:
            break

    union &= clamp_s
    fg = cv2.resize(union.astype(np.uint8), (w, h),
                    interpolation=cv2.INTER_NEAREST).astype(bool)
    return fg, (float(np.mean(scores)) if scores else 0.0)


def _disc_recall(fg: np.ndarray, prompt: Prompt) -> float | None:
    """SAM2's recall of the projected disc BEFORE stamping — a health metric
    for the probe pass (post-stamp the disc is 1.0 by construction)."""
    if prompt.disc_mask is None:
        return None
    disc_area = int(np.count_nonzero(prompt.disc_mask))
    if not disc_area:
        return None
    return round(float(np.count_nonzero(fg & prompt.disc_mask)) / disc_area, 4)


def _emit_frame(
    rel: Path,
    bgr: np.ndarray,
    prompt: Prompt,
    fg: np.ndarray,
    tune,
    masks_dir: Path,
    preview_dir: Path | None,
    *,
    i: int,
    n: int,
    sam_score: float | None = None,
    sam_disc_recall: float | None = None,
    motion_coverage: float | None = None,
    extra_log: str = "",
) -> tuple[dict, bool]:
    """Shared finalize tail for every method.

    Deterministic composite (∪ disc/board stamp), post-process, clamp back to
    the grown silhouette (under-mask the room), fail open to WHITE if empty,
    then write the mask + preview and build the manifest row. The COLMAP-facing
    semantics are identical no matter how ``fg`` was found. Returns
    ``(row, degenerate)``.
    """
    h, w = bgr.shape[:2]

    with PROF.phase("post"):
        if tune.geometry_composite and prompt.stamp is not None:
            fg = fg | (prompt.stamp > 0)

        mask, mstats = mp.finalize_mask(
            fg, min_area_frac=tune.min_area, dilate_px=tune.dilate)

        # White pixels readmit static-room features to COLMAP — cut the mask
        # back to the grown silhouette, then re-assert the exact geometry stamps.
        if prompt.silhouette is not None:
            clamp_full = _clamp_hull(prompt.silhouette, tune.clamp_grow_frac)
            mask = cv2.bitwise_and(mask, clamp_full)
            if tune.geometry_composite and prompt.stamp is not None:
                mask = cv2.bitwise_or(mask, prompt.stamp)
            mstats["coverage"] = round(
                int(np.count_nonzero(mask)) / float(h * w), 4)

    sil_area = (int(np.count_nonzero(prompt.silhouette))
                if prompt.silhouette is not None else 0)
    mask_to_sil = (round(int(np.count_nonzero(mask)) / sil_area, 4)
                   if sil_area else None)

    degenerate = mstats["empty_after_clean"]
    if degenerate:
        # Fail OPEN: an all-black mask deletes this frame from SfM. All-white =
        # "no masking here" — strictly safer than losing the image.
        mask = np.full((h, w), 255, np.uint8)
        print(f"  [{i + 1}/{n}] {rel}: empty foreground -> WHITE mask "
              f"(frame kept; check params)", file=sys.stderr)

    with PROF.phase("write"):
        mask_path = masks_dir / (str(rel) + ".png")
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(mask_path), mask)

        if preview_dir is not None:
            overlay = mp.make_overlay(bgr, mask)
            prev_path = preview_dir / rel.parent / f"{rel.stem}_overlay.jpg"
            prev_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(prev_path), overlay, [cv2.IMWRITE_JPEG_QUALITY, 90])

    row = {
        "image": rel.as_posix(),
        "coverage": mstats["coverage"],
        "mask_to_silhouette_ratio": mask_to_sil,
        "sam_disc_recall": sam_disc_recall,
        "sam_score": None if sam_score is None else round(sam_score, 4),
        "motion_coverage": motion_coverage,
        "prompt": prompt.kind,
        "degenerate": degenerate,
    }
    if not degenerate:
        extras = ""
        if mask_to_sil is not None:
            extras += f" m/sil={mask_to_sil:.2f}"
        if sam_disc_recall is not None:
            extras += f" disc_recall={sam_disc_recall:.2f}"
        extras += extra_log
        print(f"  [{i + 1}/{n}] {rel}: prompt={prompt.kind} "
              f"coverage={mstats['coverage'] * 100:.1f}%"
              + (f" score={sam_score:.3f}" if sam_score is not None else "")
              + extras)
    return row, degenerate


def _run_perframe(
    tune,
    prompter: GeomPrompter,
    masker: Sam2Masker | None,
    frames: list[tuple[Path, Path]],
    n: int,
    masks_dir: Path,
    preview_dir: Path | None,
) -> tuple[list[dict], int, int, int]:
    """SAM2 probe-grid (or the geometric silhouette) per image — the original
    per-frame path. Returns ``(rows, n_written, n_degenerate, n_heuristic)``."""
    rows: list[dict] = []
    n_written = n_degenerate = n_heuristic = 0
    for i, (rel, abspath) in enumerate(frames):
        with PROF.frame() as ftime:
            with PROF.phase("read"):
                bgr = cv2.imread(str(abspath), cv2.IMREAD_COLOR)
            if bgr is None:
                print(f"  [{i + 1}/{n}] {rel}: UNREADABLE", file=sys.stderr)
                rows.append({"image": rel.as_posix(), "error": "unreadable"})
                continue
            h, w = bgr.shape[:2]
            with PROF.phase("prompt"):
                prompt = prompter.prompt_for(rel.as_posix(), w, h)
            if prompt.kind == "heuristic":
                n_heuristic += 1

            sam_score: float | None = None
            if masker is not None:
                fg, sam_score = _sam2_foreground(masker, bgr, prompt, tune)
            elif prompt.silhouette is not None:
                fg = prompt.silhouette > 0
            else:
                print(f"  [{i + 1}/{n}] {rel}: no geometry for geometric method",
                      file=sys.stderr)
                rows.append({"image": rel.as_posix(), "error": "no geometry"})
                continue

            row, degenerate = _emit_frame(
                rel, bgr, prompt, fg, tune, masks_dir, preview_dir,
                i=i, n=n, sam_score=sam_score,
                sam_disc_recall=_disc_recall(fg, prompt))
        row["t"] = dict(ftime)
        rows.append(row)
        n_written += 1
        if degenerate:
            n_degenerate += 1
    return rows, n_written, n_degenerate, n_heuristic


def _sample_points(mask: np.ndarray, n: int) -> np.ndarray:
    """Up to ``n`` evenly-spread ``(x, y)`` float32 points from a boolean mask."""
    ys, xs = np.where(mask)
    if xs.size == 0:
        return np.zeros((0, 2), np.float32)
    idx = (np.arange(xs.size) if xs.size <= n
           else np.linspace(0, xs.size - 1, n).astype(int))
    return np.stack([xs[idx], ys[idx]], axis=1).astype(np.float32)


def _edge_refine(
    masker: Sam2Masker, bgr: np.ndarray, motion_fg: np.ndarray, tune,
) -> tuple[np.ndarray, float | None]:
    """Snap the motion mask's boundary with SAM2 — BOUNDED so it can only
    refine edges, never re-segment.

    Positives sample the eroded motion interior (reliably the object),
    negatives a ring just outside the motion mask (the static plate). SAM2's
    output is kept only within ``edge_band_frac`` of the motion boundary and
    unioned with a guaranteed interior core: it can shave or grow the edge by a
    band, but cannot delete the object or run into the room (and the downstream
    clamp + disc stamp in ``_emit_frame`` still apply). Returns
    ``(refined_fg, score)``.
    """
    h, w = bgr.shape[:2]
    s = min(1.0, tune.sam_long_side / max(w, h))
    sw, sh = int(round(w * s)), int(round(h * s))
    small = cv2.resize(bgr, (sw, sh), interpolation=cv2.INTER_AREA)
    m = cv2.resize(motion_fg.astype(np.uint8), (sw, sh),
                   interpolation=cv2.INTER_NEAREST)
    if not m.any():
        return motion_fg, None

    band_px = max(3, int(tune.edge_band_frac * min(sw, sh)))
    core_px = max(2, band_px // 2)
    core = mp.erode_disk(m * 255, core_px) > 0
    band = mp.dilate_disk(m * 255, band_px) > 0
    ring = (mp.dilate_disk(m * 255, 2 * band_px) > 0) & ~band

    pos = _sample_points(core if core.any() else (m > 0), 16)
    neg = _sample_points(ring if ring.any() else ~band, 10)
    if pos.shape[0] == 0:
        return motion_fg, None

    ys, xs = np.where(m > 0)
    box = np.array([xs.min() - band_px, ys.min() - band_px,
                    xs.max() + band_px, ys.max() + band_px], np.float32)
    box = np.clip(box, [0, 0, 0, 0], [sw - 1, sh - 1, sw - 1, sh - 1])

    with PROF.phase("encode"):
        masker.set_frame(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
    with PROF.phase("decode"):
        sam_s, score = masker.query_box(box, pos, neg)

    refined_s = (sam_s & band) | core
    refined = cv2.resize(refined_s.astype(np.uint8), (w, h),
                         interpolation=cv2.INTER_NEAREST).astype(bool)
    return refined, float(score)


def _run_motion(
    tune,
    prompter: GeomPrompter,
    masker: Sam2Masker | None,
    frames: list[tuple[Path, Path]],
    n: int,
    masks_dir: Path,
    preview_dir: Path | None,
) -> tuple[list[dict], int, int, int, list[dict]]:
    """Per-EL-ring temporal background subtraction.

    Frames of one elevation ring are shot by a physically static camera (AZ
    rotates the platform), so a per-pixel median over the ring reconstructs the
    empty scene and ``|frame − plate|`` is the moving assembly. The shared
    ``_emit_frame`` tail then composites the deterministic disc/board, clamps to
    the working volume and writes — identical COLMAP semantics to the SAM2 path.
    Returns ``(rows, n_written, n_degenerate, n_heuristic, ring_stats)``.
    """
    import motion_plate as motp

    rings = motp.group_rings(frames)
    print("motion: " + str(len(rings)) + " elevation ring(s) -> "
          + ", ".join(f"{k}({len(v)})" for k, v in rings.items()))

    rows_by_idx: dict[int, dict] = {}
    n_written = n_degenerate = n_heuristic = 0
    ring_stats: list[dict] = []

    for key, idxs in rings.items():
        # Build the plate from the ring's downscaled frames; drop unreadable
        # ones (they still get an error row and are skipped per-frame below).
        smalls: list[np.ndarray] = []
        ok_idxs: list[int] = []
        for i in idxs:
            try:
                with PROF.phase("read"):
                    small, _s, _hw = motp.load_small(
                        frames[i][1], tune.plate_long_side)
            except OSError:
                print(f"  {frames[i][0]}: UNREADABLE", file=sys.stderr)
                rows_by_idx[i] = {"image": frames[i][0].as_posix(),
                                  "error": "unreadable"}
                continue
            smalls.append(small)
            ok_idxs.append(i)

        with PROF.phase("plate"):
            plate = motp.build_plate(smalls) if smalls else None
            resid = ([motp.colour_residual(s, plate) for s in smalls]
                     if plate is not None else [])
            sigma = motp.estimate_noise(resid)
            # Robust passes: rebuild the plate ignoring first-pass motion pixels
            # to remove the on-axis ghost, then recompute residuals / noise sigma.
            for _ in range(max(0, tune.plate_iters - 1)):
                if plate is None:
                    break
                masks0 = [motp.motion_mask_small(
                    r, sigma, k=tune.motion_k, floor=tune.motion_floor,
                    open_px=tune.motion_open_px, close_px=tune.motion_close_px)
                    for r in resid]
                plate = motp.build_plate_robust(smalls, masks0, plate)
                resid = [motp.colour_residual(s, plate) for s in smalls]
                sigma = motp.estimate_noise(resid)

        thr = max(tune.motion_floor, tune.motion_k * sigma)
        ring_stats.append({"ring": key, "n_frames": len(idxs),
                           "n_usable": len(smalls), "sigma": round(sigma, 3),
                            "threshold": round(thr, 2)})
        print(f"  ring {key}: {len(smalls)}/{len(idxs)} frames  "
              f"sigma={sigma:.2f}  threshold={thr:.1f}")

        for j, i in enumerate(ok_idxs):
            rel, abspath = frames[i]
            with PROF.frame() as ftime:
                with PROF.phase("read"):
                    bgr = cv2.imread(str(abspath), cv2.IMREAD_COLOR)
                if bgr is None:
                    print(f"  {rel}: UNREADABLE", file=sys.stderr)
                    rows_by_idx[i] = {"image": rel.as_posix(),
                                      "error": "unreadable"}
                    continue
                h, w = bgr.shape[:2]
                with PROF.phase("prompt"):
                    prompt = prompter.prompt_for(rel.as_posix(), w, h)
                if prompt.kind == "heuristic":
                    n_heuristic += 1

                with PROF.phase("motion_mask"):
                    m_small = motp.motion_mask_small(
                        resid[j], sigma, k=tune.motion_k,
                        floor=tune.motion_floor,
                        open_px=tune.motion_open_px,
                        close_px=tune.motion_close_px)
                    motion_cov = round(float(m_small.mean()), 4)
                    fg = cv2.resize(m_small.astype(np.uint8), (w, h),
                                    interpolation=cv2.INTER_NEAREST).astype(bool)

                sam_score: float | None = None
                if masker is not None and tune.edge_refine and fg.any():
                    fg, sam_score = _edge_refine(masker, bgr, fg, tune)

                row, degenerate = _emit_frame(
                    rel, bgr, prompt, fg, tune, masks_dir, preview_dir,
                    i=i, n=n, sam_score=sam_score, motion_coverage=motion_cov,
                    extra_log=f" motion={motion_cov * 100:.1f}%")
            rows_by_idx[i] = row
            row["t"] = dict(ftime)
            n_written += 1
            if degenerate:
                n_degenerate += 1

    rows = [rows_by_idx[i] for i in range(len(frames)) if i in rows_by_idx]
    return rows, n_written, n_degenerate, n_heuristic, ring_stats


def main(argv: list[str] | None = None) -> int:
    """Entry point — surfaces unhandled exceptions as return code 4."""
    try:
        return _main(argv)
    except SystemExit:
        raise
    except Exception:
        import traceback
        traceback.print_exc(file=sys.stderr)
        return 4


def _main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--images", type=Path, required=True,
                   help="Folder of frames (recursed).")
    p.add_argument("--masks", type=Path, required=True,
                   help="Output folder for COLMAP masks (<image_name>.png).")
    p.add_argument("--poses", type=Path, required=True,
                   help="sfm_priors.json — per-image poses + intrinsics.")
    p.add_argument("--preview", type=Path, default=None,
                   help="Output folder for debug overlays. Omit to skip them.")
    p.add_argument("--method", choices=["sam2", "geometric", "motion"],
                   default="sam2",
                   help="sam2 (probe grid), geometric (projected volume only), "
                        "or motion (temporal background subtraction per EL ring).")
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="SAM2.1 checkpoint (.pt).")
    p.add_argument("--model-cfg", default="configs/sam2.1/sam2.1_hiera_b+.yaml",
                   help="SAM2.1 hydra config name (inside the sam2 package).")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--volume-radius-mm", type=float, default=180.0,
                   help="Turntable working-volume cylinder radius (mm).")
    p.add_argument("--volume-height-mm", type=float, default=350.0,
                   help="Cylinder height above its base (mm).")
    p.add_argument("--volume-base-mm", type=float, default=0.0,
                   help="Cylinder base Z (mm; 0 = table level).")
    p.add_argument("--volume-center-xy-mm", default="auto",
                   help="Cylinder axis XY in world mm, 'cx,cy'. Default "
                        "'auto' = the calibrated turntable axis from the "
                        "priors (falls back to 0,0).")
    p.add_argument("--dilate", type=int, default=10,
                   help="Grow the final mask by N px so edges aren't clipped.")
    p.add_argument("--min-area", type=float, default=0.005,
                   help="Minimum component area as a fraction of the frame.")
    p.add_argument("--limit", type=int, default=0,
                   help="Process only the first N frames (debugging).")
    # SAM2 probe tuning — defaults are the module constants above.
    p.add_argument("--sam-long-side", type=int, default=_SAM_LONG_SIDE,
                   help="Inference resolution, long side (model-native 1024).")
    p.add_argument("--containment-min", type=float, default=_CONTAINMENT_MIN,
                   help="Keep a probe mask when this fraction of it lies "
                        "inside the grown silhouette.")
    p.add_argument("--probe-max-fraction", type=float,
                   default=_PROBE_MAX_FRACTION,
                   help="Reject probe masks larger than this fraction of "
                        "the frame.")
    p.add_argument("--top-edge-band-frac", type=float,
                   default=_TOP_EDGE_BAND_FRAC,
                   help="Reject probe masks touching the top band of the "
                        "frame (fraction of height).")
    p.add_argument("--clamp-grow-frac", type=float, default=_CLAMP_GROW_FRAC,
                   help="Silhouette dilation (fraction of min(W,H)) for the "
                        "hard clamp on SAM2 output.")
    p.add_argument("--grid-steps", type=int, default=_GRID_STEPS,
                   help="Probe grid density: N×N samples over the projected "
                        "bbox.")
    p.add_argument("--neg-radius-mult", type=float, default=_NEG_RADIUS_MULT,
                   help="Negative-ring radius as a multiple of the volume "
                        "radius.")
    p.add_argument("--disc-radius-mm", type=float, default=_DISC_RADIUS_MM,
                   help="Physical turntable disc radius (mm); 0 disables the "
                        "disc stamp.")
    p.add_argument("--geometry-composite",
                   action=argparse.BooleanOptionalAction, default=None,
                   help="Stamp the projected disc and the nominal calibrated "
                        "board into every mask. Default ON for sam2/geometric, "
                        "OFF for motion (the projected disc is calibration-"
                        "dependent and drifts off the real disc; motion already "
                        "captures the actual rotating disc).")
    p.add_argument("--connect-overlap-min", type=float,
                   default=_CONNECT_OVERLAP_MIN,
                   help="Growth pass: fraction of a segment that must "
                        "overlap the dilated union to join the assembly.")
    # ── motion method (temporal background subtraction per EL ring) ───────────
    p.add_argument("--plate-long-side", type=int, default=1024,
                   help="Resolution (long side) for plate building + motion "
                        "detection; the mask is upsampled back to full res.")
    p.add_argument("--plate-iters", type=int, default=2,
                   help="Plate passes: 1 = plain median, 2 = + one robust "
                        "rebuild ignoring motion pixels (kills the on-axis ghost).")
    p.add_argument("--motion-k", type=float, default=2.0,
                   help="Foreground threshold = max(floor, k·noise_sigma). Low "
                        "by design: discard a pixel only when it is confidently "
                        "identical to the plate (within noise), else keep it.")
    p.add_argument("--motion-floor", type=float, default=6.0,
                   help="Absolute floor for the motion threshold "
                        "(colour-distance units, 0–441).")
    p.add_argument("--motion-open-px", type=int, default=2,
                   help="Morphological open radius (px, working res) — despeckle.")
    p.add_argument("--motion-close-px", type=int, default=4,
                   help="Morphological close radius (px, working res) — bridge gaps.")
    p.add_argument("--edge-refine", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Snap the motion mask's edges with SAM2 (bounded to a "
                        "band; needs --checkpoint). --no-edge-refine = pure motion.")
    p.add_argument("--edge-band-frac", type=float, default=0.02,
                   help="How far (fraction of min(W,H)) SAM2 may move the motion "
                        "boundary during edge refine.")
    args = p.parse_args(argv)

    # Geometry composite default is method-aware: the projected disc/board
    # stamp helps the SAM2/geometric paths (texture-poor disc), but on motion
    # it only hurts — the stamp follows the calibrated pose, which drifts off
    # the real disc and drags background into the mask. Motion already captures
    # the actual rotating disc, so default the stamp OFF there (still
    # overridable with an explicit --geometry-composite).
    if args.geometry_composite is None:
        args.geometry_composite = args.method != "motion"

    t0 = time.monotonic()

    images_dir = args.images.resolve()
    if not images_dir.is_dir():
        print(f"error: images dir not found: {images_dir}", file=sys.stderr)
        return 1
    if not args.poses.is_file():
        print(f"error: poses file not found: {args.poses}", file=sys.stderr)
        return 1

    center: tuple[float, float] | None = None
    if str(args.volume_center_xy_mm).strip().lower() != "auto":
        try:
            cx, cy = (float(v) for v in str(args.volume_center_xy_mm).split(","))
            center = (cx, cy)
        except ValueError:
            print(f"error: bad --volume-center-xy-mm "
                  f"{args.volume_center_xy_mm!r} (expected 'cx,cy' or 'auto')",
                  file=sys.stderr)
            return 1

    try:
        prompter = GeomPrompter(
            args.poses.resolve(),
            volume_radius_mm=args.volume_radius_mm,
            volume_height_mm=args.volume_height_mm,
            volume_base_mm=args.volume_base_mm,
            volume_center_xy_mm=center,
            grid_steps=args.grid_steps,
            neg_radius_mult=args.neg_radius_mult,
            disc_radius_mm=(args.disc_radius_mm
                            if args.geometry_composite else None),
        )
    except PriorsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    cx, cy = prompter.volume_center_xy_mm

    masker: Sam2Masker | None = None
    if args.method == "sam2":
        if args.checkpoint is None:
            print("error: --method sam2 needs --checkpoint (or run "
                  "--method geometric)", file=sys.stderr)
            return 1
        masker = Sam2Masker(args.checkpoint, args.model_cfg, args.device)
    elif args.method == "motion" and args.edge_refine:
        # Edge refine is optional: fall open to pure motion when SAM2 weights
        # are absent rather than failing the whole run.
        if args.checkpoint is not None and Path(args.checkpoint).is_file():
            masker = Sam2Masker(args.checkpoint, args.model_cfg, args.device)
        else:
            print("note: --edge-refine requested but no SAM2 checkpoint found "
                  "-> running pure motion (no edge snap)", file=sys.stderr)
            args.edge_refine = False

    # Never ingest our own outputs as input frames — masks/preview may sit
    # under --images on a re-run. Compare resolved paths.
    excluded = [d.resolve() for d in (args.masks, args.preview) if d is not None]
    frames = [(rel, ab) for rel, ab in _discover_images(images_dir)
              if not any(ab.is_relative_to(ex) for ex in excluded)]
    if args.limit > 0:
        frames = frames[:args.limit]
    if not frames:
        print(f"error: no images found under {images_dir}", file=sys.stderr)
        return 1

    print(f"input: {len(frames)} frames  poses={args.poses.name} "
          f"({len(prompter.poses)} pose entries)")
    print(f"method: {args.method}  device={args.device}  "
          f"volume=R{args.volume_radius_mm:g}/H{args.volume_height_mm:g}"
          f"/base{args.volume_base_mm:g}mm  dilate={args.dilate}px  "
          f"min_area={args.min_area}")
    if args.method == "motion":
        print(f"tuning: plate_long_side={args.plate_long_side}  "
              f"plate_iters={args.plate_iters}  motion_k={args.motion_k:g}  "
              f"motion_floor={args.motion_floor:g}  "
              f"open/close={args.motion_open_px}/{args.motion_close_px}px  "
              f"edge_refine={'on' if args.edge_refine else 'OFF'}"
              + (f" (band={args.edge_band_frac:g})" if args.edge_refine else ""))
    else:
        print(f"tuning: long_side={args.sam_long_side}  "
              f"containment={args.containment_min:g}  "
              f"probe_max={args.probe_max_fraction:g}  "
              f"top_band={args.top_edge_band_frac:g}  "
              f"clamp_grow={args.clamp_grow_frac:g}  grid={args.grid_steps}x"
              f"{args.grid_steps}  neg_ring={args.neg_radius_mult:g}R  "
              f"overlap_min={args.connect_overlap_min:g}")
    print(f"composite: {'on' if args.geometry_composite else 'OFF'}  "
          f"disc=R{args.disc_radius_mm:g}mm @ ({cx:g},{cy:g})  "
          f"board={'nominal pose from priors' if prompter.board_outline is not None else 'NOT in priors'}")

    args.masks.mkdir(parents=True, exist_ok=True)
    if args.preview is not None:
        args.preview.mkdir(parents=True, exist_ok=True)

    n = len(frames)
    ring_stats: list[dict] | None = None
    try:
        # Build the model up front so its load time is its own phase rather than
        # hiding inside the first frame's encode (and to fail fast on a bad setup).
        if masker is not None:
            with PROF.phase("model_load"):
                masker.ensure_loaded()
        if args.method == "motion":
            (rows, n_written, n_degenerate, n_heuristic,
             ring_stats) = _run_motion(
                args, prompter, masker, frames, n, args.masks, args.preview)
        else:
            rows, n_written, n_degenerate, n_heuristic = _run_perframe(
                args, prompter, masker, frames, n, args.masks, args.preview)
    except Sam2Unavailable as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    n_ok = n_written - n_degenerate
    print(f"\nmasks: {n_written} written ({n_ok} with foreground, "
          f"{n_degenerate} degenerate, {n_heuristic} heuristic prompts) "
          f"-> {args.masks}")
    quality = _quality_summary(rows)
    if quality["coverage_median"] is not None:
        line = f"quality: coverage median={quality['coverage_median']}"
        if quality["mean_sam_disc_recall"] is not None:
            line += f"  mean SAM2 disc recall={quality['mean_sam_disc_recall']}"
        print(line)
    if quality["coverage_outliers"]:
        print("coverage outliers (eyeball these in the preview dir): "
              + ", ".join(quality["coverage_outliers"]))

    wall = time.monotonic() - t0
    for line in PROF.format_lines(wall):
        print(line)

    manifest = {
        "tool": {"name": "colmap-object-masks", "version": _TOOL_VERSION},
        "created": _now(),
        "duration_sec": round(wall, 2),
        # Per-phase timing (model load / encode / decode / IO / post) — the
        # cross-run bottleneck signal.
        "profile": PROF.summary(wall),
        "settings": {
            "method": args.method,
            "checkpoint": (args.checkpoint.name if args.checkpoint else None),
            "device": args.device,
            "volume": {
                "radius_mm": args.volume_radius_mm,
                "height_mm": args.volume_height_mm,
                "base_mm": args.volume_base_mm,
                "center_xy_mm": [cx, cy],
            },
            "dilate_px": args.dilate,
            "min_area_frac": args.min_area,
            "tuning": {
                "sam_long_side": args.sam_long_side,
                "containment_min": args.containment_min,
                "probe_max_fraction": args.probe_max_fraction,
                "top_edge_band_frac": args.top_edge_band_frac,
                "clamp_grow_frac": args.clamp_grow_frac,
                "grid_steps": args.grid_steps,
                "neg_radius_mult": args.neg_radius_mult,
                "connect_overlap_min": args.connect_overlap_min,
            },
            "geometry_composite": {
                "enabled": bool(args.geometry_composite),
                "disc_radius_mm": args.disc_radius_mm,
                "board_from_priors": prompter.board_outline is not None,
            },
        },
        "input": {
            "images_dir": str(images_dir),
            "poses": str(args.poses.resolve()),
            "n_object_frames": n,
            "n_pose_entries": len(prompter.poses),
        },
        "output": {
            "masks_dir": str(args.masks.resolve()),
            "n_masks": n_written,
            "n_with_object": n_ok,
            "n_degenerate": n_degenerate,
            "n_heuristic_prompts": n_heuristic,
            "quality": quality,
            "per_image": rows,
        },
    }
    if args.method == "motion":
        manifest["settings"]["motion"] = {
            "plate_long_side": args.plate_long_side,
            "plate_iters": args.plate_iters,
            "motion_k": args.motion_k,
            "motion_floor": args.motion_floor,
            "motion_open_px": args.motion_open_px,
            "motion_close_px": args.motion_close_px,
            "edge_refine": bool(args.edge_refine),
            "edge_band_frac": args.edge_band_frac,
        }
        manifest["output"]["rings"] = ring_stats
    (args.masks / "_masks_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")

    if n_ok == 0:
        print("error: every frame was degenerate — no foreground isolated. "
              "Check the volume parameters against the rig and the priors "
              "poses.", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
