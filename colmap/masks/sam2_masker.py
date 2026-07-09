"""SAM 2.1 image-predictor wrapper for the mask tool.

Loads the model once (lazily, on first use) and exposes per-frame queries:
``set_frame`` computes the image embedding, then any number of cheap
decoder-only queries run against it. Torch/sam2 are imported inside the class
so ``--method geometric`` keeps working on hosts without them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


class Sam2Unavailable(RuntimeError):
    """torch / sam2 / weights are missing — caller should fall back."""


class Sam2Masker:
    """One lazily-built SAM2.1 image predictor."""

    def __init__(self, checkpoint: Path, model_cfg: str, device: str) -> None:
        self.checkpoint = Path(checkpoint)
        self.model_cfg = model_cfg
        self.device = device          # requested device
        self._eff_device = device     # actual device (set in _ensure; may fall to cpu)
        self._predictor = None

    def _ensure(self):
        if self._predictor is not None:
            return self._predictor
        if not self.checkpoint.is_file():
            raise Sam2Unavailable(
                f"SAM2 checkpoint not found: {self.checkpoint} — download it "
                "first (see colmap/masks/README.md) or run --method geometric"
            )
        try:
            import torch
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as exc:
            raise Sam2Unavailable(
                f"torch/sam2 not importable ({exc}) — install them "
                "(see colmap/masks/README.md) or run --method geometric"
            ) from exc

        device = self.device
        if device == "cuda" and not torch.cuda.is_available():
            print("warning: CUDA requested but unavailable — using CPU "
                  "(slow; consider a tiny/small checkpoint)", file=sys.stderr)
            device = "cpu"
        if device == "cuda":
            # SAM2's recommended inference setup: TF32 matmuls on Ampere+.
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        self._eff_device = device
        sam = build_sam2(self.model_cfg, str(self.checkpoint), device=device)
        self._predictor = SAM2ImagePredictor(sam)
        return self._predictor

    def ensure_loaded(self) -> None:
        """Build the model now instead of lazily on first query.

        Lets the caller time model load as its own phase (it would otherwise
        hide inside the first frame's encode) and surfaces a missing
        checkpoint / unimportable torch up front, before the frame loop."""
        self._ensure()

    def _infer_ctx(self):
        """Mixed-precision inference context. SAM2's recommended path runs the
        network under ``autocast(bfloat16)`` on CUDA — roughly halves encoder
        time and activation memory at no measurable mask-quality cost; bf16
        (not fp16) avoids overflow without a loss scaler. A no-op on CPU.

        Call ``_ensure()`` first so ``_eff_device`` reflects any CUDA->CPU
        fallback."""
        import contextlib

        if self._eff_device == "cuda":
            import torch

            return torch.autocast("cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def set_frame(self, rgb: np.ndarray) -> None:
        """Compute the embedding for one frame (HxWx3 uint8 RGB). Every
        subsequent query call decodes against this frame."""
        predictor = self._ensure()
        with self._infer_ctx():
            predictor.set_image(rgb)

    def query_point(self, xy: tuple[float, float]) -> tuple[np.ndarray, float]:
        """Single positive-point query; multimask on, best score wins.

        One point is SAM2's most reliable prompt — large boxes plus scattered
        point clouds make it hallucinate texture-noise masks on busy scenes.
        Returns ``(mask_bool HxW, score)`` for the current frame.
        """
        hyps = self.query_point_hypotheses(xy)
        best = max(hyps, key=lambda ms: ms[1])
        return best

    def query_point_hypotheses(
        self, xy: tuple[float, float],
    ) -> list[tuple[np.ndarray, float]]:
        """All three multimask hypotheses for a single positive point,
        LARGEST first.

        SAM2's three outputs roughly correspond to subpart / part / whole;
        ``argmax(score)`` routinely prefers a crisp *part* on multi-coloured
        objects. Callers that know the expected extent (e.g. a geometric
        silhouette) should walk this list largest-first and keep the first
        hypothesis that passes their filters.
        """
        predictor = self._ensure()
        with self._infer_ctx():
            masks, scores, _ = predictor.predict(
                point_coords=np.asarray([xy], dtype=np.float32),
                point_labels=np.ones(1, dtype=np.int32),
                multimask_output=True,
            )
        hyps = [
            (np.asarray(m, dtype=bool), float(s))
            for m, s in zip(masks, scores)
        ]
        hyps.sort(key=lambda ms: int(np.count_nonzero(ms[0])), reverse=True)
        return hyps

    def query_points_batched(
        self,
        pts_xy: np.ndarray,
        *,
        chunk: int = 32,
    ) -> list[list[tuple[np.ndarray, float]]]:
        """Multimask hypotheses for many single-positive-point prompts, decoded
        in batches against the current frame.

        Equivalent to calling :meth:`query_point_hypotheses` once per point —
        each result is that point's three hypotheses sorted LARGEST first — but
        the whole batch runs through the mask decoder in one forward per
        ``chunk`` (one GPU<->CPU sync per chunk, not per point). The image
        embedding is reused for every prompt (SAM2's ``repeat_image`` path), so
        only the cheap decoder re-runs; the encoder ran once in ``set_frame``.

        Returns a list aligned 1:1 with the rows of ``pts_xy`` (order kept).
        """
        predictor = self._ensure()
        pts = np.asarray(pts_xy, dtype=np.float32).reshape(-1, 2)
        out: list[list[tuple[np.ndarray, float]]] = []
        for i in range(0, pts.shape[0], chunk):
            sub = pts[i:i + chunk]
            # (B,1,2): one positive point per prompt; the public predict() is
            # single-prompt only, so go through SAM2's batched torch path —
            # _prep_prompts normalizes the coords, _predict decodes all B at
            # once (repeat_image reuses the one embedding).
            coords = sub.reshape(-1, 1, 2)
            labels = np.ones((sub.shape[0], 1), dtype=np.int32)
            with self._infer_ctx():
                _mi, unnorm, lab, _box = predictor._prep_prompts(
                    coords, labels, None, None, normalize_coords=True)
                masks, ious, _ = predictor._predict(
                    unnorm, lab, None, None, multimask_output=True)
            masks_np = masks.detach().to("cpu").numpy().astype(bool)  # (B,C,H,W)
            ious_np = ious.detach().float().to("cpu").numpy()         # (B,C)
            for bi in range(masks_np.shape[0]):
                hyps = [
                    (masks_np[bi, c], float(ious_np[bi, c]))
                    for c in range(masks_np.shape[1])
                ]
                hyps.sort(key=lambda ms: int(np.count_nonzero(ms[0])),
                          reverse=True)
                out.append(hyps)
        return out

    def query_box(
        self,
        box_xyxy: np.ndarray,
        pos_xy: np.ndarray,
        neg_xy: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Box + points query (the heuristic-prompt path)."""
        predictor = self._ensure()
        coords = labels = None
        if pos_xy.size or neg_xy.size:
            coords = np.vstack([pos_xy, neg_xy]).astype(np.float32)
            labels = np.concatenate([
                np.ones(len(pos_xy), dtype=np.int32),
                np.zeros(len(neg_xy), dtype=np.int32),
            ])
        with self._infer_ctx():
            masks, scores, _ = predictor.predict(
                point_coords=coords,
                point_labels=labels,
                box=box_xyxy.astype(np.float32),
                multimask_output=False,
            )
        return np.asarray(masks[0], dtype=bool), float(scores[0])
