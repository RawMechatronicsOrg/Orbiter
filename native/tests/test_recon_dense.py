"""The dense block: what it runs, on which card, and what it refuses.

Everything here drives `recon.run(mode="dense")` through `FakeBackend`, whose
effects leave behind the files a real COLMAP would — a binary two-camera
`cameras.bin`, the `patch-match.cfg` the undistorter writes, a geometric depth
map and a binary `fused.ply`. That is what makes the checks worth having
testable: masks undistorted through the wrong eye, a fusion that quietly fuses
everything, and a workspace half-built by a card that has no kernel.

The photographs are **real** ones. `clean_images` decodes them, finds the
stripe and paints it out, and `undistort_masks` reads the mask it cached, so a
fixture of placeholder bytes would test nothing about either. They are a
quarter of the rig's own frame in each direction — 320x180, with the intrinsics
scaled to match, which leaves the projected geometry identical — because this
file runs the whole chain a dozen times and a 1280x720 inpaint costs sixteen
times as much for no extra assertion.

The laser cloud is `test_views`' box, sampled every 5.6 mm. That is coarser
than a real laser cloud, and coarser than `MergeParams.support_mm` (3.0 mm) can
see: no dense point would have five laser neighbours within 3 mm *laterally*,
so every merge here would refuse for want of overlap. The fixture therefore
widens the lateral radius to 8 mm, which is the same rule measured against the
fixture's own density. The rule itself — the candidate ball, the signed
residual, the cells and every gate — is tested against `scan.merge_clouds`
directly in `test_merge.py`, where no backend is involved.
"""

from __future__ import annotations

import json
import shutil
import struct
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pytest

from orbiter_native import recon
from orbiter_native.laser import StripePixels
from orbiter_native.photos import (
    BoardSnapshot,
    EyePhoto,
    EyeSnapshot,
    Extrinsics,
    PhotoCandidate,
    PhotoSession,
    PhotoWriter,
    RigSnapshot,
)
from orbiter_native.recon import (
    ALL_DEVICES,
    CANONICAL_STEPS,
    FALLBACK_CLEARS,
    FALLBACK_MAX_IMAGE_SIZE,
    MODE_STEPS,
    DockerBackend,
    FakeBackend,
    ReconParams,
    ReconRefused,
    State,
    StepFailed,
)
from orbiter_native.scan import MergeParams, ScanVolume, write_ply
from orbiter_native.stereo import compose_right_pose
from orbiter_native.stripemask import MASK_IGNORE, MASK_USE, MaskParams
from orbiter_native.views import CfgParams, load_session

from test_recon import PLACES, _listed, _texture_workspace, _textured_ply
from test_stereo_scan import KL, KR, R_TRUE, T_TRUE, WH
from test_stereo_scan import _rig as _stereo_rig
from test_views import BOX_N, BOX_XYZ, _pair_pose

#: The fixture's frames are this fraction of the rig's own, in each direction.
#: `clean_images` and `undistort_masks` are the only steps in this project that
#: touch every pixel of every photograph, and they run in a dozen tests here.
SCALE = 4
WH_D = (WH[0] // SCALE, WH[1] // SCALE)

#: `nvidia-smi -L`, as this machine prints it. The runner matches
#: `ORBITER_COLMAP_GPU` (default "5060") against the name and the UUID, and
#: falls back to the first entry with a different UUID.
GPU_5060 = "GPU-d5d610ae-1111-2222-3333-444455556666"
GPU_1650 = "GPU-9f0c7b21-aaaa-bbbb-cccc-ddddeeeeffff"
GPU_LIST = [f"GPU 0: NVIDIA GeForce RTX 5060 Ti (UUID: {GPU_5060})",
            f"GPU 1: NVIDIA GeForce GTX 1650 SUPER (UUID: {GPU_1650})"]
#: `FakeBackend` keys its canned output by `argv[1]`, and the probe's argv is
#: `["nvidia-smi", "-L"]`.
GPU_TOOL = "-L"

#: The undistorted sizes the fake workspace's two cameras carry. They differ on
#: purpose: a mask built through the wrong eye's entry comes out the wrong
#: shape, which is the cheapest possible detector for the bug.
NEW_WH = {1: (WH_D[0], WH_D[1]), 2: (WH_D[0] - 4, WH_D[1] - 4)}

#: The laser cloud here is a 5.6 mm grid, so lateral support has to be counted
#: at a radius that grid can fill. Everything else is the shipping default.
MERGE = MergeParams(support_mm=8.0)
#: The stripe is 2 px wide in a 180-px frame; the shipping 24 px of growth is
#: tuned for 1080p and would mask a seventh of this fixture.
MASK = MaskParams(dilate_px=6)


def _params(**kw) -> ReconParams:
    return ReconParams(merge=MERGE, mask=MASK, **kw)


# ── the rig, a quarter of the size ───────────────────────────────────────


def _eye(k, camera_id: str) -> EyeSnapshot:
    """One eye at `WH_D`. Scaling the frame and the camera matrix by the same
    factor leaves every normalised ray where it was, so the selection, the
    buckets and the tracks come out identical to `test_recon`'s."""
    return EyeSnapshot(camera_id=camera_id, wh=WH_D, fx=k.fx / SCALE,
                       fy=k.fy / SCALE, cx=k.cx / SCALE, cy=k.cy / SCALE,
                       dist=(0.0,) * 5, rms_px=0.81)


def _rig() -> RigSnapshot:
    return RigSnapshot(
        board=BoardSnapshot(8, 8, 36.0, 26.64, "DICT_5X5_100"),
        volume=ScanVolume(height_mm=400.0, radius_mm=200.0, floor_mm=5.0),
        left=_eye(KL, "cam2"), right=_eye(KR, "cam4"),
        extrinsics=Extrinsics(R=R_TRUE, t_mm=T_TRUE, rms_px=0.89))


# ── photographs with a stripe in them ────────────────────────────────────


def _frame(k: int, laser_on: bool) -> tuple[bytes, StripePixels | None]:
    """One photograph, and the pixels its eye called stripe.

    The background is a sawtooth in grey, which gives the inpainter something
    to extend inward and the JPEG something to differ over; `stripe_score`
    measures redness, so grey scores zero however sharp its edges are. The
    stripe is a red line whose place moves with `k`, so no two photographs
    carry the same one.
    """
    width, height = WH_D
    yy, xx = np.mgrid[0:height, 0:width]
    grey = (((xx * 3 + yy * 5 + k * 7) % 200) + 30).astype(np.uint8)
    bgr = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)
    if not laser_on:
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        assert ok
        return buf.tobytes(), None

    row = 55 + (k % 30)
    lit = np.zeros((height, width), np.uint8)
    cv2.line(lit, (0, row), (width - 1, row + 10), 255, 2)
    bgr[lit > 0] = (24, 24, 245)
    ys, xs = np.nonzero(lit)
    stripe = StripePixels(
        x=xs.astype(np.int32), y=ys.astype(np.int32),
        w=np.full(len(xs), 255, np.uint8), r=np.full(len(xs), 245, np.uint8),
        wh=WH_D, along_x=True, reason=None)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    assert ok
    return buf.tobytes(), stripe


def _record(writer: PhotoWriter, az: float, k: int, *, pass_id: int,
            laser_on: bool) -> None:
    """One pair, written straight through the writer rather than through its
    bounded queue, which drops the oldest entry by design."""
    geom = _stereo_rig().geom
    R, t = _pair_pose(az)
    R_r, t_r = compose_right_pose(R, t, geom)
    left_jpeg, left_stripe = _frame(k, laser_on)
    right_jpeg, right_stripe = _frame(k + 500, laser_on)
    cand = PhotoCandidate(
        left=EyePhoto(camera_id="cam2", jpeg=left_jpeg, wh=WH_D,
                      capture_mono=1000.0 + k, R=R, t_mm=t,
                      sharpness=100.0 + k, stripe=left_stripe),
        right=EyePhoto(camera_id="cam4", jpeg=right_jpeg, wh=WH_D,
                       capture_mono=1000.0 + k - 0.012, R=R_r, t_mm=t_r,
                       sharpness=90.0 + k, stripe=right_stripe,
                       stripe_shifted=True),
        pair_capture_mono=1000.0 + k, pose_source="left+right",
        pose_rms_px=0.31, pose_gap_deg=0.2, pose_gap_mm=1.0, pose_corners=24,
        pose_smooth_mm=0.21, pass_id=pass_id, laser_on=laser_on,
        # The frame's own kept points: a slice of one wall, so the term the
        # sidecar contributes to the mask is a blob rather than the object.
        kept_xyz_board=BOX_XYZ[:1600:50].astype(np.float32) if laser_on
        else np.zeros((0, 3), np.float32))
    writer._write(cand.record("left"))
    writer._write(cand.record("right"))


def _session(tmp_path, *, places=PLACES, clean_places=PLACES[:11],
             cloud: np.ndarray | None = None) -> Path:
    """A session the dense chain can be run against: the laser pass at twelve
    places, the clean pass at eleven of them, and the box exported as
    `laser.ply`.

    The one place the operator did not re-photograph with the laser off is what
    makes this a **dense** fixture rather than `test_recon`'s. Selection prefers
    the clean photograph wherever both passes stood — the clean pass is later
    and therefore sharper — so a clean pass standing everywhere leaves nothing
    for `clean_images` to inpaint and no cached mask for `undistort_masks` to
    undistort. Eleven of twelve also keeps the clean set above both texture
    gates, so `texture_source` is "clean" and the run is not degraded.
    """
    session = PhotoSession(tmp_path, rig=_rig())
    writer = PhotoWriter(session)
    k = 0
    for pass_id, laser_on, where in ((0, True, places), (1, False, clean_places)):
        for az in where:
            _record(writer, az, k, pass_id=pass_id, laser_on=laser_on)
            k += 1
    write_ply(str(session.path / recon.LASER_PLY),
              BOX_XYZ if cloud is None else cloud)
    return session.path


# ── what the dense tools leave behind ────────────────────────────────────


def _camera_bin(camera_id: int, k, wh: tuple[int, int]) -> bytes:
    """One PINHOLE entry of `cameras.bin` — the model `image_undistorter`
    writes, and the only one `read_pinhole_bin` will measure."""
    return (struct.pack("<Ii", camera_id, 1) + struct.pack("<QQ", *wh)
            + struct.pack("<4d", k.fx / SCALE, k.fy / SCALE,
                          k.cx / SCALE, k.cy / SCALE))


def _image_bin(image_id: int, camera_id: int, name: str) -> bytes:
    return (struct.pack("<I", image_id)
            + struct.pack("<7d", 1.0, 0.0, 0.0, 0.0, 5.0, -6.0, 300.0)
            + struct.pack("<I", camera_id) + name.encode("utf-8") + b"\x00"
            + struct.pack("<Q", 0))


def _dense_workspace(session_dir: Path, names: list[str]) -> None:
    """What `image_undistorter --output_path colmap/dense` leaves behind: the
    undistorted images, a BINARY two-camera model, and the `patch-match.cfg`
    that is the authoritative list of what COLMAP registered."""
    dense = session_dir / "colmap" / "dense"
    for part in ("images", "sparse", "stereo"):
        (dense / part).mkdir(parents=True, exist_ok=True)
    for name in names:
        (dense / "images" / name).write_bytes(b"\xff\xd8undistorted\xff\xd9")
    (dense / "sparse" / "cameras.bin").write_bytes(
        struct.pack("<Q", 2) + _camera_bin(1, KL, NEW_WH[1])
        + _camera_bin(2, KR, NEW_WH[2]))
    (dense / "sparse" / "images.bin").write_bytes(
        struct.pack("<Q", len(names))
        + b"".join(_image_bin(i + 1, 1 if name.startswith("left") else 2, name)
                   for i, name in enumerate(names)))
    (dense / "stereo" / "patch-match.cfg").write_text(
        "".join(f"{name}\n__auto__, 20\n" for name in names), encoding="utf-8")


def _depth_maps(session_dir: Path, names: list[str]) -> None:
    """One geometric depth map, in COLMAP's own `<w>&<h>&<c>&` layout, so the
    runner can read back whether a forced depth range took effect."""
    out = session_dir / "colmap" / "dense" / "stereo" / "depth_maps"
    out.mkdir(parents=True, exist_ok=True)
    values = np.array([0.0, 210.0, 0.0, 480.0], np.float32)
    for name in names[:2]:
        (out / f"{name}.geometric.bin").write_bytes(
            b"2&2&1&" + values.tobytes())


#: A patch of dense surface 30 mm off the box's +x wall: no lateral laser
#: support anywhere near it, so it is what the merge keeps as hole fill.
_FILL = np.stack(np.meshgrid(np.linspace(140.0, 160.0, 25),
                             np.linspace(-10.0, 10.0, 20),
                             [50.0], indexing="ij"), axis=-1).reshape(-1, 3)


def _fused(session_dir: Path, xyz: np.ndarray | None = None,
           normals: np.ndarray | None = None) -> None:
    """`fused.ply`, binary and with normals, which is what COLMAP writes."""
    if xyz is None:
        xyz = np.vstack([BOX_XYZ, _FILL])
        normals = np.vstack([BOX_N, np.tile([1.0, 0.0, 0.0], (len(_FILL), 1))])
    write_ply(str(session_dir / "colmap" / "dense" / "fused.ply"), xyz, None,
              normals if normals is not None else np.tile([0.0, 0.0, 1.0],
                                                          (len(xyz), 1)))


def _effects(session_dir: Path, *, fused: np.ndarray | None = None,
             fused_normals: np.ndarray | None = None,
             ) -> dict[str, Callable[[list[str]], None]]:
    """The files each COLMAP tool leaves behind, keyed by tool.

    `image_undistorter` runs twice in the dense chain — once for the texture
    workspace and once for `colmap/dense` — and the two leave behind entirely
    different trees, so the effect reads its own `--output_path` exactly as the
    real tool would.
    """
    def undistort(argv: list[str]) -> None:
        output = argv[argv.index("--output_path") + 1]
        if output.endswith("/texture"):
            _texture_workspace(session_dir, _listed(session_dir))
        else:
            _dense_workspace(session_dir, _selected(session_dir))

    return {
        "model_converter": lambda argv: (
            session_dir / "colmap" / "_validate.ply").write_bytes(b"ply\n"),
        "image_undistorter": undistort,
        "patch_match_stereo": lambda argv: _depth_maps(
            session_dir, _selected(session_dir)),
        "stereo_fusion": lambda argv: _fused(session_dir, fused, fused_normals),
        "poisson_mesher": lambda argv: (
            session_dir / "mesh" / "meshed-poisson.ply").write_bytes(b"ply\n"),
        "mesh_texturer": lambda argv: _textured_ply(session_dir),
    }


def _selected(session_dir: Path) -> list[str]:
    """The selected photographs' names, in `images.txt` order."""
    return [line.split()[-1] for line
            in (session_dir / "colmap" / "sparse" / "images.txt")
            .read_text(encoding="utf-8").splitlines()
            if line and not line.startswith("#") and line.endswith(".jpg")]


def _fake(session_dir: Path, *, exits: dict[str, int] | None = None,
          gpus: list[str] | None = None, **kw) -> FakeBackend:
    output = kw.pop("output", None) or {}
    output.setdefault(GPU_TOOL, GPU_LIST if gpus is None else gpus)
    return FakeBackend(exits=exits, output=output,
                       effects=_effects(session_dir, **kw))


def _run(session_dir: Path, backend: FakeBackend | None = None, **kw):
    backend = backend or _fake(session_dir)
    params = kw.pop("params", None) or _params()
    result = recon.run(session_dir, kw.pop("mode", "dense"), backend=backend,
                       params=params, **kw)
    return result, backend


def _calls(backend: FakeBackend, tool: str) -> list[list[str]]:
    return [argv for argv in backend.calls if len(argv) > 1 and argv[1] == tool]


def _gpu_of(backend: FakeBackend, tool: str):
    """The `GpuSpec` the single call to `tool` carried."""
    found = [gpu for argv, gpu in zip(backend.calls, backend.gpus)
             if len(argv) > 1 and argv[1] == tool]
    assert len(found) == 1, f"{tool} ran {len(found)} times"
    return found[0]


def _log(session_dir: Path) -> str:
    return (session_dir / recon.LOG_NAME).read_text(encoding="utf-8")


def _block(session_dir: Path) -> dict:
    return json.loads((session_dir / "session.json")
                      .read_text(encoding="utf-8"))["reconstruct"]


# ── the argv, exactly ────────────────────────────────────────────────────


def _expected_argv(names: int = 24) -> list[list[str]]:
    """Every command the dense chain sends, in order. D1a owns the
    texture-only fixture; this is D1b's, so the two stories cannot fight over
    one golden list."""
    return [
        ["colmap", "-h"],
        ["colmap", "model_converter",
         "--input_path", "/data/colmap/sparse",
         "--output_path", "/data/colmap/_validate.ply",
         "--output_type", "PLY"],
        ["colmap", "image_undistorter",
         "--image_path", "/data/colmap/images_clean",
         "--input_path", "/data/colmap/sparse",
         "--output_path", "/data/colmap/texture",
         "--output_type", "COLMAP",
         "--max_image_size", "-1",
         "--image_list_path", "/data/colmap/texture_images.txt"],
        ["colmap", "image_undistorter",
         "--image_path", "/data/colmap/images_clean",
         "--input_path", "/data/colmap/sparse",
         "--output_path", "/data/colmap/dense",
         "--output_type", "COLMAP",
         "--max_image_size", "1600"],
        ["nvidia-smi", "-L"],
        ["colmap", "patch_match_stereo",
         "--workspace_path", "/data/colmap/dense",
         "--workspace_format", "COLMAP",
         "--PatchMatchStereo.geom_consistency", "true",
         "--PatchMatchStereo.num_iterations", "5",
         "--PatchMatchStereo.filter_min_ncc", "0.1",
         "--PatchMatchStereo.filter_min_triangulation_angle", "3",
         "--PatchMatchStereo.gpu_index", "0"],
        ["colmap", "stereo_fusion",
         "--workspace_path", "/data/colmap/dense",
         "--workspace_format", "COLMAP",
         "--input_type", "geometric",
         "--StereoFusion.max_reproj_error", "3.0",
         "--StereoFusion.max_depth_error", "0.015",
         "--StereoFusion.min_num_pixels", "4",
         "--StereoFusion.mask_path", "/data/colmap/dense/masks",
         "--output_path", "/data/colmap/dense/fused.ply"],
        ["colmap", "poisson_mesher",
         "--input_path", "/data/colmap/dense/merged.ply",
         "--output_path", "/data/mesh/meshed-poisson.ply",
         "--PoissonMeshing.trim", "7"],
        ["colmap", "mesh_texturer",
         "--workspace_path", "/data/colmap/texture",
         "--input_path", "/data/mesh/meshed-poisson.ply",
         "--output_path", "/data/mesh",
         "--MeshTextureMapping.texture_scale_factor", "1",
         "--MeshTextureMapping.atlas_patch_padding", "4",
         "--MeshTextureMapping.inpaint_radius", "5",
         "--MeshTextureMapping.min_visible_vertices", "3",
         "--MeshTextureMapping.apply_color_correction", "1"],
    ]


def test_dense_step_order_and_exact_argv(tmp_path) -> None:
    """The dense chain, argument for argument.

    Golden because every one of these flags was read off COLMAP 4.2.0 and a
    silent change to any of them is a run that finishes and is wrong. Six of
    the thirteen steps send no command at all — `write_sparse`, `clean_images`,
    `undistort_masks`, `write_patch_match_cfg`, `merge` and `glb` are host
    steps — and that shape is part of what is being asserted.
    """
    session_dir = _session(tmp_path)
    result, backend = _run(session_dir)

    assert result.ran == MODE_STEPS["dense"] == CANONICAL_STEPS
    assert backend.calls == _expected_argv()
    assert result.texture_source == "clean"


def test_clean_images_runs_before_texture_workspace_and_undistorter(
        tmp_path) -> None:
    """Both dense consumers of `colmap/images_clean/` read it after it exists.

    Revision 3 built the inpainted images inside `undistort_masks`, which runs
    after `image_undistorter` — so PatchMatch matched against raw stripes and a
    degraded texture workspace was undistorted from raw photographs while
    `session.json` claimed it came from inpainted ones.
    """
    session_dir = _session(tmp_path)
    result, backend = _run(session_dir)

    order = list(result.ran)
    assert order.index("clean_images") < order.index("texture_workspace")
    assert order.index("clean_images") < order.index("image_undistorter")
    for argv in _calls(backend, "image_undistorter"):
        assert argv[3] == "/data/colmap/images_clean"

    state = State.read(session_dir)
    assert state.mode == "dense"
    assert set(state.steps) == set(CANONICAL_STEPS)


# ── the pixels COLMAP reads ──────────────────────────────────────────────


def test_dense_images_are_inpainted_not_raw(tmp_path) -> None:
    """A laser photograph in `colmap/images_clean/` has had its stripe painted
    out, and the same NAME under `colmap/images/` still has it.

    This is the regression net for the ordering bug: the bytes differ, the
    redness where the stripe was is gone, and the archive copy in `clean/` is
    the lossless one.
    """
    session_dir = _session(tmp_path)
    _run(session_dir, to_step="clean_images")

    laser = [name for name in _selected(session_dir)
             if _meta(session_dir, name).laser_on]
    assert laser, "the fixture has laser-pass photographs in the selection"
    name = laser[0]

    raw = (session_dir / "colmap" / "images" / name).read_bytes()
    clean = (session_dir / "colmap" / "images_clean" / name).read_bytes()
    assert clean != raw

    def redness(data: bytes) -> float:
        bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR
                           ).astype(float)
        return float((bgr[:, :, 2] - bgr[:, :, :2].max(axis=2)).max())

    assert redness(raw) > 100.0                    # the stripe is in there
    assert redness(clean) < 30.0                   # and it is not in this one

    mask = session_dir / "colmap" / "_masks_raw" / f"{name}.png"
    assert mask.is_file(), "the raw-frame mask is cached for undistort_masks"
    archive = session_dir / "clean" / f"{Path(name).stem}.png"
    assert archive.is_file() and archive.read_bytes()[1:4] == b"PNG"


def test_clean_images_never_touches_colmap_images(tmp_path) -> None:
    """`colmap/images/` is `write_sparse`'s, verbatim in both modes, and no
    later step rewrites it.

    Revision 4 overwrote it in place, which made one path mean different pixels
    depending on the mode of the last run — and no step could tell by looking.
    """
    session_dir = _session(tmp_path)
    _run(session_dir, to_step="validate_sparse")

    images = session_dir / "colmap" / "images"
    before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns)
              for p in images.iterdir()}
    _run(session_dir, only="clean_images")

    after = {p.name: (p.read_bytes(), p.stat().st_mtime_ns)
             for p in images.iterdir()}
    assert after == before

    clean = session_dir / "colmap" / "images_clean"
    assert sorted(p.name for p in clean.iterdir()) == sorted(before)


def test_clean_photos_are_copied_verbatim_into_images_clean(tmp_path) -> None:
    """`colmap/images_clean/` holds the FULL selection — the clean-pass
    photographs copied across byte for byte, not only the inpainted ones — so
    `--image_path` can point at that one directory."""
    session_dir = _session(tmp_path)
    _run(session_dir, to_step="clean_images")

    names = _selected(session_dir)
    verbatim = 0
    for name in names:
        raw = (session_dir / "colmap" / "images" / name).read_bytes()
        clean = (session_dir / "colmap" / "images_clean" / name).read_bytes()
        if not _meta(session_dir, name).laser_on:
            assert clean == raw, name
            verbatim += 1
            assert not (session_dir / "colmap" / "_masks_raw"
                        / f"{name}.png").exists()
    # The clean pass stood at eleven of the twelve places, and selection
    # prefers it wherever both stood.
    assert verbatim == 22 and len(names) == 24


def _meta(session_dir: Path, name: str):
    """One photograph's manifest row, by the name `images.txt` knows it by."""
    _, photos = load_session(session_dir)
    return next(photo for photo in photos if photo.name == name)


# ── the fusion masks ─────────────────────────────────────────────────────


def test_undistort_masks_writes_one_mask_per_selected_image(tmp_path) -> None:
    """One file per SELECTED image, not per image that carried a stripe.

    A clean-pass photograph has no cached mask and gets the silhouette term
    alone — which is precisely what keeps the table out of fusion for those
    photographs, and they need it as much as the laser ones do. A silent
    mismatch here makes `stereo_fusion` fuse everything for the images it
    missed.
    """
    session_dir = _session(tmp_path)
    _run(session_dir, to_step="undistort_masks")

    names = _selected(session_dir)
    masks = sorted(p.name for p in
                   (session_dir / "colmap" / "dense" / "masks").iterdir())
    assert masks == sorted(f"{name}.png" for name in names)
    assert len(masks) == 24

    # A clean-pass photograph's mask is silhouette-only: its interior is white
    # and nothing in it was ignored for being stripe.
    clean = next(name for name in names if not _meta(session_dir, name).laser_on)
    mask = _read_mask(session_dir, clean)
    assert (mask == MASK_USE).any()
    assert set(np.unique(mask)) <= {MASK_IGNORE, MASK_USE}


def _read_mask(session_dir: Path, name: str) -> np.ndarray:
    data = (session_dir / "colmap" / "dense" / "masks" / f"{name}.png"
            ).read_bytes()
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)


def test_fusion_mask_excludes_pixels_outside_the_laser_silhouette(
        tmp_path) -> None:
    """The mask is two things unioned as *ignore*: the stripe, and everything
    outside the projected laser silhouette.

    Without the second term the table, the backdrop and the operator's hands
    are all fusable surface — inside no `ScanVolume`, but `stereo_fusion` does
    not know that, and every point fused out there deflates `agree_frac`.
    """
    session_dir = _session(tmp_path)
    _run(session_dir, to_step="undistort_masks")

    session, _ = load_session(session_dir)
    laser_xyz, _, _ = recon.read_ply_full(str(session_dir / recon.LASER_PLY))
    name = next(n for n in _selected(session_dir)
                if n.startswith("left") and _meta(session_dir, n).laser_on)
    photo = _meta(session_dir, name)
    mask = _read_mask(session_dir, name)
    assert mask.shape == (NEW_WH[1][1], NEW_WH[1][0])

    # Where the cloud actually lands, computed here rather than trusted.
    eye = session.eye(photo.side)
    cam = laser_xyz @ np.asarray(photo.R).T + np.asarray(photo.t_mm)
    cam = cam[cam[:, 2] > 1e-6]
    u = eye.fx * cam[:, 0] / cam[:, 2] + eye.cx
    v = eye.fy * cam[:, 1] / cam[:, 2] + eye.cy
    # The step grows the silhouette by half of each element it applies.
    grow = (recon.SILHOUETTE_DISC_PX + recon.SILHOUETTE_CLOSE_PX
            + recon.SILHOUETTE_DILATE_PX) // 2 + 2
    used_v, used_u = np.nonzero(mask == MASK_USE)
    assert len(used_u), "the object's own pixels survive"
    assert used_u.min() >= u.min() - grow and used_u.max() <= u.max() + grow
    assert used_v.min() >= v.min() - grow and used_v.max() <= v.max() + grow

    # Nothing far from where the cloud landed survives, whatever the object's
    # own shape in this frame happens to be. Camera 1's `newK` is the left
    # eye's own, so these pixels and the mask's are the same pixels.
    hit = np.zeros(mask.shape, np.uint8)
    ui, vi = np.rint(u).astype(int), np.rint(v).astype(int)
    on = ((ui >= 0) & (ui < mask.shape[1]) & (vi >= 0) & (vi < mask.shape[0]))
    hit[vi[on], ui[on]] = 1
    far = cv2.dilate(hit, np.ones((2 * grow + 1,) * 2, np.uint8)) == 0
    assert far.mean() > 0.05, "there is frame outside the object to be ignored"
    assert (mask[far] == MASK_IGNORE).all()

    # And the stripe is ignored inside the silhouette, which is the other term:
    # the undistorted mask still carries the band the raw one did.
    raw = cv2.imdecode(np.frombuffer(
        (session_dir / "colmap" / "_masks_raw" / f"{name}.png").read_bytes(),
        np.uint8), cv2.IMREAD_GRAYSCALE)
    band = (raw == MASK_IGNORE)
    assert band.any()
    assert (mask[band[:mask.shape[0], :mask.shape[1]]] == MASK_IGNORE).all()


def test_undistort_masks_uses_each_eyes_own_newK(tmp_path) -> None:
    """A two-camera `cameras.bin`, and the right image's mask is built from
    camera 2.

    `image_undistorter` chooses a `newK` and an output size per camera. Using
    the first one for both — which is what the reference does, because it runs
    a single camera — warps every right-eye mask it touches, and the cheapest
    detector for that is the shape of the file it wrote.
    """
    session_dir = _session(tmp_path)
    _run(session_dir, to_step="undistort_masks")

    assert NEW_WH[1] != NEW_WH[2], "the fixture's two eyes undistort differently"
    for name in _selected(session_dir):
        expected = NEW_WH[1 if name.startswith("left") else 2]
        assert _read_mask(session_dir, name).shape == (expected[1], expected[0]), name


def test_mask_flag_is_dropped_when_no_masks_were_written(tmp_path) -> None:
    """No masks, no `--StereoFusion.mask_path` — and a warning saying what that
    costs.

    COLMAP looks a mask up by name, so a mask directory that exists and is
    empty and one that was never made are the same thing to it. Passing the
    flag at an empty directory would fuse the stripe and the table while the
    argv claimed otherwise.
    """
    session_dir = _session(tmp_path)
    _run(session_dir, to_step="write_patch_match_cfg")
    shutil.rmtree(session_dir / "colmap" / "dense" / "masks")

    result, backend = _run(session_dir, only="stereo_fusion")
    argv = backend.argv_for("stereo_fusion")
    assert "--StereoFusion.mask_path" not in argv
    assert argv[-2:] == ["--output_path", "/data/colmap/dense/fused.ply"]
    assert any("no masks were written" in w for w in result.warnings)


# ── the cfg the undistorter wrote ────────────────────────────────────────


def test_patch_match_cfg_is_rewritten_from_the_undistorters_file(
        tmp_path) -> None:
    """The image lines are the undistorter's, verbatim and in order; only the
    source lines change.

    Naming a frame COLMAP did not register aborts the whole stage before it
    computes a single depth map, so the file is edited rather than generated —
    and the edit keeps Unix endings, because the file it edits had them.
    """
    session_dir = _session(tmp_path)
    _run(session_dir, to_step="image_undistorter")
    cfg = session_dir / "colmap" / "dense" / "stereo" / "patch-match.cfg"
    before = cfg.read_text(encoding="utf-8").splitlines()
    assert before[1] == "__auto__, 20"

    _run(session_dir, only="write_patch_match_cfg")
    raw = cfg.read_bytes()
    assert b"\r\n" not in raw
    after = raw.decode("utf-8").splitlines()

    assert after[0::2] == before[0::2], "every image line, verbatim and in order"
    rewritten = [line for line in after[1::2] if line != "__auto__, 20"]
    assert len(rewritten) == len(before) // 2
    for line in rewritten:
        sources = [part.strip() for part in line.split(",")]
        assert sources and all(name.endswith(".jpg") for name in sources)

    logged = _log(session_dir)
    assert f"write_patch_match_cfg: {len(rewritten)} images given" in logged
    assert "0 selected images the undistorter never registered" in logged


# ── the one invocation that asks for a card ──────────────────────────────


def test_only_patch_match_carries_the_gpu_and_the_jit_cache(tmp_path) -> None:
    """`--gpus device=GPU-<uuid>` and the JIT volume ride on `patch_match_stereo`
    and on nothing else.

    A broken nvidia container runtime must not be able to fail `write_sparse`,
    and the JIT cache exists so a compiled compute_90 PTX outlives the `--rm`
    container that compiled it. The listing probe is the one other invocation
    that asks for a device — every card, so that it can name them — and it
    mounts no cache, because a listing compiles nothing.
    """
    session_dir = _session(tmp_path)
    _, backend = _run(session_dir)

    docker = DockerBackend(session_dir, image="colmap/colmap:latest")
    with_cache = [argv[1] for argv, gpu in zip(backend.calls, backend.gpus)
                  if "jitcache" in " ".join(docker.command(argv, gpu))]
    assert with_cache == ["patch_match_stereo"]

    pinned = _gpu_of(backend, "patch_match_stereo")
    assert pinned is not None and pinned.uuid == GPU_5060
    command = docker.command(backend.argv_for("patch_match_stereo"), pinned)
    assert command[command.index("--gpus") + 1] == f"device={GPU_5060}"
    assert "CUDA_CACHE_PATH=/jitcache" in command

    probe = _gpu_of(backend, GPU_TOOL)
    assert probe is not None and probe.uuid == ALL_DEVICES
    listing = docker.command(["nvidia-smi", "-L"], probe)
    assert listing[listing.index("--gpus") + 1] == "all"
    assert "jitcache" not in " ".join(listing)


def test_gpu_is_resolved_by_uuid_lazily_before_patch_match(
        tmp_path, monkeypatch) -> None:
    """The probe runs immediately before the step that needs it, and the card
    is chosen by identity.

    `--gpus all` plus `CUDA_DEVICE_ORDER=PCI_BUS_ID` does not make a chosen card
    device 0 — slot order decides — so the runner passes a UUID and the
    container sees exactly one device, which is then its `gpu_index 0`.
    """
    session_dir = _session(tmp_path)
    _, backend = _run(session_dir)

    tools = [argv[1] for argv in backend.calls]
    assert tools[tools.index("patch_match_stereo") - 1] == GPU_TOOL
    assert tools.index(GPU_TOOL) > tools.index("image_undistorter")
    assert _gpu_of(backend, "patch_match_stereo").name == \
        "NVIDIA GeForce RTX 5060 Ti"

    # A name substring picks the other card...
    other = _session(tmp_path / "other")
    monkeypatch.setenv(recon.COLMAP_GPU_ENV, "1650")
    _, backend = _run(other)
    assert _gpu_of(backend, "patch_match_stereo").uuid == GPU_1650

    # ...a UUID picks it exactly...
    exact = _session(tmp_path / "exact")
    monkeypatch.setenv(recon.COLMAP_GPU_ENV, GPU_1650)
    _, backend = _run(exact)
    assert _gpu_of(backend, "patch_match_stereo").uuid == GPU_1650

    # ...and nothing matching falls to the first entry, and says so.
    none = _session(tmp_path / "none")
    monkeypatch.setenv(recon.COLMAP_GPU_ENV, "Quadro")
    _, backend = _run(none)
    assert _gpu_of(backend, "patch_match_stereo").uuid == GPU_5060
    assert "no card matches ORBITER_COLMAP_GPU='Quadro'" in _log(none)


# ── the fallback ─────────────────────────────────────────────────────────


def _healing(session_dir: Path, output: list[str], *, single: bool = False
             ) -> FakeBackend:
    """A backend whose PatchMatch fails once with `output` and then succeeds.

    The second attempt is let through by the `image_undistorter` the fallback
    itself re-runs, which is the only hook a scripted-exit fake gives — and it
    is the right one: it fires exactly when the workspace has been rebuilt.
    """
    backend = _fake(session_dir, exits={"patch_match_stereo": 1},
                    output={"patch_match_stereo": output},
                    gpus=GPU_LIST[:1] if single else None)
    undistort = backend.effects["image_undistorter"]
    rebuilds: list[int] = []

    def heal(argv: list[str]) -> None:
        undistort(argv)
        if argv[argv.index("--output_path") + 1].endswith("/dense"):
            rebuilds.append(1)
            if len(rebuilds) >= 2:            # the workspace the retry rebuilt
                backend.exits.pop("patch_match_stereo", None)

    backend.effects["image_undistorter"] = heal
    return backend


KERNEL = ["Preparing configuration", "CUDA error: no kernel image is "
          "available for execution on the device"]
OOM = ["Preparing configuration", "CUDA error: out of memory"]


def test_kernel_error_falls_back_from_undistort_at_a_smaller_size(
        tmp_path) -> None:
    """A missing sm_120 kernel re-runs from `image_undistorter`, on the other
    card, at 1000.

    `--max_image_size` is a flag of the undistorter, so lowering it means
    rebuilding the workspace — and the retry runs on the fallback device
    because retrying the same missing kernel on the same card cannot help.
    """
    session_dir = _session(tmp_path)
    result, backend = _run(session_dir, _healing(session_dir, KERNEL))

    assert result.ran == CANONICAL_STEPS
    sizes = [argv[argv.index("--max_image_size") + 1]
             for argv in _calls(backend, "image_undistorter")
             if argv[argv.index("--output_path") + 1].endswith("/dense")]
    assert sizes == ["1600", str(FALLBACK_MAX_IMAGE_SIZE)]

    attempts = [gpu for argv, gpu in zip(backend.calls, backend.gpus)
                if argv[1] == "patch_match_stereo"]
    assert [gpu.uuid for gpu in attempts] == [GPU_5060, GPU_1650]

    logged = _log(session_dir)
    assert "no CUDA kernel for this card" in logged
    assert "This is the one retry." in logged
    assert "out of memory" not in logged

    gpu = _block(session_dir)["gpu"]
    assert gpu["uuid"] == GPU_1650 and gpu["fallback_used"] is True
    assert gpu["max_image_size"] == FALLBACK_MAX_IMAGE_SIZE
    assert State.read(session_dir).max_image_size == FALLBACK_MAX_IMAGE_SIZE


def test_oom_takes_the_same_fallback_with_its_own_log_line(tmp_path) -> None:
    """An exhausted card takes the same path and says a different thing.

    `CUDA error` matches both causes, so it is deliberately not the pattern:
    retrying an out-of-memory failure unchanged is not a fallback, and the two
    log lines are how an operator tells which one they are reading.
    """
    session_dir = _session(tmp_path)
    result, backend = _run(session_dir, _healing(session_dir, OOM))

    assert result.ran == CANONICAL_STEPS
    logged = _log(session_dir)
    assert "the card ran out of memory — deleting colmap/dense/" in logged
    assert "no CUDA kernel for this card" not in logged
    assert _block(session_dir)["gpu"]["reason"] == "the card ran out of memory"


def test_fallback_wipes_the_dense_workspace_and_reruns_masks_and_cfg(
        tmp_path) -> None:
    """The workspace is deleted whole and rebuilt, and the state file forgets
    what built it.

    All three of these were missing from the first design and all three are
    silent: masks undistorted at the old `newK`, a `patch-match.cfg` the
    undistorter regenerates over the host's rewrite, and depth maps
    `patch_match_stereo` skips because they already exist.
    """
    session_dir = _session(tmp_path)
    backend = _healing(session_dir, KERNEL)
    dense = session_dir / "colmap" / "dense"

    # A witness dropped into the tree after the first undistort: it survives
    # every later step, so finding it gone at the retry is proof the whole
    # directory was deleted rather than overwritten in place.
    witness = dense / "_witness"
    seen: list[bool] = []
    rebuild = backend.effects["image_undistorter"]

    def watch(argv: list[str]) -> None:
        if not argv[argv.index("--output_path") + 1].endswith("/dense"):
            rebuild(argv)
            return
        seen.append(witness.exists())
        rebuild(argv)
        witness.write_bytes(b"the first attempt was here")

    backend.effects["image_undistorter"] = watch
    _run(session_dir, backend)

    assert seen == [False, False]

    order = [argv[1] for argv in backend.calls]
    first, second = [i for i, tool in enumerate(order)
                     if tool == "patch_match_stereo"]
    assert "image_undistorter" in order[first:second]
    assert order.count(GPU_TOOL) == 1            # the probe is not repeated
    logged = _log(session_dir)
    assert logged.count("undistort_masks: 24 masks") == 2
    assert logged.count("write_patch_match_cfg: running") == 2

    assert set(FALLBACK_CLEARS) == {
        "image_undistorter", "undistort_masks", "write_patch_match_cfg",
        "patch_match_stereo", "stereo_fusion", "merge"}


def test_fallback_leaves_the_texture_workspace_alone(tmp_path) -> None:
    """`colmap/texture/`, `clean_images`, `colmap/images/`,
    `colmap/images_clean/` and `mesh/` all survive the wipe.

    The clean images are device-independent and expensive to rebuild, the
    texture workspace has nothing to do with the GPU, and `mesh/` is where
    `poisson_mesher` writes — which is why it is outside `colmap/dense/` in the
    first place.
    """
    session_dir = _session(tmp_path)
    _run(session_dir, to_step="texture_workspace")

    texture = session_dir / "colmap" / "texture"
    before = {p.relative_to(session_dir): p.read_bytes()
              for p in list(texture.rglob("*"))
              + list((session_dir / "colmap" / "images_clean").iterdir())
              + list((session_dir / "colmap" / "images").iterdir())
              if p.is_file()}
    clean_done = State.read(session_dir).steps["clean_images"]

    _run(session_dir, _healing(session_dir, KERNEL))

    after = {path: (session_dir / path).read_bytes() for path in before}
    assert after == before
    assert State.read(session_dir).steps["clean_images"] == clean_done
    assert (session_dir / "mesh" / "model.glb").is_file()


def test_only_one_fallback_attempt(tmp_path) -> None:
    """A second kernel error aborts by name rather than wiping again.

    A failure that matches neither pattern is not retried at all: nothing about
    a smaller workspace would help it, and an hour is too long to spend finding
    that out twice.
    """
    session_dir = _session(tmp_path)
    backend = _fake(session_dir, exits={"patch_match_stereo": 1},
                    output={"patch_match_stereo": KERNEL})
    with pytest.raises(StepFailed) as caught:
        _run(session_dir, backend)
    assert "the fallback attempt failed too" in str(caught.value)
    assert "--mode texture-only" in str(caught.value)
    assert len(_calls(backend, "patch_match_stereo")) == 2

    unrelated = _session(tmp_path / "unrelated")
    backend = _fake(unrelated, exits={"patch_match_stereo": 1},
                    output={"patch_match_stereo": ["Segmentation fault"]})
    with pytest.raises(StepFailed) as caught:
        _run(unrelated, backend)
    assert "neither a missing kernel nor an exhausted card" in str(caught.value)
    assert len(_calls(backend, "patch_match_stereo")) == 1
    # And nothing was wiped for it: the workspace is what the operator reads to
    # find out why.
    assert (unrelated / "colmap" / "dense" / "stereo"
            / "patch-match.cfg").is_file()


def test_single_gpu_kernel_error_aborts_naming_the_local_dockerfile(
        tmp_path) -> None:
    """One card and no kernel for it: the run stops and names the fix.

    Retrying the same missing kernel on the same card cannot help, however
    small the workspace is made, so the remedy is the image that has the
    kernel.
    """
    session_dir = _session(tmp_path)
    backend = _fake(session_dir, exits={"patch_match_stereo": 1},
                    output={"patch_match_stereo": KERNEL},
                    gpus=GPU_LIST[:1])
    with pytest.raises(StepFailed) as caught:
        _run(session_dir, backend)
    message = str(caught.value)
    assert "native/docker/colmap-cuda128" in message
    assert "this machine has one card" in message
    assert "--mode texture-only" in message
    assert len(_calls(backend, "patch_match_stereo")) == 1
    # Nothing was wiped: there was nowhere to retry.
    assert (session_dir / "colmap" / "dense" / "sparse" / "cameras.bin").is_file()


def test_single_gpu_oom_retries_once_on_the_same_card(tmp_path) -> None:
    """One card that ran out of memory retries on it, smaller — which is what
    the smaller size is for."""
    session_dir = _session(tmp_path)
    result, backend = _run(session_dir,
                           _healing(session_dir, OOM, single=True))

    assert result.ran == CANONICAL_STEPS
    attempts = [gpu.uuid for argv, gpu in zip(backend.calls, backend.gpus)
                if argv[1] == "patch_match_stereo"]
    assert attempts == [GPU_5060, GPU_5060]
    assert "the only card in this machine" in _log(session_dir)
    gpu = _block(session_dir)["gpu"]
    assert gpu["fallback_used"] is True and gpu["fallback_uuid"] == GPU_5060


# ── the depth range, only when the tracks are too thin ───────────────────


def test_depth_range_flags_are_absent_unless_the_fallback_asked_for_them(
        tmp_path) -> None:
    """COLMAP takes each image's depth range from the points it observes, and
    no global pair can beat that — so the flags are passed only when the tracks
    are demonstrably too thin, and the run then records whether they took
    effect.
    """
    session_dir = _session(tmp_path)
    _, backend = _run(session_dir)
    argv = backend.argv_for("patch_match_stereo")
    assert not any(part.startswith("--PatchMatchStereo.depth") for part in argv)
    assert _block(session_dir)["depth_range"] is None

    # Every image is "thin" when the bar is above any track count.
    thin = _session(tmp_path / "thin")
    result, backend = _run(thin, params=_params(
        cfg=CfgParams(min_observations=10 ** 9)))
    argv = backend.argv_for("patch_match_stereo")
    low = float(argv[argv.index("--PatchMatchStereo.depth_min") + 1])
    high = float(argv[argv.index("--PatchMatchStereo.depth_max") + 1])
    assert 0.0 < low < high
    assert any("observe fewer than" in w for w in result.warnings)

    # And what the depth maps came back with, beside what was asked for.
    measured = _block(thin)["depth_range"]
    assert measured["requested"] == pytest.approx([low, high], rel=1e-5)
    assert (measured["observed_min"], measured["observed_max"]) == (210.0, 480.0)


# ── the merge ────────────────────────────────────────────────────────────


def test_merge_refusal_aborts_with_the_merge_message(tmp_path) -> None:
    """A refused merge stops the run with `merge_clouds`' own sentence, and
    that sentence names `--mode texture-only`.

    Not `--to poisson_mesher`: in dense mode `merge` precedes it in the chain,
    so that escape would run the merge again and refuse again. The numbers are
    written to `session.json` before the refusal is raised — they are the point
    of a refused merge as much as of a passing one.
    """
    # A cloud floating inside the box: no laser point within any candidate
    # ball, so nothing is supported and there is no bias to measure.
    inside = np.random.default_rng(7).uniform(-40.0, 40.0, (3000, 3))
    inside[:, 2] = np.random.default_rng(8).uniform(60.0, 80.0, 3000)
    session_dir = _session(tmp_path)
    backend = _fake(session_dir, fused=inside,
                    fused_normals=np.tile([0.0, 0.0, 1.0], (len(inside), 1)))

    with pytest.raises(ReconRefused) as caught:
        _run(session_dir, backend)
    message = str(caught.value)
    assert "merge refused: no overlap" in message
    assert "--mode texture-only" in message
    assert "poisson_mesher" not in backend.tools

    block = _block(session_dir)["merge"]
    assert block["refused"] is True and block["refused_by"] == "G1b"
    assert block["message"] == message
    assert block["dense_supported"] == 0
    assert not (session_dir / "colmap" / "dense" / "merged.ply").exists()
    assert (session_dir / recon.LASER_PLY).is_file()
    assert (session_dir / "colmap" / "dense" / "fused.ply").is_file()


def test_merge_stats_and_gpu_land_in_session_json(tmp_path) -> None:
    """Every `MergeStats` field, every `MergeParams` threshold and the card that
    ran, in the `reconstruct` block — and `merged.ply` with its normals.

    Poisson is undefined without normals, and the merged cloud is what
    Milestone 2 meshes, so the normals are not optional decoration.
    """
    session_dir = _session(tmp_path)
    result, _ = _run(session_dir)
    block = _block(session_dir)
    merge = block["merge"]

    assert merge["params"] == {
        "support_mm": 8.0, "support_min": 5, "max_normal_mm": 10.0,
        "outlier_mm": 4.0, "density_mm": 1.5, "patch_frac": 0.3,
        "agree_mm": 1.5, "band_min": 500, "min_supported": 2000,
        "warn_frac": 0.25}
    assert merge["candidate_radius_mm"] == pytest.approx(float(np.hypot(8.0, 10.0)))
    assert merge["refused"] is False and merge["refused_by"] is None
    assert merge["laser_points"] == len(BOX_XYZ)
    assert merge["dense_points"] == len(BOX_XYZ) + len(_FILL)
    assert merge["dense_supported"] + merge["dense_kept"] + merge["floaters"] \
        == merge["dense_points"]
    # The patch 30 mm off the wall is what dense is for: the laser never saw it.
    assert merge["dense_kept"] == len(_FILL)
    assert merge["agree_frac"] > 0.9 and merge["keep_frac"] > 0.01
    assert merge["cells"]["grid"] == [4, 8] and merge["cells"]["gated"] >= 1
    assert merge["cells"]["worst"] in merge["cells"]["medians_mm"]
    for key in ("p90_abs_residual_mm", "kept_median_mm",
                "median_dense_neighbours"):
        assert isinstance(merge[key], float)

    assert block["gpu"] == {
        "requested": recon.DEFAULT_GPU_MATCH, "uuid": GPU_5060,
        "name": "NVIDIA GeForce RTX 5060 Ti", "fallback_used": False,
        "fallback_uuid": None, "reason": None, "max_image_size": 1600}
    assert block["disk"]["estimate_bytes_rechecked"] > 0

    merged = session_dir / "colmap" / "dense" / "merged.ply"
    xyz, _, normals = recon.read_ply_full(str(merged))
    assert len(xyz) == merge["laser_points"] + merge["dense_kept"]
    assert normals is not None and len(normals) == len(xyz)

    line = next(ln for ln in _log(session_dir).splitlines()
                if "merge: supported" in ln)
    assert "candidate radius 12.81 mm" in line
    assert result.degraded == []


# ── switching modes, with the real steps behind it ───────────────────────


def test_switching_mode_invalidates_the_shared_tail_both_directions(
        tmp_path) -> None:
    """`texture_workspace` is the first step whose input directory depends on
    the mode, so a `--mode` change invalidates it and everything after it — in
    both directions, with the real steps behind the rule this time.

    Asserting only the mesh tail would pass while the workspace stayed built
    from the wrong pixels, which is precisely the bug the rule exists for.
    """
    session_dir = _session(tmp_path)
    _run(session_dir, mode="texture-only")
    assert not (session_dir / "colmap" / "images_clean").exists()

    result, backend = _run(session_dir, mode="dense")
    assert "write_sparse" in result.skipped and "validate_sparse" in result.skipped
    assert "texture_workspace" in result.ran and "clean_images" in result.ran
    for argv in _calls(backend, "image_undistorter"):
        assert argv[3] == "/data/colmap/images_clean"
    assert backend.argv_for("poisson_mesher")[3] == "/data/colmap/dense/merged.ply"

    line = next(ln for ln in _log(session_dir).splitlines()
                if "mode changed texture-only -> dense" in ln)
    assert "invalidating texture_workspace and the 3 steps after it" in line

    # ...and back, where the workspace has to be rebuilt from the verbatim
    # pixels even though every file it needs is already on disk.
    result, backend = _run(session_dir, mode="texture-only")
    assert "texture_workspace" in result.ran
    assert backend.argv_for("image_undistorter")[3] == "/data/colmap/images"
    assert backend.argv_for("poisson_mesher")[3] == "/data/laser.ply"
    assert "dense -> texture-only" in _log(session_dir)


def test_texture_source_is_inpainted_when_the_clean_set_falls_back(
        tmp_path) -> None:
    """Below either texture gate the whole selection is textured from — and in
    dense mode the laser photographs in it carry **inpainted** pixels, because
    the texture workspace is undistorted from `colmap/images_clean/`.

    This is the one named exception to "inpainted pixels never reach the
    atlas". It is accepted rather than engineered around: those exact images
    were already judged fit for PatchMatch, the warning names what the clean
    set missed, and the remedy — a top-up clean pass and a texture-only re-run
    — costs minutes. A degraded run is not a pass, and the label is what says
    so afterwards.
    """
    session_dir = _session(tmp_path, clean_places=PLACES[:4])
    result, backend = _run(session_dir)

    assert result.texture_source == "inpainted"
    assert result.degraded == ["texture_source=inpainted"]
    assert any("texture_source=inpainted" in w for w in result.warnings)
    assert _listed(session_dir) == _selected(session_dir)

    texture = next(argv for argv in _calls(backend, "image_undistorter")
                   if argv[argv.index("--output_path") + 1].endswith("/texture"))
    assert texture[3] == "/data/colmap/images_clean"
    assert _block(session_dir)["texture_source"] == "inpainted"


def test_from_image_undistorter_is_refused_until_clean_images_has_run(
        tmp_path) -> None:
    """The refusal D1a built, with the real `clean_images` behind it: starting
    after that step without one on record would undistort a directory that is
    missing or left over from a run whose selection was different."""
    session_dir = _session(tmp_path)
    with pytest.raises(ReconRefused, match="no finished clean_images"):
        _run(session_dir, from_step="image_undistorter")

    _run(session_dir, to_step="clean_images")
    result, _ = _run(session_dir, from_step="image_undistorter")
    assert result.ran[0] == "image_undistorter"
    assert result.ran[-1] == "glb"
