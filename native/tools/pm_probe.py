"""PatchMatch probe: does a given COLMAP image run dense stereo on a given GPU?

Renders a textured plane (400 x 400 mm at z = 0 of a board-like world frame)
from eight known camera poses, writes the COLMAP 4 text model with our own
colmapio writers (cameras / images / points3D / rigs / frames), then drives
image_undistorter -> patch_match_stereo -> stereo_fusion inside the container
and fits a plane to fused.ply. The pass criterion is geometric: the fused
points must lie on z = 0 to well under a millimetre, which only happens when
PatchMatch actually ran its CUDA kernels on that card.

Usage:
  python pm_probe.py --image colmap/colmap:latest --gpu GPU-<uuid> --out <dir>
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from orbiter_native import colmapio
from orbiter_native.scan import read_ply_full

W, H = 1920, 1080
FX = FY = 1400.0
CX, CY = 960.0, 540.0
PLANE_MM = 400.0          # the plane spans [-200, 200] mm in x and y at z = 0
TEX = 2048                # texture pixels per plane edge


def texture(rng: np.random.Generator) -> np.ndarray:
    """A speckle-and-blotch texture that gives PatchMatch something to match
    at every scale: low-frequency blotches so wide windows agree, fine
    speckle so the depth is pinned to the pixel."""
    coarse = cv2.resize(rng.random((64, 64)).astype(np.float32), (TEX, TEX),
                        interpolation=cv2.INTER_CUBIC)
    mid = cv2.resize(rng.random((256, 256)).astype(np.float32), (TEX, TEX),
                     interpolation=cv2.INTER_CUBIC)
    fine = rng.random((TEX, TEX)).astype(np.float32)
    g = 0.45 * coarse + 0.35 * mid + 0.20 * fine
    g = (g - g.min()) / (g.max() - g.min() + 1e-9)
    bgr = np.stack([g * 0.9, g, g * 0.8], axis=-1)
    return np.clip(bgr * 255.0, 0, 255).astype(np.uint8)


def look_at(centre: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Board->camera (R, t) for a camera at `centre` looking at `target`,
    OpenCV axes: +z forward, +y down, +x right."""
    z = target - centre
    z /= np.linalg.norm(z)
    up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(z, up)) > 0.99:
        up = np.array([0.0, 1.0, 0.0])
    x = np.cross(-up, z)            # y points down, so "up" enters with a minus
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])          # rows: camera axes expressed in the world
    t = -R @ centre
    return R, t


def render(tex: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray,
           rng: np.random.Generator) -> np.ndarray:
    """Warp the plane texture into the camera: texture px -> plane mm -> image."""
    # texture pixel (u, v) -> plane point (x, y, 0): x = u*s - 200, y = 200 - v*s
    s = PLANE_MM / TEX
    A = np.array([[s, 0.0, -PLANE_MM / 2], [0.0, -s, PLANE_MM / 2], [0.0, 0.0, 1.0]])
    P = K @ np.column_stack([R[:, 0], R[:, 1], t])   # plane (x, y, 1) -> image
    Hm = P @ A
    img = cv2.warpPerspective(tex, Hm, (W, H), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(40, 40, 40))
    noise = rng.normal(0.0, 1.5, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def poses(n: int = 8) -> list[tuple[np.ndarray, np.ndarray]]:
    out = []
    for i in range(n):
        a = 2 * np.pi * i / n
        centre = np.array([150.0 * np.cos(a), 150.0 * np.sin(a), 420.0 + 15.0 * np.sin(2 * a)])
        out.append(look_at(centre, np.array([0.0, 0.0, 0.0])))
    return out


def write_model(out: Path, K: np.ndarray, cams: list[tuple[np.ndarray, np.ndarray]]) -> None:
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "sparse").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    tex = texture(rng)
    eye = colmapio.EyeCamera(camera_id=1, side="left", width=W, height=H, fx=FX, fy=FY, cx=CX, cy=CY,
                             dist=(0.0, 0.0, 0.0, 0.0, 0.0), photo_wh=((W, H),))
    images = []
    for i, (R, t) in enumerate(cams, start=1):
        name = f"view_{i:02d}.jpg"
        img = render(tex, K, R, t, rng)
        cv2.imwrite(str(out / "images" / name), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        images.append(colmapio.ImageRecord(image_id=i, name=name, camera_id=1, R=R, t=t, points2d=[]))
    # Seeds with tracks — the same shape the real chain writes from the laser
    # cloud. stereo_fusion decides which images overlap from the sparse
    # model's shared points, so with an empty points3D.txt it fuses nothing
    # (measured: 0 points from perfectly good depth maps). A 10 mm grid on
    # the plane, projected into every view that sees it, is enough.
    points = []
    g = np.arange(-190.0, 191.0, 10.0)
    pid = 0
    for x in g:
        for y in g:
            X = np.array([x, y, 0.0])
            track = []
            for img_rec in images:
                pc = img_rec.R @ X + img_rec.t
                if pc[2] <= 0:
                    continue
                u, v = FX * pc[0] / pc[2] + CX, FY * pc[1] / pc[2] + CY
                if 0 <= u < W and 0 <= v < H:
                    track.append((img_rec.image_id, len(img_rec.points2d)))
                    img_rec.points2d.append((float(u), float(v), pid + 1))
            if len(track) >= 2:
                pid += 1
                points.append(colmapio.Point3D(point_id=pid, xyz=X, rgb=(128, 128, 128), error=1.0, track=track))
            else:
                for img_id, idx in track:      # roll back a lonely observation
                    images[img_id - 1].points2d.pop(idx)
    print(f"seeds {len(points)}, observations {sum(len(r.points2d) for r in images)}")
    colmapio.write_cameras(out / "sparse" / "cameras.txt", [eye])
    colmapio.write_images(out / "sparse" / "images.txt", images)
    colmapio.write_points3d(out / "sparse" / "points3D.txt", points)
    colmapio.write_rigs(out / "sparse" / "rigs.txt", [eye])
    colmapio.write_frames(out / "sparse" / "frames.txt", images)


def docker(image: str, out: Path, args: list[str], gpu: str | None, log: list[str]) -> int:
    cmd = ["docker", "run", "--rm", "-v", f"{out.resolve().as_posix()}:/data"]
    if gpu:
        cmd += ["--gpus", f"device={gpu}", "-v", "orbiter-colmap-jit:/jitcache",
                "-e", "CUDA_CACHE_PATH=/jitcache", "-e", "CUDA_CACHE_MAXSIZE=1073741824"]
    cmd += [image] + args
    log.append("$ " + " ".join(cmd))
    t0 = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    log.append(p.stdout[-6000:] + p.stderr[-6000:])
    log.append(f"exit {p.returncode} in {time.perf_counter() - t0:.1f}s")
    return p.returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--gpu", default=None, help="GPU-<uuid>; omit for the CPU-only steps")
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-render", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1.0]])
    cams = poses()
    if not a.skip_render:
        write_model(out, K, cams)
    log: list[str] = []
    rc = docker(a.image, out, ["colmap", "model_converter", "--input_path", "/data/sparse",
                               "--output_path", "/data/_validate.ply", "--output_type", "PLY"], None, log)
    print("\n".join(log[-3:]))
    if rc:
        print("MODEL REJECTED"); return 2
    log.clear()
    rc = docker(a.image, out, ["colmap", "image_undistorter", "--image_path", "/data/images",
                               "--input_path", "/data/sparse", "--output_path", "/data/dense",
                               "--output_type", "COLMAP", "--max_image_size", "1920"], None, log)
    print(log[-1])
    if rc:
        print("\n".join(log)); return 3
    # explicit sources: every other view
    names = [f"view_{i:02d}.jpg" for i in range(1, len(cams) + 1)]
    cfg = out / "dense" / "stereo" / "patch-match.cfg"
    lines = []
    for n in names:
        lines += [n, ", ".join(m for m in names if m != n)]
    cfg.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    log.clear()
    rc = docker(a.image, out, ["colmap", "patch_match_stereo", "--workspace_path", "/data/dense",
                               "--workspace_format", "COLMAP",
                               "--PatchMatchStereo.geom_consistency", "true",
                               "--PatchMatchStereo.num_iterations", "5",
                               "--PatchMatchStereo.depth_min", "200", "--PatchMatchStereo.depth_max", "700",
                               "--PatchMatchStereo.gpu_index", "0"], a.gpu, log)
    tail = log[-2][-3000:]
    print(log[-1])
    if rc:
        m = re.findall(r"(no kernel image|invalid device function|out of memory|CUDA error[^\n]*)", tail)
        print("PATCHMATCH FAILED:", m or tail[-800:]); return 4
    log.clear()
    rc = docker(a.image, out, ["colmap", "stereo_fusion", "--workspace_path", "/data/dense",
                               "--workspace_format", "COLMAP", "--input_type", "geometric",
                               "--StereoFusion.max_reproj_error", "2", "--StereoFusion.max_depth_error", "0.01",
                               "--StereoFusion.min_num_pixels", "3",
                               "--output_path", "/data/dense/fused.ply"], None, log)
    print(log[-1])
    print("fusion said:", log[-2][-2500:])
    if rc:
        print("\n".join(log)); return 5
    xyz, rgb, nrm = read_ply_full(str(out / "dense" / "fused.ply"))
    inside = (np.abs(xyz[:, 0]) < 180) & (np.abs(xyz[:, 1]) < 180)
    z = xyz[inside, 2]
    print(f"fused points {len(xyz)}, on-plane region {inside.sum()}; "
          f"z median {np.median(z):+.3f} mm, MAD {np.median(np.abs(z - np.median(z))):.3f} mm, "
          f"p95 |z| {np.percentile(np.abs(z), 95):.3f} mm")
    ok = len(xyz) > 100_000 and np.percentile(np.abs(z), 95) < 1.5
    print("PATCHMATCH OK on this GPU" if ok else "PATCHMATCH RAN BUT THE GEOMETRY IS OFF")
    return 0 if ok else 6


if __name__ == "__main__":
    sys.exit(main())
