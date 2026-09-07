"""Normals for Poisson, and the PLY that carries them.

PCA gives a normal's line; only the cameras give it a direction, and Poisson
needs the direction. The reader half is the other side of the same coin:
COLMAP's `fused.ply` is binary little-endian with normals between the
coordinates and the colours, and a reader that only ever saw our own files
would have discovered that on the first live run.
"""

from __future__ import annotations

import inspect
import struct

import numpy as np

from orbiter_native.scan import (
    normals_pca,
    orient_normals,
    read_ply,
    read_ply_full,
    write_ply,
)


def _plane(half: float = 10.0, step: float = 1.0) -> np.ndarray:
    """A flat sheet in z = 0, sampled `step` apart."""
    g = np.arange(-half, half + step / 2, step)
    x, y = np.meshgrid(g, g)
    return np.column_stack([x.ravel(), y.ravel(), np.zeros(x.size)])


def _sphere(n: int = 600, radius: float = 20.0) -> tuple[np.ndarray, np.ndarray]:
    """`n` points spread evenly over a sphere by the golden angle, with the
    outward unit direction of each — the answer `orient_normals` must find."""
    i = np.arange(n) + 0.5
    polar = np.arccos(1.0 - 2.0 * i / n)
    azimuth = np.pi * (1.0 + 5.0 ** 0.5) * i
    u = np.column_stack([np.cos(azimuth) * np.sin(polar),
                         np.sin(azimuth) * np.sin(polar),
                         np.cos(polar)])
    return u * radius, u


def test_normals_on_a_plane_are_the_plane_normal() -> None:
    n = normals_pca(_plane(), k=16)
    assert n.shape == (441, 3)
    # The sign is arbitrary before orientation; the line is not.
    assert np.abs(np.abs(n[:, 2]) - 1.0).max() < 1e-6


def test_normals_on_a_sphere_point_outward_after_orientation() -> None:
    pts, outward = _sphere()
    raw = normals_pca(pts, k=16)
    # PCA alone leaves the signs scattered — that is the whole reason
    # `orient_normals` exists.
    assert np.einsum("ij,ij->i", raw, outward).min() < 0.0
    cams, _ = _sphere(26, radius=100.0)
    oriented = orient_normals(raw, pts, cams)
    assert np.einsum("ij,ij->i", oriented, outward).min() > 0.0
    # Only the sign changed.
    assert np.abs(np.abs(oriented) - np.abs(raw)).max() < 1e-12


def test_write_read_round_trip_with_normals_and_colours(tmp_path) -> None:
    pts = np.array([[1.0, 2.0, 3.0], [-4.0, 5.5, 6.25]])
    nrm = np.array([[0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
    rgb = np.array([[10, 20, 30], [200, 210, 220]], np.uint8)
    path = tmp_path / "cloud.ply"
    assert write_ply(str(path), pts, rgb, normals=nrm) == 2
    got, got_rgb, got_n = read_ply_full(str(path))
    assert np.allclose(got, pts) and got_rgb.tolist() == rgb.tolist()
    assert np.allclose(got_n, nrm)
    # Normals alone, no colours: the other combination the merge writes.
    bare = tmp_path / "bare.ply"
    write_ply(str(bare), pts, normals=nrm)
    got, got_rgb, got_n = read_ply_full(str(bare))
    assert got_rgb is None and np.allclose(got_n, nrm)


def test_read_ply_full_reads_a_binary_little_endian_fused_ply(tmp_path) -> None:
    # Hand-packed rather than written by `write_ply`: this is COLMAP's file,
    # `x y z nx ny nz red green blue`, and the point is to read it without
    # having produced it. ASCII would test a file COLMAP never writes.
    rows = [(1.0, 2.0, 3.0, 0.0, 0.0, 1.0, 255, 0, 0),
            (4.0, 5.0, 6.0, 0.0, 1.0, 0.0, 0, 128, 64)]
    header = ("ply\n"
              "format binary_little_endian 1.0\n"
              f"element vertex {len(rows)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property float nx\nproperty float ny\nproperty float nz\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\n"
              "end_header\n")
    path = tmp_path / "fused.ply"
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        for r in rows:
            f.write(struct.pack("<6f3B", *r))
    xyz, rgb, nrm = read_ply_full(str(path))
    assert xyz.tolist() == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    assert nrm.tolist() == [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]
    assert rgb.tolist() == [[255, 0, 0], [0, 128, 64]]


def test_existing_read_ply_signature_unchanged(tmp_path) -> None:
    assert list(inspect.signature(read_ply).parameters) == ["path"]
    path = tmp_path / "old.ply"
    # The call every existing caller makes: positional, two values back.
    write_ply(str(path), np.array([[1.0, 2.0, 3.0]]), np.array([[9, 8, 7]], np.uint8))
    xyz, rgb = read_ply(str(path))
    assert xyz.tolist() == [[1.0, 2.0, 3.0]] and rgb.tolist() == [[9, 8, 7]]
