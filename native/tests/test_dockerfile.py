"""Text checks on native/docker/colmap-cuda128/Dockerfile — no Docker daemon
needed. The photogrammetry plan names this from-source build as the expected
route to the RTX 5060 Ti (sm_120): the public 'colmap/colmap:latest' image's
CUDA kernels stop at sm_90 SASS plus compute_90 PTX, and sm_120 support did
not exist in nvcc before CUDA 12.8. These are the cheap, offline half of the
acceptance test — the build itself is a 40-60 minute background job the
operator runs by hand, not something a test suite invokes."""

from __future__ import annotations

import re
from pathlib import Path

DOCKERFILE = Path(__file__).resolve().parent.parent / "docker" / "colmap-cuda128" / "Dockerfile"


def _text() -> str:
    assert DOCKERFILE.is_file(), f"missing {DOCKERFILE}"
    return DOCKERFILE.read_text(encoding="utf-8")


def test_dockerfile_pins_the_colmap_tag_and_sets_sm_120() -> None:
    text = _text()

    # Pinned to the exact release every flag and default elsewhere in the
    # plan was read off — a later tag could silently change one.
    assert "ARG COLMAP_TAG=4.2.0" in text

    # sm_120 (RTX 5060 Ti / Blackwell) does not exist before CUDA 12.8 —
    # every 'FROM nvidia/cuda:<version>-...' base must clear that floor.
    cuda_bases = re.findall(r"nvidia/cuda:(\d+)\.(\d+)(?:\.\d+)?-", text)
    assert cuda_bases, "no 'FROM nvidia/cuda:<version>-...' base image found"
    for major, minor in cuda_bases:
        assert (int(major), int(minor)) >= (12, 8), (major, minor)

    # The three architectures this image is built for: 75 (GTX 1650 SUPER,
    # the fallback card), 89 (Ada), 120 (RTX 5060 Ti — the card the public
    # image cannot run PatchMatch stereo on without an unverified driver
    # JIT).
    arch_flags = re.findall(r'CMAKE_CUDA_ARCHITECTURES="([0-9;]+)"', text)
    assert arch_flags, "no -DCMAKE_CUDA_ARCHITECTURES=\"...\" flag found"
    architectures = set(arch_flags[0].split(";"))
    assert {"75", "89", "120"} <= architectures, architectures
