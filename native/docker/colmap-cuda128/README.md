# `orbiter/colmap:cuda129-sm120` — COLMAP with real sm_120 kernels

This is the **expected route to the RTX 5060 Ti**, not a contingency. The public
`colmap/colmap:latest` image on this machine is COLMAP 4.2.0 built against CUDA
12.9.1, but its CUDA binary embeds SASS only for sm_50/60/70/75/90, plus
forward-compatible PTX for compute_90. The RTX 5060 Ti is **sm_120**, which did
not exist when those kernels were compiled. Whether `patch_match_stereo` runs
on it at all through the public image therefore depends on the driver
JIT-compiling the compute_90 PTX forward to sm_120 the first time it launches —
plausible, unverified, and slow on that first launch. This image removes the
gamble: COLMAP is compiled from source with real machine code for sm_120,
alongside sm_75 (the GTX 1650 SUPER, the fallback card) and sm_89 (Ada, so a
future Ada card needs no rebuild either).

The directory is named `colmap-cuda128` for the *requirement* — CUDA >= 12.8 is
the first toolkit whose `nvcc` knows sm_120 exists at all — not for the exact
version pinned in the Dockerfile, which tracks the currently-pulled public
image's CUDA (12.9.1) so the two builds stay comparable.

## Build

```
docker build -t orbiter/colmap:cuda129-sm120 native/docker/colmap-cuda128
```

Expect **40-60 minutes**. COLMAP and Ceres both compile from source, and `nvcc`
compiles every CUDA kernel three times over (once per architecture: 75, 89,
120). This is a background job the operator runs by hand — it is not part of
any test or CI gate, and nothing in the runner requires this image to exist
before Milestone 1 (which needs no GPU at all) or before a first Milestone 2
run against the public image.

## Pointing the runner at it

The runner's `DockerBackend` reads `ORBITER_COLMAP_IMAGE` and defaults to the
public `colmap/colmap:latest`. Once this image is built, point at it instead:

```
ORBITER_COLMAP_IMAGE=orbiter/colmap:cuda129-sm120
```

set in the shell (or the sibling `docker/.env`) before running
`orbiter-recon --mode dense`. Everything else — session layout, `--gpus
device=GPU-<uuid>`, the JIT-cache volume, `QT_QPA_PLATFORM=offscreen` — is
unchanged; only the image name differs.

## What is pinned, and why

- `ARG COLMAP_TAG=4.2.0` — the exact release every flag, default and option
  name in the photogrammetry plan (`native/docs/PHOTOGRAMMETRY.md`) was read
  off. A later tag could silently change one of those defaults.
- `nvidia/cuda:12.9.1-devel-ubuntu24.04` (builder) /
  `nvidia/cuda:12.9.1-runtime-ubuntu24.04` (runtime) — matches the public
  image's own CUDA version. The hard floor is **12.8**, not 12.9.1
  specifically: that is the first CUDA release whose `nvcc` targets sm_120 at
  all.
- `-DCMAKE_CUDA_ARCHITECTURES="75;89;120"` — sm_75 (GTX 1650 SUPER, the
  fallback card already in this machine), sm_89 (Ada), sm_120 (RTX 5060 Ti).

## Dependency list

The build-stage package list follows the shape of COLMAP's own
`docker/Dockerfile` (github.com/colmap/colmap) — reproduced from training
knowledge, not fetched from the 4.2.0 tag itself. This box has no route to
GitHub during the build, and the shipped `colmap/colmap:latest` image does not
carry its own build recipe or a dependency manifest as a doc file inside the
image, so there was nothing to inspect from a running container. Package names
target Ubuntu 24.04; Qt6 is used rather than Qt5 because 24.04's archive
treats Qt6 as the primary Qt package and COLMAP's CMakeLists has preferred Qt6
(falling back to Qt5) for several releases before this one. If 4.2.0 disagrees,
CMake's configure step fails immediately and by name — it does not silently
misbuild.

The runtime stage installs the same `-dev` packages as the builder rather than
upstream's narrower, version-pinned runtime package list (e.g.
`libboost-filesystem1.74.0`) — a wrong minor-version guess there would only
surface as an `apt-get install` failure *after* the 40-60 minute builder stage
has already finished. The larger runtime image is the trade made for that.
