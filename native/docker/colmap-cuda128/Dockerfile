# Orbiter — COLMAP built from source, with real sm_120 kernels.
#
# WHY THIS EXISTS: the public 'colmap/colmap:latest' image already pulled
# onto this machine is COLMAP 4.2.0 built against CUDA 12.9.1 — but its CUDA
# binary embeds SASS only for sm_50/60/70/75/90, plus forward-compatible PTX
# for compute_90 (verified: docker/Dockerfile's sibling `colmap/Dockerfile`
# runs on that same pulled image). The RTX 5060 Ti is sm_120, which did not
# exist when those kernels were compiled. Whether it can run PatchMatch
# stereo at all then depends on the driver JIT-compiling the compute_90 PTX
# forward to sm_120 the first time it launches — plausible, unverified, and
# the compiled result is thrown away by '--rm' unless the JIT cache volume
# is mounted (the runner does this, but it only pays off across runs, not
# within a first one). This image is the answer that does not depend on
# that gamble: COLMAP compiled with real sm_120 machine code baked in,
# alongside sm_75 (the GTX 1650 SUPER — this box's fallback card, so it
# keeps working if this image is ever used for it too) and sm_89 (Ada, so a
# future Ada card needs no rebuild either).
#
# The directory is named 'colmap-cuda128' for the *requirement* this image
# satisfies — CUDA >= 12.8 is the earliest toolkit whose nvcc knows sm_120
# exists — not for the exact version pinned below, which tracks whatever
# CUDA the currently-pulled public image uses (12.9.1), so the two stay
# comparable.
#
# Build:  docker build -t orbiter/colmap:cuda129-sm120 native/docker/colmap-cuda128
#         Expect 40-60 minutes — COLMAP and Ceres both compile from source,
#         and nvcc compiles the CUDA kernels three times over (75, 89, 120).
#         Run it as a background job; it is not part of any test or CI gate.
# Run:    point the runner at this image instead of the public one:
#             ORBITER_COLMAP_IMAGE=orbiter/colmap:cuda129-sm120
#         (native/orbiter_native/recon.py's DockerBackend reads that
#         variable; its default is 'colmap/colmap:latest'.)
# See also: native/docker/colmap-cuda128/README.md.

# ---------------------------------------------------------------------------
# Builder stage — compiles COLMAP from source against the 'devel' image,
# which carries nvcc and the CUDA headers the 'runtime' image deliberately
# omits to stay small. 12.9.1 matches the pulled public image so the two
# builds are otherwise comparable; the requirement that actually matters
# here is only "at least 12.8".
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.9.1-devel-ubuntu24.04 AS builder

# Pinned to the exact release every flag and default in the photogrammetry
# plan (native/docs/PHOTOGRAMMETRY.md) was read off against — a later tag
# could silently change a default this whole plan depends on.
ARG COLMAP_TAG=4.2.0

# Matches upstream CUDA images' own build convention: apt would otherwise
# stop mid-install asking for a timezone on a base image that has none
# configured, which would hang the 40-60 minute build on an unattended box.
ENV DEBIAN_FRONTEND=noninteractive

# Build dependencies, following the shape of COLMAP's own 'docker/Dockerfile'
# build-stage package list (github.com/colmap/colmap) — reproduced here from
# training knowledge, not fetched from the 4.2.0 tag itself: this box has no
# route to GitHub during the build, and the shipped 'colmap/colmap:latest'
# image does not carry its own build recipe or dependency manifest as a
# runtime doc file, so there is nothing to 'docker run ... cat' to check it
# against. Names below are Ubuntu 24.04 packages. No Qt at all: the GUI is
# switched off at configure time (-DGUI_ENABLED=OFF below), the runner only
# ever calls the CLI, and COLMAP loads images through FreeImage, so Qt was
# the one dependency whose 4.2.0 configure requirements could not be
# verified offline and it is simply not needed. OpenBLAS/LAPACK are for the
# FAISS nearest-neighbour library that COLMAP 3.12+ fetches and builds
# during configure (FetchContent, so the build needs network — as does the
# git clone below). This list is not claimed to be exhaustive — only
# complete enough for the CLI COLMAP the runner actually invokes — and apt
# pulls in whatever transitive dependency (mesa/OpenGL under libglew-dev,
# zlib/libjpeg/libpng under libfreeimage-dev, etc.) each of these needs.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        cmake \
        ninja-build \
        build-essential \
        libboost-program-options-dev \
        libboost-filesystem-dev \
        libboost-graph-dev \
        libboost-system-dev \
        libeigen3-dev \
        libopenimageio-dev \
        openimageio-tools \
        libmetis-dev \
        libgoogle-glog-dev \
        libgflags-dev \
        libgtest-dev \
        libgmock-dev \
        libsqlite3-dev \
        libglew-dev \
        libopenblas-dev \
        liblapack-dev \
        libcgal-dev \
        libceres-dev \
        libcurl4-openssl-dev \
        libssl-dev \
 && rm -rf /var/lib/apt/lists/*

# Ubuntu's OpenImageIO CMake config insists on OpenCV's include directory
# existing even though nothing here uses OIIO's OpenCV bridge — the same
# one-line workaround COLMAP's own docker/Dockerfile carries for 24.04.
RUN mkdir -p /usr/include/opencv4

# Shallow, tag-only clone — this is a build input pinned by COLMAP_TAG, not
# a working checkout anyone here edits, so there is no reason to fetch
# history nobody will look at.
RUN git clone --branch "${COLMAP_TAG}" --depth 1 \
        https://github.com/colmap/colmap.git /colmap-src

# CMAKE_CUDA_ARCHITECTURES is the whole point of this image: 75 is the GTX
# 1650 SUPER (this box's fallback card), 89 is Ada, and 120 is the RTX 5060
# Ti — the card the public image cannot run PatchMatch stereo on without an
# unverified driver JIT. Ninja over Make purely for build speed on a
# from-source compile this large; Release because a debug COLMAP build is
# both far slower to run and not what any acceptance test in the plan
# depends on.
RUN cmake -S /colmap-src -B /colmap-build -GNinja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES="75;89;120" \
        -DCUDA_ENABLED=ON \
        -DGUI_ENABLED=OFF \
        -DOPENGL_ENABLED=OFF \
        -DONNX_ENABLED=OFF \
        -DTESTS_ENABLED=OFF \
        -DCMAKE_INSTALL_PREFIX=/colmap-installed \
 && cmake --build /colmap-build \
 && cmake --install /colmap-build

# ---------------------------------------------------------------------------
# Runtime stage — the 'devel' image's nvcc and headers are dead weight once
# COLMAP is compiled; the CUDA 'runtime' image carries only the driver
# libraries a compiled binary actually calls into, which is most of what
# made the public image 3.17 GB.
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.9.1-runtime-ubuntu24.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive

# The shared libraries COLMAP links against at runtime. Upstream's own
# Dockerfile pins exact versioned runtime packages here (e.g.
# 'libboost-filesystem1.74.0') for a smaller image, but that ties this file
# to whatever Boost/Qt/Ceres minor versions happen to be current in Ubuntu
# 24.04's archive on the day it is built — get one version number wrong and
# 'apt-get install' fails here, after the 40-60 minute builder stage has
# already finished. Installing the same unversioned '-dev' packages the
# builder stage used costs a larger runtime image but guarantees the names
# resolve, and apt pulls in whatever runtime .so each one actually needs.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libboost-program-options-dev \
        libboost-filesystem-dev \
        libboost-graph-dev \
        libboost-system-dev \
        libopenimageio-dev \
        libmetis-dev \
        libgoogle-glog-dev \
        libgflags-dev \
        libsqlite3-dev \
        libglew-dev \
        libopenblas-dev \
        liblapack-dev \
        libcgal-dev \
        libceres-dev \
        libcurl4-openssl-dev \
        libssl-dev \
 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /colmap-installed /usr/local

# No Qt in this image (the GUI is compiled out), so nothing here goes
# looking for an X server and the QT_QPA_PLATFORM the public image needs
# has nothing to switch off.

# Sessions are bind-mounted here, matching every other Orbiter container
# (docker/docker-compose.yml's './data:/data') — a session path is the same
# regardless of which COLMAP image ran it.
WORKDIR /data

# Clear any ENTRYPOINT the base CUDA image sets and default CMD to a shell —
# the same convention native/../colmap/Dockerfile uses for the public
# image — so 'docker run orbiter/colmap:cuda129-sm120 colmap ...' runs the
# command directly instead of through an inherited entrypoint, while a bare
# 'docker run ... orbiter/colmap:cuda129-sm120' still lands in a shell.
ENTRYPOINT []
CMD ["bash"]
