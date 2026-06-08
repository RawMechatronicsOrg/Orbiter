# Orbiter v0.1 — COLMAP container.
#
# Base: the official 'colmap/colmap:latest' image (CUDA-enabled COLMAP
# build from upstream). We layer Python 3 + a small wrapper script and a
# JSON-to-cameras.txt converter on top.
#
# Build:    docker compose --profile colmap build colmap
# Run:      docker compose --profile colmap run --rm colmap run_colmap_session.sh <sid>

FROM colmap/colmap:latest

# python3 is needed for sfm_priors_to_colmap.py. ca-certificates is handy
# if we ever shell out to HTTPS for downloading auxiliary data. Keep the
# install slim — the base image is already large.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        ca-certificates \
        bash \
 && rm -rf /var/lib/apt/lists/*

# Wrapper scripts. Both go on PATH so they can be invoked without an
# absolute path from inside the container.
COPY run_colmap_session.sh /usr/local/bin/run_colmap_session.sh
COPY sfm_priors_to_colmap.py /usr/local/bin/sfm_priors_to_colmap.py
RUN chmod +x /usr/local/bin/run_colmap_session.sh \
 && chmod +x /usr/local/bin/sfm_priors_to_colmap.py

# /data is the bind-mount target from docker-compose (./data:/data on the
# host). Sessions live at /data/scans/<sid>/.
WORKDIR /data

# The base image's entrypoint already drops you near the colmap binary;
# we use bash as CMD so 'docker compose run colmap run_colmap_session.sh'
# works while 'docker compose run colmap' lands in an interactive shell.
ENTRYPOINT []
CMD ["bash"]
