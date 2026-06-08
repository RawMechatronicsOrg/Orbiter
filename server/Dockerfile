FROM python:3.11-slim

# Keep the image small; uvicorn-standard pulls in uvloop + httptools which
# already need a couple of build tools at install time. We install them,
# pip-install, then drop them again.
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml ./
COPY orbiter_server ./orbiter_server

RUN pip install --no-cache-dir . \
    && apt-get purge -y gcc \
    && apt-get autoremove -y

ENV ORBITER_STORAGE_DIR=/data

EXPOSE 8000

CMD ["uvicorn", "orbiter_server.app:app", "--host", "0.0.0.0", "--port", "8000"]
