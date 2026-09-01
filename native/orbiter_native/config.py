"""Client for the Orbiter server's configuration.

The server stays the single owner of state (the project's Viser pattern); this
app is another client of it, one that happens to be able to touch pixels. It
reads `GET /config`, which returns exactly `PERSISTED_FIELDS` — and that
already includes `stereo_rig` (which camera is which eye, per-eye orientation,
the nominal baseline) plus the ChArUco board params. One documented endpoint
covers everything this workbench needs.

Polling rather than a WebSocket: the payload is about a kilobyte, so a poll
every couple of seconds costs nothing and reflects a change made in the web
Stereo tab within one interval. A `/ws/scene` client would buy sub-second
latency on a settings change that a human is making by hand — not worth a
second protocol implementation here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import httpx

from .cvcore import BoardSpec, Intrinsics, board_spec_from_config, intrinsics_from_eye
from .orient import Orientation

log = logging.getLogger("orbiter_native.config")

DEFAULT_SERVER = "http://localhost:8000"

#: The server is on the LAN or on localhost; a slow answer means something is
#: wrong, and the poll will come round again shortly anyway.
_TIMEOUT_S = 4.0


@dataclass(frozen=True)
class Eye:
    """One eye, resolved from `stereo_rig` into what the pipeline needs."""

    side: str                      # "left" | "right"
    camera_id: str
    orientation: Orientation
    #: As stored on the server, or None until the pair itself is calibrated.
    #: Kept raw because whether it is USABLE depends on the live frame size,
    #: which config parsing does not know — see `intrinsics_for`.
    intrinsics_raw: dict[str, Any] | None = None

    @property
    def configured(self) -> bool:
        return bool(self.camera_id)

    @property
    def has_intrinsics(self) -> bool:
        return isinstance(self.intrinsics_raw, dict)

    def intrinsics_for(self, frame_wh: tuple[int, int] | None) -> Intrinsics | None:
        """Intrinsics usable at `frame_wh`, or None.

        Resolved per frame rather than once at parse time: a camera matrix is
        only valid at the resolution it was solved at, and camserver can be
        reconfigured under a running app.
        """
        return intrinsics_from_eye({"intrinsics": self.intrinsics_raw}, frame_wh)


@dataclass(frozen=True)
class RigConfig:
    """A resolved snapshot of everything this app reads from the server."""

    camserver: str = ""
    token: str = ""
    baseline_mm: float = 0.0
    left: Eye | None = None
    right: Eye | None = None
    board: BoardSpec | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def eyes(self) -> tuple[Eye | None, Eye | None]:
        return self.left, self.right

    def stream_url(self, eye: Eye | None) -> str | None:
        """MJPEG URL on camserver for one eye, pair-synchronised.

        `sync=1` asks camserver for the frame stream it holds aligned with the
        other camera — which is the entire point of using a pair.
        """
        host = self.camserver.strip().rstrip("/")
        if not host or eye is None or not eye.camera_id:
            return None
        url = f"{host}/stream/{eye.camera_id}?sync=1"
        return f"{url}&token={self.token}" if self.token else url


def _eye_from(side: str, rig: dict[str, Any]) -> Eye:
    raw = rig.get(side) or {}
    k = raw.get("intrinsics")
    return Eye(
        side=side,
        camera_id=str(raw.get("camera_id") or ""),
        orientation=Orientation.from_eye(raw),
        intrinsics_raw=k if isinstance(k, dict) else None,
    )


def parse(cfg: dict[str, Any]) -> RigConfig:
    """Turn a raw `/config` payload into a `RigConfig`.

    Kept separate from the fetching so it can be unit-tested against a literal
    dict, with no server and no network.
    """
    rig = cfg.get("stereo_rig") or {}
    if not isinstance(rig, dict):
        rig = {}
    try:
        baseline = float(rig.get("baseline_mm", 0.0))
    except (TypeError, ValueError):
        baseline = 0.0
    return RigConfig(
        camserver=str(rig.get("host") or ""),
        token=str(rig.get("token") or ""),
        baseline_mm=baseline,
        left=_eye_from("left", rig),
        right=_eye_from("right", rig),
        board=board_spec_from_config(cfg),
        raw=cfg,
    )


class ConfigClient:
    """Fetches `/config` on demand. Errors are returned, never raised."""

    def __init__(self, server: str = DEFAULT_SERVER) -> None:
        self.server = server.rstrip("/")
        self._client = httpx.Client(timeout=_TIMEOUT_S)

    def close(self) -> None:
        self._client.close()

    def fetch(self) -> tuple[RigConfig | None, str | None]:
        """`(config, None)` on success, `(None, reason)` on failure.

        A failure is a normal state here — the server may simply not be running
        yet — so it is reported as a value the UI can display rather than an
        exception that would have to be caught at every call site.
        """
        try:
            resp = self._client.get(f"{self.server}/config")
            resp.raise_for_status()
            return parse(resp.json()), None
        except httpx.HTTPStatusError as exc:
            return None, f"HTTP {exc.response.status_code} from {self.server}/config"
        except httpx.HTTPError as exc:
            return None, f"{exc.__class__.__name__}: {exc}"
        except ValueError as exc:
            return None, f"malformed JSON from {self.server}/config: {exc}"
