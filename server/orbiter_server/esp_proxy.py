"""Async proxy to the ESP32 firmware — the storage-api is the sole point of
contact with the device (Viser-pattern migration).

Live state has two paths, primary + fallback:

  * **Stream** — a persistent WebSocket to the firmware's `/ws/log` consumes
    `pose` (~10 Hz), `task` and `log` frames. This is the low-latency primary
    path and the only source of firmware log lines.
  * **Poll** — `GET /state` at ~4 Hz, *dormant while the stream is fresh*.
    It activates whenever no `pose` frame has arrived within `_POSE_STALE_S`,
    so az/el/motion/motors/runner stay live even when the WS stream is
    unavailable (e.g. a firmware build whose `/ws/log` never emits).

Commands (`/move`, `/calibrate`, ...) are request/response over REST.

See API.md (`GET /ws/log`, `GET /state`) for the contracts.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from typing import Any, Callable

import httpx
import websockets

from config import settings
from orbiter_model import ModelState, model

log = logging.getLogger("orbiter.esp")

_TERMINAL = {"done", "error", "timeout"}

# Firmware contract: GET /state nests the persisted encoder-zero offsets
# under this key. The firmware's `/calibrate` endpoint writes them — this
# is the on-device encoder zero, NOT the (removed) calibration-board flow.
_FW_ENCODER_ZERO_KEY = "calibration"

#: A `pose` frame older than this means the /ws/log stream isn't delivering —
#: the REST poll takes over until frames resume.
_POSE_STALE_S = 2.0
#: REST /state poll cadence while the stream is stale.
_POLL_INTERVAL_S = 0.25
#: No /ws/log frame for this long ⇒ the socket is silently dead (the ESP
#: rebooted without a TCP FIN, network dropped, …). Forces a reconnect — the
#: firmware emits `pose` at ~10 Hz, so a multi-second gap is never legitimate.
_STREAM_IDLE_TIMEOUT_S = 5.0

LogCallback = Callable[[dict[str, Any]], None]


class EspError(RuntimeError):
    """A firmware command was rejected or the device is unreachable."""


class EspProxy:
    """Owns the HTTP command client, the /ws/log stream, the /state poll
    fallback, and the command wrappers."""

    def __init__(self, state: ModelState, esp_ip: str) -> None:
        self.model = state
        self._fallback_ip = esp_ip
        self.base_url = f"http://{esp_ip}"
        self._ws_url = f"ws://{esp_ip}/ws/log"
        #: Called with each firmware log frame. Wired to the WS hub in app.py.
        self.on_log: LogCallback | None = None
        self._client: httpx.AsyncClient | None = None
        self._stream_task: asyncio.Task[None] | None = None
        self._poll_task: asyncio.Task[None] | None = None
        self._stopping = asyncio.Event()
        #: monotonic ts of the last `pose` frame — drives the poll fallback.
        self._last_pose_at = 0.0
        #: one-shot guard so the "degraded to REST poll" notice fires once.
        self._degraded_logged = False

    # ── lifecycle ───────────────────────────────────────────────────────────

    def _set_endpoints(self, ip: str) -> None:
        self.base_url = f"http://{ip}"
        self._ws_url = f"ws://{ip}/ws/log"

    async def start(self) -> None:
        # Prefer the persisted/discovered IP over the settings default —
        # `model.esp_ip` is loaded from orbiter_state.json (or set by the UI /
        # mDNS discovery before lifespan finished). Falls back to whatever
        # was passed at construction (settings.esp_ip from .env).
        self._set_endpoints(self.model.esp_ip or self._fallback_ip)
        # Keep keep-alive enabled (forcing fresh TCP per request triggered
        # `httpcore.ReadError` storms — the firmware can't accept new
        # sockets fast enough). The `_http_lock` below + retry-on-transient
        # errors in `_post` handles the rare WS-into-HTTP cross-talk we
        # logged ("illegal status line: \\x81m...HTTP/1.1 200 OK") without
        # hammering ESP with reconnects.
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=8.0,
            limits=httpx.Limits(max_connections=4),
        )
        # Serialise outgoing HTTP so the /state poll never overlaps with a
        # command (/move, ...) — same firmware-multiplexing concern,
        # mitigated by sequencing rather than reconnecting.
        self._http_lock = asyncio.Lock()
        self._stopping.clear()
        self._stream_task = asyncio.create_task(self._stream_loop())
        self._poll_task = asyncio.create_task(self._poll_loop())
        # React to UI / discovery flipping the device address mid-run.
        self.model.subscribe(self._on_model_update)
        log.info("ESP proxy started — stream %s + /state poll fallback",
                 self._ws_url)

    def _on_model_update(self, patch: dict[str, Any]) -> None:
        """Re-target the proxy when `model.esp_ip` changes (UI / mDNS).
        Subscribers run synchronously inside `model.update()`; the actual
        client + stream swap is async, so we schedule it on the loop."""
        new_ip = patch.get("esp_ip")
        if not new_ip or f"http://{new_ip}" == self.base_url:
            return
        log.info("esp_ip changed → re-targeting %s", new_ip)
        asyncio.get_running_loop().create_task(self._retarget(new_ip))

    async def _retarget(self, new_ip: str) -> None:
        self._set_endpoints(new_ip)
        old_client = self._client
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=8.0,
            limits=httpx.Limits(max_connections=4),
        )
        if old_client is not None:
            await old_client.aclose()
        # Restart the /ws/log loop so it dials the new host. The poll loop
        # reads `self._client` each tick and picks up the new base_url
        # without a restart.
        if self._stream_task is not None and not self._stream_task.done():
            self._stream_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._stream_task
        if not self._stopping.is_set():
            self._stream_task = asyncio.create_task(self._stream_loop())
        # Force a fresh liveness verdict — the next stream pose or poll
        # /state will flip it back to True.
        self.model.update(esp_online=False)

    async def stop(self) -> None:
        self._stopping.set()
        for task in (self._stream_task, self._poll_task):
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        if self._client is not None:
            await self._client.aclose()
        log.info("ESP proxy stopped")

    # ── /ws/log stream (primary) ────────────────────────────────────────────

    async def _stream_loop(self) -> None:
        """Hold a WebSocket to the firmware's /ws/log; reconnect with backoff.
        Liveness (`esp_online`) is owned by the poll loop / `pose` frames, so a
        dropped stream does not by itself mark the device offline.

        We disable websockets' own keepalive (`ping_interval=None`), so a
        half-open socket — e.g. the ESP rebooted without sending a TCP FIN —
        would leave `recv()` blocked forever and the stream would never
        reconnect. The per-frame `_STREAM_IDLE_TIMEOUT_S` read timeout detects
        that: the firmware streams `pose` at ~10 Hz, so any multi-second gap
        means the connection is dead and we should reconnect."""
        backoff = 1.0
        while not self._stopping.is_set():
            connected = False
            try:
                async with websockets.connect(
                    self._ws_url,
                    ping_interval=None,   # firmware pings us; we just pong
                    open_timeout=8,
                    close_timeout=4,
                ) as ws:
                    connected = True
                    backoff = 1.0
                    if self._degraded_logged:
                        # Stream is back after a spell of REST-poll fallback.
                        self._degraded_logged = False
                        log.info("ESP /ws/log reconnected — stream restored")
                        self._emit_log("I", "esp",
                                        "firmware /ws/log stream restored")
                    else:
                        log.info("ESP /ws/log connected")
                    while not self._stopping.is_set():
                        try:
                            raw = await asyncio.wait_for(
                                ws.recv(), timeout=_STREAM_IDLE_TIMEOUT_S)
                        except asyncio.TimeoutError:
                            log.warning(
                                "ESP /ws/log silent >%.0fs — reconnecting",
                                _STREAM_IDLE_TIMEOUT_S)
                            break  # drop the dead socket, reconnect below
                        self._handle_frame(raw)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                if connected:
                    log.warning("ESP /ws/log lost: %s", exc)
            if self._stopping.is_set():
                break
            await asyncio.sleep(backoff)
            backoff = min(backoff * 1.7, 8.0)

    def _handle_frame(self, raw: str | bytes) -> None:
        """Dispatch one /ws/log frame: pose / task / log."""
        try:
            frame = json.loads(raw)
        except (ValueError, TypeError):
            return
        kind = frame.get("kind")

        if kind == "pose":
            self._last_pose_at = time.monotonic()
            patch: dict[str, Any] = {
                "esp_online": True,
                "motion_state": frame.get("st", "unknown"),
                "motors_on": bool(frame.get("motors", False)),
            }
            if "az" in frame:
                patch["az"] = float(frame["az"])
            if "el" in frame:
                patch["el"] = float(frame["el"])
            self.model.update(**patch)

        elif kind == "task":
            self.model.update(runner={
                "id": frame.get("task_id"),
                "status": frame.get("status"),
                "result": frame.get("result"),
            })

        elif kind == "log":
            cb = self.on_log
            if cb is not None:
                cb({
                    "level": frame.get("lvl", "I"),
                    "source": "fw",
                    "tag": frame.get("tag", ""),
                    "msg": frame.get("msg", ""),
                    "ts": frame.get("ts_ms"),
                    "seq": frame.get("seq"),
                })

    # ── /state poll (fallback) ──────────────────────────────────────────────

    async def fetch_state(self) -> dict[str, Any]:
        """One-shot `GET /state` + apply to model. Useful when a caller
        needs a field the WS pose stream doesn't carry (encoder_zero,
        runner) and the /state poll has been dormant because the stream
        is fresh."""
        async with self._http_lock:
            resp = await self._client.get("/state")  # type: ignore[union-attr]
        resp.raise_for_status()
        data = resp.json()
        self._apply_state(data)
        return data

    async def _poll_loop(self) -> None:
        """REST `/state` fallback. Dormant while the /ws/log `pose` stream is
        fresh; polls when no pose frame has arrived within `_POSE_STALE_S` so
        the model stays live even if the firmware WS stream never emits."""
        while not self._stopping.is_set():
            await asyncio.sleep(
                # ~2 s while throttled (sweep/scan in flight), ~0.25 s otherwise.
                2.0 if self._poll_throttled else _POLL_INTERVAL_S,
            )
            if (
                not self._poll_throttled
                and time.monotonic() - self._last_pose_at < _POSE_STALE_S
            ):
                continue  # stream is delivering — stay out of the way
            # Under throttle we ALWAYS poll (even if WS log is fresh) so
            # `wait_for_task` has a guaranteed runner.status refresh
            # within ~2 s even if the WS log frame is delayed/dropped.
            try:
                async with self._http_lock:
                    # Same lock as _post — never let /state and /move
                    # race onto the firmware's HTTP server simultaneously.
                    resp = await self._client.get("/state")  # type: ignore[union-attr]
                resp.raise_for_status()
                self._apply_state(resp.json())
                if not self._degraded_logged:
                    self._degraded_logged = True
                    log.warning("ESP /ws/log stream silent — using /state poll")
                    self._emit_log("W", "esp",
                                   "firmware /ws/log stream silent — "
                                   "live state via REST /state poll")
            except asyncio.CancelledError:
                raise
            except self._RETRY_EXC as exc:
                # Transient firmware/WS-multiplex hiccup — poll fires every
                # 250 ms, so just skip this tick. We don't flip esp_online
                # off because the device is almost certainly fine; the next
                # tick (or the WS stream) will refresh state.
                log.debug("ESP /state poll transient %s: %s",
                          type(exc).__name__, exc)
            except Exception as exc:  # noqa: BLE001
                if self.model.esp_online:
                    log.warning("ESP /state poll failed: %s", exc)
                self.model.update(esp_online=False, runner=None)

    def _apply_state(self, data: dict[str, Any]) -> None:
        """Map a `GET /state` response onto the model (same fields the `pose`
        stream would set, plus encoder-zero / runner)."""
        patch: dict[str, Any] = {"esp_online": True}
        if data.get("state") is not None:
            patch["motion_state"] = data["state"]
        if "motors_enabled" in data:
            patch["motors_on"] = bool(data["motors_enabled"])
        az = data.get("azimuth")
        if isinstance(az, dict) and "angle_deg" in az:
            patch["az"] = float(az["angle_deg"])
        el = data.get("elevation")
        if isinstance(el, dict) and "angle_deg" in el:
            patch["el"] = float(el["angle_deg"])
        # Firmware returns its persisted encoder-zero offsets under the
        # legacy key kept in `_FW_ENCODER_ZERO_KEY`; we expose it as
        # `encoder_zero` on the server-side model.
        enc = data.get(_FW_ENCODER_ZERO_KEY)
        if isinstance(enc, dict) and enc:
            patch["encoder_zero"] = enc
        runner = data.get("runner")
        if isinstance(runner, dict) and runner.get("id"):
            patch["runner"] = {
                "id": runner.get("id"),
                "status": runner.get("status"),
                "result": runner.get("result"),
            }
        self.model.update(**patch)

    def _emit_log(self, level: str, tag: str, msg: str) -> None:
        cb = self.on_log
        if cb is not None:
            cb({"level": level, "source": "api", "tag": tag, "msg": msg})

    # ── HTTP helpers ────────────────────────────────────────────────────────

    #: Transient HTTP errors that warrant a retry — the firmware
    #: occasionally a) leaks WS frames into an HTTP response (caught as
    #: RemoteProtocolError), b) drops the TCP socket mid-response
    #: (ReadError / WriteError), c) is too busy to answer in time
    #: (ReadTimeout), d) refuses a new TCP connect briefly while it
    #: services the WS stream (ConnectError). Up to `_POST_RETRIES` total
    #: attempts with a tiny back-off in between.
    _RETRY_EXC: tuple[type[Exception], ...] = (
        httpx.RemoteProtocolError,
        httpx.ReadError,
        httpx.WriteError,
        httpx.ReadTimeout,
        httpx.ConnectError,
        httpx.PoolTimeout,
    )
    _POST_RETRIES = 3
    _POST_RETRY_BACKOFF_S = 0.15
    # `_poll_throttled` — scan jobs flip this true so the /state
    # poll loop runs at a much slower cadence for their duration. The
    # firmware tips over when /state pounds it at 4 Hz alongside /move,
    # but we CAN'T fully suspend the poll: `wait_for_task` relies
    # on poll/WS to refresh `model.runner.status`, and if the WS log
    # channel hiccups (it does under load) wait_for_task hangs for the
    # full 37-s timeout. Throttling to 0.5 Hz keeps a safety net without
    # the contention storm.
    _poll_throttled: bool = False

    #: Retry-on-409 settings. The firmware returns 409 (`{"status":"busy"}`)
    #: when a previous /move task is still flagged `s_busy=true` in
    #: motion_runner. Even though our `wait_for_task` returned (driven by
    #: the WS-stream `task.done`), the firmware's HTTP server can hold
    #: `s_busy` for ~10-50 ms after the runner reports done; the next
    #: /move in a tight sweep loop hits this window. Retry with a
    #: gentler backoff than the TCP-error path because the firmware
    #: clears `s_busy` quickly once it's freed.
    _BUSY_RETRIES = 10
    _BUSY_BACKOFF_S = 0.2
    # After a /move task reports done via WS, leave the firmware this much
    # quiet time before sending the next command. The runner's `s_busy`
    # bit clears ~10-50 ms AFTER the `done` frame goes out — our 409 retry
    # covers the gap, but a sleep here saves a network round-trip and is
    # what keeps long sweeps reliable. Cheap relative to the move itself.
    _MOVE_SETTLE_S = 0.12

    async def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        if self._client is None:
            raise EspError("ESP proxy not started")
        async with self._http_lock:
            last_exc: Exception | None = None
            resp = None
            for attempt in range(self._POST_RETRIES):
                try:
                    resp = await self._client.post(path, json=body)
                    last_exc = None
                    break
                except self._RETRY_EXC as exc:
                    last_exc = exc
                    if attempt + 1 < self._POST_RETRIES:
                        log.warning(
                            "ESP %s: %s (attempt %d/%d) — retrying",
                            path, type(exc).__name__, attempt + 1,
                            self._POST_RETRIES,
                        )
                        await asyncio.sleep(self._POST_RETRY_BACKOFF_S)
                        continue
                except httpx.HTTPError as exc:
                    # Include the exception TYPE in the message — many
                    # httpx errors (especially around abrupt TCP closes)
                    # carry an empty `args`, and `str(exc)` is "". Without
                    # the type name the operator sees a useless "/move: "
                    # in the sweep log. Same trick in the busy-retry
                    # branch and the final last_exc raise below.
                    raise EspError(f"{path}: {type(exc).__name__}: {exc}") from exc
            if last_exc is not None or resp is None:
                raise EspError(
                    f"{path}: {type(last_exc).__name__}: {last_exc}"
                ) from last_exc
            # 409 retry — firmware still flagged busy from the previous
            # task. Re-post with a tiny backoff; the runner clears
            # `s_busy` within ~10-50 ms.
            for busy_attempt in range(self._BUSY_RETRIES):
                if resp.status_code != 409:
                    break
                if busy_attempt + 1 >= self._BUSY_RETRIES:
                    break
                log.debug(
                    "ESP %s: HTTP 409 busy (attempt %d/%d) — retrying",
                    path, busy_attempt + 1, self._BUSY_RETRIES,
                )
                await asyncio.sleep(self._BUSY_BACKOFF_S)
                try:
                    resp = await self._client.post(path, json=body)
                except self._RETRY_EXC as exc:
                    raise EspError(
                        f"{path}: {type(exc).__name__}: {exc}",
                    ) from exc
                except httpx.HTTPError as exc:
                    raise EspError(
                        f"{path}: {type(exc).__name__}: {exc}",
                    ) from exc
        data: dict[str, Any] = resp.json() if resp.content else {}
        if resp.status_code >= 400:
            raise EspError(data.get("message") or f"{path}: HTTP {resp.status_code}")
        return data

    async def wait_for_task(self, task_id: int, timeout_s: float) -> dict[str, Any]:
        """Wait until the task reaches a terminal status. `model.runner` is kept
        fresh by the /ws/log `task` frames or the /state poll fallback."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            r = self.model.runner
            if isinstance(r, dict) and r.get("id") is not None:
                rid = int(r["id"])
                if rid == task_id and r.get("status") in _TERMINAL:
                    return r
                if rid > task_id:
                    # Runner moved on — our task finished by construction.
                    return {"id": task_id, "status": "done", "result": r.get("result")}
            await asyncio.sleep(0.1)
        raise EspError(f"task {task_id} did not finish within {timeout_s:.0f}s")

    @staticmethod
    def _task_id(ack: dict[str, Any], what: str) -> int:
        if ack.get("status") != "accepted" or ack.get("task_id") is None:
            raise EspError(f"{what} not accepted: {ack.get('status')}")
        return int(ack["task_id"])

    # ── command wrappers ────────────────────────────────────────────────────

    async def move(
        self,
        azimuth_deg: float | None = None,
        elevation_deg: float | None = None,
        timeout_ms: int = 30_000,
    ) -> dict[str, Any]:
        """Submit a /move (202 + task_id). Does not wait — see `move_and_await`."""
        body: dict[str, Any] = {"timeout_ms": timeout_ms}
        if azimuth_deg is not None:
            body["azimuth_deg"] = azimuth_deg
        if elevation_deg is not None:
            body["elevation_deg"] = elevation_deg
        return await self._post("/move", body)

    async def move_and_await(
        self,
        azimuth_deg: float | None = None,
        elevation_deg: float | None = None,
        timeout_ms: int = 30_000,
    ) -> dict[str, Any]:
        """Submit a /move and block until the runner finishes."""
        tgt_az = self.model.az if azimuth_deg is None else float(azimuth_deg)
        tgt_el = self.model.el if elevation_deg is None else float(elevation_deg)
        self.model.update(move_target_az=tgt_az, move_target_el=tgt_el)
        try:
            ack = await self.move(azimuth_deg, elevation_deg, timeout_ms)
            task_id = self._task_id(ack, "/move")
            result = await self.wait_for_task(task_id, timeout_ms / 1000.0 + 7.0)
            # Brief settle — see `_MOVE_SETTLE_S`. Without it the very next
            # /move (sweep loop) races the firmware clearing s_busy and we
            # eat at least one 409 retry per cell.
            await asyncio.sleep(self._MOVE_SETTLE_S)
            return result
        finally:
            self.model.update(move_target_az=None, move_target_el=None)

    def throttle_poll(self) -> None:
        """Slow the /state poll to ~0.5 Hz — used by long-running
        sweep/scan jobs to ease HTTP contention on the firmware. We
        DON'T fully suspend it: `wait_for_task` polls model.runner and
        the WS log channel isn't reliable enough under load to be the
        sole source of `task.done` events. 0.5 Hz still detects a
        terminal status within 2 s of it happening, well under the
        37-s wait_for_task timeout."""
        self._poll_throttled = True

    def resume_poll(self) -> None:
        self._poll_throttled = False

    # Backwards-compat shim — older callers said `suspend_poll`. Kept so
    # we don't have to grep every job file when we rename the concept.
    def suspend_poll(self) -> None:
        self.throttle_poll()

    async def calibrate(
        self,
        axis: str,
        mode: str = "current",
        az_raw_deg: float | None = None,
        el_raw_deg: float | None = None,
    ) -> dict[str, Any]:
        """Set the firmware encoder zero. Hits the firmware's `/calibrate`
        REST endpoint — kept verbatim to match the API.md contract."""
        body: dict[str, Any] = {"axis": axis, "mode": mode}
        if az_raw_deg is not None:
            body["az_raw_deg"] = az_raw_deg
        if el_raw_deg is not None:
            body["el_raw_deg"] = el_raw_deg
        result = await self._post("/calibrate", body)
        # The zero offsets are not in the pose stream — refresh them from the
        # command response so the model stays in sync.
        if "az_zero_raw_deg" in result and "el_zero_raw_deg" in result:
            self.model.update(encoder_zero={
                "az_zero_raw_deg": result["az_zero_raw_deg"],
                "el_zero_raw_deg": result["el_zero_raw_deg"],
            })
        return result

    async def motors(self, enabled: bool) -> dict[str, Any]:
        return await self._post("/motors", {"enabled": enabled})

    async def reboot(self) -> dict[str, Any]:
        """Ask the firmware to restart. The device acks 200 then reboots
        ~500 ms later, so `esp_online` flips false shortly after this
        returns and back to true once the firmware is up again."""
        return await self._post("/reboot", {})


# Process-wide singleton sharing the model singleton.
esp = EspProxy(model, settings.esp_ip)
