"""WebSocket hub — broadcasts scene-graph + model state to all clients.

One shared scene, Viser-style: every connected browser sees the same nodes.
The server is the single writer. On connect a client gets a full
`scene_snapshot` + `model`; thereafter it receives `scene_update` diffs and
`model_patch` deltas whenever the model changes.

Wire envelope: `{ "t": <type>, "seq": N, "ts": <unix>, "data": {...} }`.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from typing import Any

from starlette.websockets import WebSocket

import commands
from orbiter_model import ModelState
from scene_graph import Node, build_scene, diff

log = logging.getLogger("orbiter.ws")

#: Max browser-facing broadcast rate. The firmware streams pose at ~100 Hz;
#: there's no point repainting faster than a display refresh, so model
#: changes are coalesced and flushed at most this often. Latency stays low
#: (one flush interval) while CPU / socket traffic stays bounded.
_BROADCAST_HZ = 60.0
_BROADCAST_MIN_INTERVAL = 1.0 / _BROADCAST_HZ

#: Per-WS send budget. A wedged client (laptop sleep, NAT timeout that
#: drops packets without closing the socket) used to stall every other
#: client because `_broadcast` awaited sends serially. We now fan out
#: with `asyncio.gather` and time-out each individual send: a slow socket
#: gets evicted, the rest get their frames on time.
_SEND_TIMEOUT_S = 3.0


class WsHub:
    def __init__(self) -> None:
        self._conns: set[WebSocket] = set()
        self._seq = 0
        self._prev: dict[str, Node] = {}
        self._model: ModelState | None = None
        #: Coalesced model patch awaiting the next flush, and the single
        #: flush task — one writer, so scene diffs never race.
        self._pending: dict[str, Any] = {}
        self._flush_task: asyncio.Task[None] | None = None

    # ── lifecycle ───────────────────────────────────────────────────────────

    def start(self, model: ModelState) -> None:
        """Bind the hub to the model and subscribe for change notifications."""
        self._model = model
        self._prev = {n["id"]: n for n in build_scene(model)}
        model.subscribe(self._on_model_change)

    async def stop(self) -> None:
        """Cancel the coalescing flush task and close any open client sockets
        so the event loop is free to exit during server shutdown. Without
        this the flush task can keep getting re-armed by late `_on_model_change`
        callbacks (from background tasks unwinding), and open WebSockets
        keep uvicorn's listen socket pinned."""
        if self._flush_task is not None and not self._flush_task.done():
            self._flush_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                try:
                    await asyncio.wait_for(self._flush_task, timeout=1.0)
                except asyncio.TimeoutError:
                    log.warning("ws hub flush task did not finish within 1s")
        self._flush_task = None
        # Close every still-open client. Best-effort — a client that has
        # already gone away will error, ignore those.
        for ws in list(self._conns):
            await self._safe_close(ws)
        self._conns.clear()

    # ── model change → broadcast ────────────────────────────────────────────

    def _on_model_change(self, patch: dict[str, Any]) -> None:
        """Sync subscriber callback — coalesce the patch and ensure a flush
        task is running. Merging keeps a burst of 100 Hz pose updates from
        spawning 100 racing broadcast tasks."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return  # no loop yet (import-time) — nothing to broadcast to
        self._pending.update(patch)
        if self._flush_task is None or self._flush_task.done():
            self._flush_task = asyncio.create_task(self._flush_loop())

    async def _flush_loop(self) -> None:
        """Drain coalesced patches, one broadcast at a time, rate-limited to
        `_BROADCAST_HZ`. Single task → scene diffs are serialised.

        Wrapped in try/except so a transient `build_scene` failure (e.g. a
        missing GLB file mid-development) doesn't kill the flush task and
        wedge every subsequent state change. Next pending patch picks up
        where we left off."""
        while self._pending:
            patch = self._pending
            self._pending = {}
            try:
                await self._broadcast_change(patch)
            except Exception:  # noqa: BLE001
                log.exception("broadcast_change failed — continuing")
            await asyncio.sleep(_BROADCAST_MIN_INTERVAL)
        self._flush_task = None

    def emit_log(self, entry: dict[str, Any]) -> None:
        """Forward a firmware log line to all /ws/scene clients as a `log`
        message. Sync — safe to wire as the ESP proxy's `on_log` callback."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return
        asyncio.create_task(self._broadcast("log", entry))

    async def _broadcast_change(self, patch: dict[str, Any]) -> None:
        if not self._conns or self._model is None:
            # Keep the scene baseline current even with no clients attached.
            self._prev = {n["id"]: n for n in build_scene(self._model)} \
                if self._model else {}
            return
        await self._broadcast("model_patch", patch)
        nodes = build_scene(self._model)
        update = diff(self._prev, nodes)
        self._prev = {n["id"]: n for n in nodes}
        if update["added"] or update["updated"] or update["removed"]:
            await self._broadcast("scene_update", update)

    async def broadcast_scene_refresh(self) -> None:
        """Push a full ``scene_snapshot`` after structural model changes.

        Diff-only ``scene_update`` can miss visual changes when node ids are
        stable but transforms/props should fully replace.
        """
        if self._model is None:
            return
        nodes = build_scene(self._model)
        self._prev = {n["id"]: n for n in nodes}
        if not self._conns:
            return
        await self._broadcast("scene_snapshot", {"nodes": nodes})

    # ── connection handling ─────────────────────────────────────────────────

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        # Register BEFORE the first send so a broadcast that fires while
        # we're awaiting `send_json` for the snapshot still reaches this
        # client. The reconciler is idempotent — any pre-snapshot diff lands
        # on an empty scene as a no-op, then the snapshot wipes and rebuilds.
        # If the snapshot itself fails (build_scene crash, write failure)
        # we unregister and close so the browser retries with a fresh socket
        # instead of being left on the previous frame forever.
        self._conns.add(ws)
        log.info("ws client connected (%d total)", len(self._conns))
        if self._model is not None:
            try:
                await self._send_initial(ws)
            except Exception:  # noqa: BLE001
                log.exception("ws/scene initial snapshot failed")
                self._conns.discard(ws)
                await self._safe_close(ws)
                return

    async def _send_initial(self, ws: WebSocket) -> None:
        """Snapshot + model for a freshly-accepted client. Centralised so the
        same code runs on initial connect AND on explicit `snapshot`
        commands from a client that wants to re-sync."""
        if self._model is None:
            return
        nodes = build_scene(self._model)
        await self._send(ws, "scene_snapshot", {"nodes": nodes})
        await self._send(ws, "model", self._model.to_dict())

    @staticmethod
    async def _safe_close(ws: WebSocket) -> None:
        try:
            await ws.close()
        except Exception:  # noqa: BLE001
            pass

    def disconnect(self, ws: WebSocket) -> None:
        self._conns.discard(ws)
        log.info("ws client disconnected (%d total)", len(self._conns))

    async def handle_client_message(self, ws: WebSocket, msg: dict[str, Any]) -> None:
        """Inbound client message. Phase 3 handles `ping`; `command`/`pick`
        arrive in later phases."""
        t = msg.get("t")
        if t == "ping":
            await self._send(ws, "pong", {})
        elif t == "snapshot":
            # Belt-and-braces: the client lost confidence in its local scene
            # (lifecycle quirk, post-reconnect doubt, HMR) and asks for a
            # fresh baseline. Cheap on the server — one build_scene per call.
            try:
                await self._send_initial(ws)
            except Exception:  # noqa: BLE001
                log.exception("snapshot resend failed")
        elif t == "command":
            # Run off the receive loop so a slow command (e.g. /move) does not
            # stall this client's inbound message processing.
            asyncio.create_task(self._run_command(ws, msg.get("data") or {}))
        elif t == "pick":
            log.info("ws pick: %s", msg.get("data"))
        elif t == "camera":
            log.debug("ws camera received (not yet handled)")
        else:
            log.debug("ws unknown message type: %r", t)

    async def _run_command(self, ws: WebSocket, data: dict[str, Any]) -> None:
        """Dispatch a `command` message and reply with `command_result`."""
        name = str(data.get("name", ""))
        args = data.get("args") or {}
        try:
            result = await commands.dispatch(name, args)
            payload: dict[str, Any] = {"name": name, "ok": True, "result": result}
        except Exception as exc:  # noqa: BLE001 — surface every failure to the client
            log.warning("command %s failed: %s", name, exc)
            payload = {"name": name, "ok": False, "error": str(exc)}
        try:
            await self._send(ws, "command_result", payload)
        except Exception:  # noqa: BLE001 — client vanished mid-command
            self._conns.discard(ws)

    # ── send helpers ────────────────────────────────────────────────────────

    def _envelope(self, t: str, data: Any) -> dict[str, Any]:
        self._seq += 1
        return {"t": t, "seq": self._seq, "ts": time.time(), "data": data}

    async def _send(self, ws: WebSocket, t: str, data: Any) -> None:
        await ws.send_json(self._envelope(t, data))

    async def _broadcast(self, t: str, data: Any) -> None:
        if not self._conns:
            return
        msg = self._envelope(t, data)
        # Parallel sends with a per-socket timeout so one slow client (NAT
        # drop, laptop asleep) can't stall pong replies / scene updates for
        # the rest. A timing-out send is treated as a dead connection.
        ws_list = list(self._conns)
        results = await asyncio.gather(
            *(self._send_with_timeout(ws, msg) for ws in ws_list),
            return_exceptions=True,
        )
        for ws, ok in zip(ws_list, results):
            if ok is not True:
                self._conns.discard(ws)
                await self._safe_close(ws)

    async def _send_with_timeout(self, ws: WebSocket, msg: dict[str, Any]) -> bool:
        try:
            await asyncio.wait_for(ws.send_json(msg), timeout=_SEND_TIMEOUT_S)
        except (asyncio.TimeoutError, Exception):  # noqa: BLE001
            return False
        return True


# Process-wide singleton.
hub = WsHub()
