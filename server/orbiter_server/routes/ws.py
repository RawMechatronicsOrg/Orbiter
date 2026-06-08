"""WebSocket route — the `/ws/scene` scene + model channel."""

from __future__ import annotations

import logging

from fastapi import APIRouter, WebSocket
from starlette.websockets import WebSocketDisconnect

from ws_hub import hub

log = logging.getLogger("orbiter.ws")

router = APIRouter()


@router.websocket("/ws/scene")
async def ws_scene(ws: WebSocket) -> None:
    """Generic scene channel: pushes scene_snapshot/scene_update + model state,
    receives command/pick/ping from the client."""
    await hub.connect(ws)
    try:
        while True:
            msg = await ws.receive_json()
            await hub.handle_client_message(ws, msg)
    except WebSocketDisconnect:
        pass
    except Exception:  # noqa: BLE001
        log.exception("ws/scene error")
    finally:
        hub.disconnect(ws)
