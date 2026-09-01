"""HTTP access to the named-command channel.

`commands.dispatch` is the single place state is mutated, and this route does
not change that — it is the same door with a thinner transport. The WebSocket
at `/ws/scene` carries commands *and* the live model push, which is what a
browser needs; a desktop client that only issues the occasional command should
not have to implement that protocol to do it.

Used by the native workbench to store a solved calibration. Anything that
mutates state still goes through `dispatch`, so validation, persistence and the
resulting model broadcast are identical whichever transport asked.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException

import commands

log = logging.getLogger("orbiter.routes.command")

router = APIRouter(prefix="/command", tags=["command"])


@router.post("/{name}")
async def run_command(name: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
    """Dispatch one named command. Body is the argument object.

    A rejected command is a 400 carrying the handler's own message: these are
    operator errors (both eyes on one camera, a malformed calibration) and the
    reason is the useful part.
    """
    if name not in commands.known_commands():
        raise HTTPException(status_code=404, detail=f"unknown command {name!r}")
    try:
        return {"ok": True, "result": await commands.dispatch(name, args or {})}
    except commands.CommandError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("")
def list_commands() -> dict[str, Any]:
    """The command names this server accepts."""
    return {"commands": commands.known_commands()}
