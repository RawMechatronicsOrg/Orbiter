"""Non-blocking file logging shared by both services.

Why this exists: a service launched with its stdout attached to a pipe
(IDE debug console, launcher harness, docker without a TTY) hangs FOREVER
once the pipe buffer fills and nobody drains it — caught live with py-spy:
the uvicorn event loop blocked inside ``logging emit → stream.write`` (the
access log). See docs/RELIABILITY.md §1.1.

The fix is to never write to a stream from the event loop:

    root logger → QueueHandler (never blocks)
        → QueueListener thread → RotatingFileHandler (data/logs/<service>.log)
                                → StreamHandler (console, best effort)

Worst case with a blocked console the listener thread stalls and the queue
grows — log lines are delayed, the SERVICE keeps running. The on-disk log
also gives every crash a post-mortem (uncaught exceptions are hooked below).

The uvicorn access log is disabled outright: it was the exact blocking
write, and per-request noise at our scale; uvicorn's error/startup logs
are re-routed through the queue instead.
"""

from __future__ import annotations

import asyncio
import atexit
import logging
import logging.handlers
import queue
import sys
import threading
from pathlib import Path

from config import settings

_FILE_MAX_BYTES = 5 * 1024 * 1024
_FILE_BACKUPS = 3
_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"

_configured: str | None = None


def log_dir() -> Path:
    return settings.storage_dir / "logs"


def setup_logging(service: str, *, level: int = logging.INFO) -> Path:
    """Configure root logging for one service. Idempotent; returns the log
    file path. Call before (or instead of) ``logging.basicConfig``."""
    global _configured
    logfile = log_dir() / f"{service}.log"
    if _configured is not None:
        return logfile

    log_dir().mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter(_FORMAT)

    file_handler = logging.handlers.RotatingFileHandler(
        logfile, maxBytes=_FILE_MAX_BYTES, backupCount=_FILE_BACKUPS,
        encoding="utf-8",
    )
    file_handler.setFormatter(fmt)
    console = logging.StreamHandler()
    console.setFormatter(fmt)

    q: queue.SimpleQueue = queue.SimpleQueue()
    listener = logging.handlers.QueueListener(
        q, file_handler, console, respect_handler_level=True,
    )
    listener.start()
    atexit.register(listener.stop)

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers = [logging.handlers.QueueHandler(q)]

    # uvicorn installs its own stream handlers before the app imports —
    # re-route its error/startup logs through the queue and silence the
    # access log entirely (the blocking writer from the incident).
    for name in ("uvicorn", "uvicorn.error"):
        lg = logging.getLogger(name)
        lg.handlers = []
        lg.propagate = True
    access = logging.getLogger("uvicorn.access")
    access.handlers = []
    access.propagate = False

    logging.getLogger("httpx").setLevel(logging.WARNING)

    _install_crash_hooks()
    _configured = service
    logging.getLogger("orbiter.logsetup").info(
        "%s logging → %s (access log off)", service, logfile)
    return logfile


def _install_crash_hooks() -> None:
    """Every crash leaves a traceback in the log file (post-mortem)."""

    def _excepthook(exc_type, exc, tb) -> None:
        logging.getLogger("orbiter.crash").critical(
            "uncaught exception", exc_info=(exc_type, exc, tb))
        sys.__excepthook__(exc_type, exc, tb)

    def _thread_excepthook(args: threading.ExceptHookArgs) -> None:
        logging.getLogger("orbiter.crash").critical(
            "uncaught exception in thread %s", args.thread.name if args.thread else "?",
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback))

    sys.excepthook = _excepthook
    threading.excepthook = _thread_excepthook


def install_loop_exception_handler(loop: asyncio.AbstractEventLoop) -> None:
    """Log unhandled task exceptions (call from the app lifespan startup)."""

    def _handler(_loop, context: dict) -> None:
        exc = context.get("exception")
        msg = context.get("message", "unhandled loop exception")
        if exc is not None:
            logging.getLogger("orbiter.crash").error(msg, exc_info=exc)
        else:
            logging.getLogger("orbiter.crash").error("%s: %r", msg, context)

    loop.set_exception_handler(_handler)
