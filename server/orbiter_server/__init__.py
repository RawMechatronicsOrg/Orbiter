"""Orbiter server package.

The modules inside this package use bare-name imports (``import config``,
``from esp_proxy import esp``, ...) rather than relative imports — a style
inherited from the parent storage-api so the same files can be run both as a
script (``uvicorn app:app``) and as a package (``import orbiter_server.app``).

This ``__init__`` makes that work by inserting the package directory into
``sys.path`` ahead of any other entry.
"""

from __future__ import annotations

import os
import sys

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)
