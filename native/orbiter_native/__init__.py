"""Orbiter native CV workbench.

A desktop client for the binocular camera pair. The Orbiter server stays the
owner of state — this app reads its configuration over HTTP and pulls pixels
straight from camserver, doing the frame-rate work (ChArUco detection, laser
line, timing) that a browser cannot: a cross-origin `<img>` taints the canvas,
so the web UI can display the pair but never look inside a frame.

The web Stereo tab remains the settings surface for the pair; `stereo_rig` is
the baseline this app consumes.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
