"""The guide through the main window: the banner, the eye views and the
flow are wired together, against no server at all.

The one wiring slip so far (`scan.is_active` for `scan.scanning`) raised
only inside a Qt timer, at runtime. This builds the window without showing
it, ticks the guide by hand and reads what it put where.
"""

from __future__ import annotations

import os

import pytest


def _app():
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "0")
    from PySide6.QtWidgets import QApplication
    try:
        return QApplication.instance() or QApplication([])
    except Exception as exc:                                   # noqa: BLE001
        pytest.skip(f"no Qt platform here: {exc}")


def test_the_guide_is_wired_through_the_main_window() -> None:
    _app()
    from orbiter_native.app import MainWindow

    win = MainWindow("http://127.0.0.1:9")                    # nothing listens there
    try:
        win._guide_tick()
        title, action = win.banner.title.text(), win.banner.action.text()
        assert action and ("STEP" in title or "CALIBRATION" in title)
        assert "BOARD SPEC" in action                          # no server: no board
        assert win.calib.flow.solo is None
        # With a board the first stage is the left lens, and the window is
        # what carries that into the flow — the one line joining the two.
        import cv2
        from orbiter_native.cvcore import BoardSpec, build_board
        spec = BoardSpec(8, 8, 36.0, 26.64, cv2.aruco.DICT_5X5_100)
        win.calib.flow.set_board(spec, build_board(spec))
        win._guide_tick()
        assert win.calib.flow.solo == "left" and "STEP 1/6" in win.banner.title.text()
        assert win.panels["left"].view._highlight and not win.panels["right"].view._highlight
        # Navigation reaches the guide and re-ticks without raising.
        win._guide_next()
        win._guide_back()
        win._guide_restart()
        assert win.guide.index == 0 and not win.guide.pinned
        # Off: the flow pairs as usual and nothing of the guide is left drawn.
        win.banner.enabled.setChecked(False)
        assert win.calib.flow.solo is None
        for panel in win.panels.values():
            assert panel.view._target is None and not panel.view._highlight
        assert not win.banner.action.isVisibleTo(win.banner)
        win.banner.enabled.setChecked(True)
        win._guide_tick()
        assert win.banner.action.text()
        # An offline eye is reported through the same path the panels use.
        win._on_status("left", "connection refused")
        win._guide_tick()
        assert win._offline["left"] == "connection refused"
        # The panel's buttons reach the guide.
        win.guide.next()
        win.calib.cleared.emit()
        assert win.guide.index == 0 and not win.guide.pinned
        win.guide.next()
        win.calib.rig_moved.emit()
        assert win.guide.index == 0 and not win.guide.pinned
    finally:
        win.close()
