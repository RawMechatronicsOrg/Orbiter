"""Story E1: the PHOTOS group, the RECONSTRUCT row, and what they reach.

Every failure this file is written against is a silent one. A half-wired "take
photos" checkbox looks exactly like a working one until the session comes out
empty an hour later; a reconstruction started on the GUI thread looks like a
frozen app; a `closeEvent` that forgets the writer loses the last photographs
of a pass, and one that forgets the recon thread leaves a container running
past the window that started it.

So the window is built offline — pointed at a dead port, with the session root
and the exposure file redirected into `tmp_path` — and the three calls, the
thread, the drain and the shutdown are read back where they happen.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from orbiter_native.scanworker import ScanStatus


def _app():
    import os

    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "0")
    from PySide6.QtWidgets import QApplication
    try:
        return QApplication.instance() or QApplication([])
    except Exception as exc:                                   # noqa: BLE001
        pytest.skip(f"no Qt platform here: {exc}")


def _window(tmp_path, monkeypatch):
    """A `MainWindow` that talks to nothing and writes nowhere near home."""
    monkeypatch.setenv("ORBITER_SESSIONS_DIR", str(tmp_path / "sessions"))
    _app()
    from orbiter_native.app import MainWindow

    win = MainWindow("http://127.0.0.1:9")                     # nothing listens there
    win.exposure.path = tmp_path / "exposure.json"
    return win


def _config():
    """A `/config` payload with both eyes solved, the pair solved and a board:
    what the window needs before it will open a session at all."""
    from orbiter_native.config import parse

    def eye(camera_id: str, fx: float) -> dict:
        return {"camera_id": camera_id,
                "intrinsics": {"fx": fx, "fy": fx, "cx": 960.0, "cy": 540.0,
                               "dist": [0.1, -0.2, 0.0, 0.0, 0.0],
                               "width": 1920, "height": 1080, "rms_px": 0.81}}

    return parse({
        "stereo_rig": {
            "host": "http://192.168.0.222:8088",
            "baseline_mm": 120.0,
            "left": eye("cam2", 1500.0),
            "right": eye("cam1", 1510.0),
            "extrinsics": {"R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                           "T": [-120.0, 0.0, 0.0], "width": 1920, "height": 1080,
                           "rms_px": 0.89},
        },
        "charuco_squares_x": 8,
        "charuco_squares_y": 8,
        "charuco_square_length_mm": 36.0,
        "charuco_marker_length_mm": 26.64,
        "aruco_dict_id": 5,
    })


def _status(session_id: str = "", **fields) -> ScanStatus:
    return ScanStatus(n_points=0, bounds=None, pairs=0, session_id=session_id, **fields)


def _arm(win) -> None:
    """The config the window refuses to open a session without."""
    win._config = _config()


# ── the PHOTOS group ─────────────────────────────────────────────────────


def test_panel_shows_the_session_and_counts(tmp_path, monkeypatch) -> None:
    win = _window(tmp_path, monkeypatch)
    try:
        assert win.scan.photo_stats.text() == "no session yet"
        _arm(win)
        win.scan.photos.setChecked(True)
        session = win._session
        assert session is not None
        win.scan.on_status(_status(session.session_id, pass_id=1, photos_left=7,
                                   photos_right=6, photos_dropped=2,
                                   session_bytes=3 * 1024 * 1024))
        text = win.scan.photo_stats.text()
        assert session.session_id in text and "pass 1" in text
        assert "L 7" in text and "R 6" in text and "dropped 2" in text
        # The size is the counter the writer keeps, shown at a scale an
        # operator reads — and no directory was walked to produce it.
        assert "3.0 MB" in text
        assert win.scan.btn_folder.isEnabled()
    finally:
        win.close()


def test_the_session_records_the_rig_from_the_config(tmp_path, monkeypatch) -> None:
    """`session.json` is the only thing a reconstruction knows the rig by, and
    this window is the one place that holds a `RigConfig` to fill it from."""
    win = _window(tmp_path, monkeypatch)
    try:
        _arm(win)
        win.scan.photos.setChecked(True)
        session = win._session
        assert session is not None
        data = json.loads((session.path / "session.json").read_text(encoding="utf-8"))
        assert data["eyes"]["left"]["camera_id"] == "cam2"
        assert data["eyes"]["right"]["camera_id"] == "cam1"
        assert data["eyes"]["left"]["wh"] == [1920, 1080]
        assert data["eyes"]["left"]["intrinsics"]["fx"] == 1500.0
        assert data["eyes"]["left"]["intrinsics"]["rms_px"] == 0.81
        assert data["extrinsics"]["T_mm"] == [-120.0, 0.0, 0.0]
        assert data["extrinsics"]["rms_px"] == 0.89
        # The dictionary travels as its name: nobody reading this file wants
        # to look DICT_5X5_100 up from 5.
        assert data["board"]["dictionary"] == "DICT_5X5_100"
        assert data["board"]["squares_x"] == 8
        volume = win.scan.params().volume
        assert data["volume"]["radius_mm"] == volume.radius_mm
        assert data["volume"]["floor_mm"] == volume.floor_mm
    finally:
        win.close()


def test_take_photos_toggle_arms_both_eye_workers_and_the_scan_worker(
        tmp_path, monkeypatch) -> None:
    """The one that catches a half-wired checkbox. Without the eye workers
    there are no bytes to write; without the scan worker nothing is decided."""
    win = _window(tmp_path, monkeypatch)
    try:
        calls: list[tuple[str, bool]] = []
        for side, worker in win.workers.items():
            monkeypatch.setattr(
                worker, "set_photo_capture",
                lambda on, s=side: calls.append((s, bool(on))))
        monkeypatch.setattr(win.scanner, "arm_photos",
                            lambda on: calls.append(("scan", bool(on))))
        _arm(win)
        win.scan.photos.setChecked(True)
        assert sorted(calls) == [("left", True), ("right", True), ("scan", True)]
        calls.clear()
        win.scan.photos.setChecked(False)
        assert sorted(calls) == [("left", False), ("right", False), ("scan", False)]
    finally:
        win.close()


def test_arming_photos_without_a_config_is_refused_and_says_why(
        tmp_path, monkeypatch) -> None:
    """A session with no intrinsics, no pair and no board is eighty
    photographs nobody can reconstruct, discovered an hour later."""
    win = _window(tmp_path, monkeypatch)
    try:
        armed: list[bool] = []
        monkeypatch.setattr(win.scanner, "arm_photos", lambda on: armed.append(bool(on)))
        assert win._config is None                             # no server answered
        win.scan.photos.setChecked(True)
        assert win._session is None
        assert not win.scan.photos.isChecked()
        assert armed == [False]                                # disarmed, never armed
        assert "no rig config" in win.statusBar().currentMessage()
    finally:
        win.close()


def test_photo_pass_toggle_reaches_the_scan_worker(tmp_path, monkeypatch) -> None:
    win = _window(tmp_path, monkeypatch)
    try:
        seen: list[bool] = []
        monkeypatch.setattr(win.scanner, "set_photo_pass",
                            lambda on: seen.append(bool(on)))
        for worker in win.workers.values():
            monkeypatch.setattr(
                worker, "set_photo_capture",
                lambda on: pytest.fail("photo pass must not touch an eye worker"))
        win.scan.photo_pass.setChecked(True)
        win.scan.photo_pass.setChecked(False)
        assert seen == [True, False]
    finally:
        win.close()


def test_a_session_starts_on_clear_cloud(tmp_path, monkeypatch) -> None:
    """A session belongs to the cloud it was taken alongside, so emptying the
    cloud ends one session and opens the next."""
    win = _window(tmp_path, monkeypatch)
    try:
        _arm(win)
        assert win._session is None
        win.scan.btn_clear.click()
        first = win._session
        assert first is not None and first.path.is_dir()
        # Named at once: nothing is pairing with scanning off, so a status
        # would not arrive for as long as the operator stood there.
        assert first.session_id in win.scan.photo_stats.text()
        assert "pass 0" in win.scan.photo_stats.text()
        win.scan.btn_clear.click()
        second = win._session
        assert second is not None and second is not first
        assert second.path != first.path
        assert second.session_id in win.scan.photo_stats.text()
    finally:
        win.close()


# ── the RECONSTRUCT row ──────────────────────────────────────────────────


def test_reconstruct_is_disabled_without_photos(tmp_path, monkeypatch) -> None:
    win = _window(tmp_path, monkeypatch)
    try:
        assert not win.scan.btn_recon.isEnabled()
        win.scan.set_session("20260907-101500")
        assert not win.scan.btn_recon.isEnabled()              # a session, no photos
        win.scan.on_status(_status("20260907-101500", photos_left=1))
        assert win.scan.btn_recon.isEnabled()
        # A status left over from the session before this one is not this
        # session's evidence, and does not arm the button either.
        win.scan.set_session("20260907-102000")
        assert not win.scan.btn_recon.isEnabled()
    finally:
        win.close()


def test_log_lines_drain_from_the_list_without_blocking(tmp_path, monkeypatch) -> None:
    """Accumulate-and-swap, not a one-slot mailbox: the tick takes every line
    the thread wrote, and the step survives the lines that follow it."""
    win = _window(tmp_path, monkeypatch)
    try:
        win._recon_line("write_sparse: running")
        win._recon_line("Reading 42 images")
        win._recon_tick()
        assert win.scan.recon_status.text() == "write_sparse · Reading 42 images"
        assert win._recon_lines == []                          # swapped, not copied

        # A step's own line is the only thing that moves the step, so the
        # chatter after it does not push it out of the status line.
        win._recon_line("Elapsed time: 0.4 [minutes]")
        win._recon_tick()
        assert win.scan.recon_status.text().startswith("write_sparse · Elapsed")

        # And the thread may write as fast as it likes: nothing is dropped.
        def flood() -> None:
            for i in range(500):
                win._recon_line(f"line {i}")

        writer = threading.Thread(target=flood, name="flood")
        writer.start()
        writer.join(5.0)
        assert not writer.is_alive()
        win._recon_tick()
        assert win.scan.recon_status.text().endswith("line 499")
        assert win._recon_lines == []
    finally:
        win.close()


def _ready_session(win, monkeypatch) -> None:
    """A window with a session and photographs in it, so Reconstruct is live."""
    _arm(win)
    win.scan.photos.setChecked(True)
    session = win._session
    assert session is not None
    win.scan.on_status(_status(session.session_id, photos_left=12, photos_right=12))
    assert win.scan.btn_recon.isEnabled()


def test_reconstruct_runs_off_the_gui_thread_and_writes_laser_ply_first(
        tmp_path, monkeypatch) -> None:
    win = _window(tmp_path, monkeypatch)
    try:
        from orbiter_native import recon as reconmod

        seen: dict[str, object] = {}

        def fake_run(session_dir, mode="texture-only", *, on_line=None,
                     cancel=None, **kwargs):
            seen["thread"] = threading.current_thread().name
            seen["main"] = threading.current_thread() is threading.main_thread()
            seen["mode"] = mode
            # The chain's first step reads it, so it has to exist by now.
            seen["laser_ply"] = (Path(session_dir) / "laser.ply").is_file()
            on_line("write_sparse: running")
            return None

        monkeypatch.setattr(reconmod, "run", fake_run)
        _ready_session(win, monkeypatch)
        win.scan.mode.setCurrentText("dense")
        win.scan.btn_recon.click()

        thread = win._recon_thread
        assert thread is not None
        thread.join(10.0)
        assert not thread.is_alive()
        assert seen["thread"] == "recon" and seen["main"] is False
        assert seen["mode"] == "dense"
        assert seen["laser_ply"] is True

        # The tick notices the thread ended and gives the button back.
        win._recon_tick()
        assert win._recon_thread is None
        assert win.scan.btn_recon.text() == "Reconstruct"
        assert win.scan.btn_recon.isEnabled() and win.scan.mode.isEnabled()
        assert "write_sparse" in win.scan.recon_status.text()
    finally:
        win.close()


def _blocking_run(started: threading.Event):
    """A `recon.run` that only returns when its cancel Event is set — an
    hour-long chain, in the time a test can wait."""
    def fake_run(session_dir, mode="texture-only", *, on_line=None,
                 cancel=None, **kwargs):
        assert cancel is not None
        started.set()
        cancel.wait(10.0)
        on_line("cancelled before validate_sparse")
        return None
    return fake_run


def test_abort_sets_the_cancel_event(tmp_path, monkeypatch) -> None:
    win = _window(tmp_path, monkeypatch)
    try:
        from orbiter_native import recon as reconmod

        started = threading.Event()
        monkeypatch.setattr(reconmod, "run", _blocking_run(started))
        _ready_session(win, monkeypatch)
        win.scan.btn_recon.click()
        assert started.wait(10.0)
        assert win.scan.btn_recon.text() == "Abort"
        assert not win.scan.mode.isEnabled()
        assert not win._recon_cancel.is_set()

        win.scan.btn_recon.click()                             # the same button, aborting
        assert win._recon_cancel.is_set()
        thread = win._recon_thread
        assert thread is not None
        thread.join(10.0)
        assert not thread.is_alive()
        win._recon_tick()
        assert win.scan.btn_recon.text() == "Reconstruct"
        assert "cancelled" in win.scan.recon_status.text()
    finally:
        win.close()


def test_close_event_stops_the_writer_and_the_recon_thread(
        tmp_path, monkeypatch) -> None:
    """A container outliving the window that started it is nobody's job to
    kill, and the queued photographs are the operator's own work."""
    win = _window(tmp_path, monkeypatch)
    closed = False
    try:
        from orbiter_native import recon as reconmod

        started = threading.Event()
        monkeypatch.setattr(reconmod, "run", _blocking_run(started))
        _ready_session(win, monkeypatch)
        writer = win._writer
        assert writer is not None
        stopped: list[bool] = []
        real_stop = writer.stop

        def stop(*args, **kwargs):
            stopped.append(True)
            return real_stop(*args, **kwargs)

        monkeypatch.setattr(writer, "stop", stop)
        win.scan.btn_recon.click()
        assert started.wait(10.0)
        thread = win._recon_thread
        assert thread is not None

        win.close()
        closed = True
        assert win._recon_cancel.is_set()
        assert not thread.is_alive()
        assert stopped == [True]
        assert win._writer is None and win._session is None
    finally:
        if not closed:
            win.close()
