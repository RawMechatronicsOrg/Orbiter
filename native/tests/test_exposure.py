"""The stripe's brightness read off its pixels, and the keeper that steers
the cameras' exposure by it and puts it back after a reopen."""

from __future__ import annotations

import json

import numpy as np

from orbiter_native.exposure import (
    CLIPPED_MAX,
    EXPOSURE,
    EXPOSURE_MIN,
    HELD,
    PEAK_HIGH,
    PEAK_LOW,
    SETTLE_S,
    VERIFY_S,
    ExposureKeeper,
    steer,
)
from orbiter_native.laser import CLIP_R, LaserParams, exposure_of, find_stripe_pixels, red_at


# ── the measurement ──────────────────────────────────────────────────────

def test_exposure_of_reads_the_peak_and_the_clipped_share() -> None:
    peak, clipped = exposure_of(np.array([100, 150, 200, 220, 230, 255, 255, 255, 255, 255], np.uint8))
    assert peak == 255.0 and clipped == 0.5
    peak, clipped = exposure_of(np.arange(100, 240, dtype=np.uint8))
    assert 220 < peak < 240 and clipped == 0.0
    assert all(v != v for v in exposure_of(np.empty(0, np.uint8)))       # NaN, NaN


def test_stripe_pixels_carry_the_red_under_them() -> None:
    bgr = np.full((40, 60, 3), 30, np.uint8)
    bgr[18:23, :, 2] = 240                                             # a stripe, not clipped
    bgr[20, 10:30, 2] = 255                                            # its core clips for a stretch
    s = find_stripe_pixels(bgr, LaserParams(redness_min=45))
    assert s.ok and len(s.r) == s.count
    assert set(np.unique(s.r)) <= {240, 255}
    peak, clipped = exposure_of(s.r)
    assert peak == 240.0 and 0.05 < clipped < 0.1        # the core row alone clips: the share says so
    pts = np.array([[15.4, 20.2], [5.0, 5.0], [200.0, -3.0]])          # the last is clipped to the frame
    assert red_at(bgr, pts).tolist() == [255, 30, 30]
    assert CLIP_R == 250


# ── steering ─────────────────────────────────────────────────────────────

def test_steer_moves_down_hard_up_gently_and_holds_in_the_band() -> None:
    assert steer(100, 255.0, 0.3) == 70                                # clipped: the big step
    assert steer(100, PEAK_HIGH + 10, 0.0) == 70                       # too bright even unclipped
    assert steer(100, 150.0, 0.0) == 121                               # dim: gently up
    assert steer(100, 220.0, 0.01) == 100                              # in the band: hold
    assert steer(EXPOSURE_MIN, 255.0, 0.5) == EXPOSURE_MIN             # floor
    assert steer(320, 100.0, 0.0) == 330                               # ceiling: a frame at 30 fps
    assert steer(100, 220.0, CLIPPED_MAX * 2) == 70                    # clipped share alone decides too
    assert PEAK_LOW < PEAK_HIGH


# ── the keeper against a fake camserver ──────────────────────────────────

class _Controls:
    """camserver as the keeper sees it: knobs per camera, a reopen counter,
    and a log of what was written."""

    host = "http://fake"

    def __init__(self) -> None:
        self.held = {"cam2": {"auto_exposure": 3, "power_line_frequency": 0, EXPOSURE: 0},
                     "cam4": {"auto_exposure": 1, "power_line_frequency": 1, EXPOSURE: 80}}
        self.reopen_count = {"cam2": 2, "cam4": 0}
        self.writes: list[tuple[str, dict]] = []
        self.fail = False

    def read(self, cam):
        if self.fail:
            raise ValueError("camera in error")
        return dict(self.held[cam])

    #: Per camera: slugs the firmware refuses, and whether the time reads back.
    refuse: dict[str, str] = {}
    blind: set[str] = set()

    def write(self, cam, values):
        self.writes.append((cam, dict(values)))
        applied, rejected = {}, {}
        for k, v in values.items():
            if k in self.refuse:
                rejected[k] = self.refuse[k]
                continue
            self.held[cam][k] = 0 if (k == EXPOSURE and cam in self.blind) else v
            applied[k] = {"requested": v, "sent": v, "readback": self.held[cam][k],
                          "honoured": True}
        return {"camera": cam, "applied": applied, "rejected": rejected}

    def reopens(self, cam):
        return self.reopen_count[cam]

    def close(self):
        pass


def _keeper(tmp_path, ctl):
    k = ExposureKeeper(tmp_path / "exposure.json", controls=ctl)
    k.configure(None, {"left": "cam2", "right": "cam4"})
    return k


def test_first_verify_adopts_the_camera_or_the_default_and_restores_what_a_reopen_forgot(tmp_path) -> None:
    ctl = _Controls()
    k = _keeper(tmp_path, ctl)
    k.tick(now=1000.0)
    # cam2 read back 0 (unset) and the wrong modes: default time, everything written.
    assert k.eyes["left"].target == 100
    assert ("cam2", {**HELD, EXPOSURE: 100}) in ctl.writes
    # cam4 held a real time in the right modes: adopted, nothing to write.
    assert k.eyes["right"].target == 80
    assert all(cam != "cam4" for cam, _ in ctl.writes)
    # A reopen: the counter moves, the camera is set again even if it reads the same.
    ctl.reopen_count["cam4"] = 1
    k.tick(now=1000.0 + VERIFY_S)
    assert ("cam4", {**HELD, EXPOSURE: 80}) in ctl.writes
    assert "reopened" in k.eyes["right"].note
    # A camera that stops answering is noted, not fatal.
    ctl.fail = True
    k.tick(now=1000.0 + 2 * VERIFY_S)
    assert "camserver" in k.eyes["left"].note


def test_auto_steers_by_fresh_stripe_peaks_and_persists(tmp_path) -> None:
    ctl = _Controls()
    k = _keeper(tmp_path, ctl)
    k.set_laser(True)
    k.tick(now=1000.0)                                                 # targets known
    import orbiter_native.exposure as ex
    ex_time = ex.time.monotonic()
    for _ in range(4):                                                 # a clipped left stripe
        k.observe("left", 255.0, 0.4)
        k.observe("right", 225.0, 0.0)                                 # the right is in the band
    k.tick(now=1000.0 + SETTLE_S)
    assert k.eyes["left"].target == 70 and ("cam2", {**HELD, EXPOSURE: 70}) in ctl.writes
    assert k.eyes["right"].target == 80
    assert "clipped" in k.eyes["left"].note
    # Not again before the sensor had time to apply it.
    k.tick(now=1000.0 + SETTLE_S + 0.5)
    assert k.eyes["left"].target == 70
    # Persisted: a new keeper starts from the same times.
    saved = json.loads((tmp_path / "exposure.json").read_text())
    assert saved["left"] == 70 and saved["right"] == 80 and saved["auto"] is True
    again = ExposureKeeper(tmp_path / "exposure.json", controls=_Controls())
    assert again.eyes["left"].target == 70
    # Off: the stripe no longer steers, the operator's time is written and kept.
    k.set_auto(False)
    k.set_target("left", 55)
    for _ in range(4):
        k.observe("left", 255.0, 0.5)
    k.tick(now=1000.0 + 3 * SETTLE_S)
    assert k.eyes["left"].target == 55 and ("cam2", {**HELD, EXPOSURE: 55}) in ctl.writes
    assert k.saturated() == {"left"}                                   # what the guide reads
    assert ex_time <= ex.time.monotonic()


def test_a_camera_that_refuses_exposure_writes_is_named_and_left_alone(tmp_path) -> None:
    ctl = _Controls()
    ctl.refuse = {EXPOSURE: "rejected by firmware"}
    k = _keeper(tmp_path, ctl)
    k.set_laser(True)
    k.tick(now=1000.0)
    assert k.eyes["left"].rejected == "rejected by firmware"
    assert "refuses exposure writes" in k.eyes["left"].note
    n = len(ctl.writes)
    for _ in range(4):
        k.observe("left", 255.0, 0.5)
    k.tick(now=1000.0 + SETTLE_S)
    assert k.eyes["left"].target == 100 and len(ctl.writes) == n     # no steering, no spam
    assert k.saturated() == {"left"}                                   # the guide still knows


def test_a_camera_that_never_reads_the_time_back_is_not_written_every_verify(tmp_path) -> None:
    ctl = _Controls()
    ctl.blind = {"cam2"}
    k = _keeper(tmp_path, ctl)
    k.tick(now=1000.0)                                                 # first write: the modes were wrong
    assert k.eyes["left"].blind and k.eyes["left"].seen == 100
    n = len(ctl.writes)
    k.tick(now=1000.0 + VERIFY_S)                                      # reads back 0 again: trusted
    k.tick(now=1000.0 + 2 * VERIFY_S)
    assert len(ctl.writes) == n
    ctl.reopen_count["cam2"] += 1                                      # a reopen still writes
    k.tick(now=1000.0 + 3 * VERIFY_S)
    assert len(ctl.writes) == n + 1


def test_camservers_own_words_reach_the_note(tmp_path) -> None:
    import httpx
    from orbiter_native.exposure import _reason
    resp = httpx.Response(400, json={"camera": "cam4", "applied": {},
                                     "rejected": {"*": "camera is not running"}},
                          request=httpx.Request("POST", "http://x/api/controls/cam4"))
    exc = httpx.HTTPStatusError("400", request=resp.request, response=resp)
    assert _reason(exc) == "camera is not running"
    assert _reason(ValueError("x")) == "ValueError"


def test_without_the_laser_or_a_camserver_the_keeper_does_nothing(tmp_path) -> None:
    k = ExposureKeeper(tmp_path / "exposure.json")                    # no controls yet
    k.observe("left", 255.0, 0.5)
    k.tick(now=5.0)                                                    # no host: nothing to do
    ctl = _Controls()
    k = _keeper(tmp_path, ctl)
    k.tick(now=1000.0)
    n = len(ctl.writes)
    for _ in range(4):
        k.observe("left", 255.0, 0.5)
    k.tick(now=1000.0 + SETTLE_S)                                      # laser off: no steering
    assert len(ctl.writes) == n and k.eyes["left"].target == 100
