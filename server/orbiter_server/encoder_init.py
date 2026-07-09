"""Encoder-zero bookkeeping.

The rig's encoder zeros are set **manually** from the UI (Machine config →
Encoder zero → "Zero EL"): level the arm, click, and the firmware records the
current elevation as 0. `encoder_zero_initialized` flips True on the first
manual zero so `calibrate_geometry` knows the rig has a sane reference before
it sweeps to absolute poses.

(An earlier build auto-zeroed AZ at boot and aligned EL to the phone IMU.
That was removed: the phone-derived EL zero proved unreliable and drove the
arm to bogus elevations — `el ≈ -388°` — so the calibration sweep pointed the
camera away from the board. Zeroing is explicit and manual now.)
"""

from __future__ import annotations

import logging

from orbiter_model import model

log = logging.getLogger("orbiter.encoder_init")


def mark_initialized(reason: str) -> None:
    """Flip the persisted first-zero flag (idempotent). Called when the
    operator zeroes an encoder by hand — that is what makes the rig 'ready'
    for a calibration sweep."""
    if model.encoder_zero_initialized:
        return
    model.update(encoder_zero_initialized=True)
    log.info("encoder zero initialized (%s)", reason)
    try:
        from ws_hub import hub
        hub.emit_log({"level": "I", "source": "api", "tag": "enc0",
                      "msg": f"encoder zero initialized ({reason})"})
    except Exception:  # noqa: BLE001 — best-effort UI mirror
        pass
