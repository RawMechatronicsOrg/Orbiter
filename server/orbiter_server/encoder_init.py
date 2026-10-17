"""Encoder-zero bookkeeping.

The rig's encoder zeros are set **manually** from the UI (Machine config →
Step 2 · Zero encoders): level the arm and press "Zero EL", and the firmware
records the current elevation as 0; "Zero AZ" does the same for the platform
heading. The zeros live in the firmware's NVS (`POST /calibrate`).

`encoder_zero_initialized` mirrors whether the **elevation** zero has been
set: it flips True when EL is zeroed and back to False when the EL zero is
reset. Only EL matters for it — the calibration sweep and every scan move to
absolute elevations, and an unzeroed EL encoder reads an arbitrary angle
(the arm would be driven toward the cross rail). The azimuth zero only sets
which way the platform calls 0°, so it does not gate anything.
"""

from __future__ import annotations

import logging

from orbiter_model import model

log = logging.getLogger("orbiter.encoder_init")


def set_initialized(value: bool, reason: str) -> None:
    """Record whether the EL zero is set (idempotent), mirrored to the UI log."""
    if model.encoder_zero_initialized == value:
        return
    model.update(encoder_zero_initialized=value)
    msg = f"encoder zero {'initialized' if value else 'cleared'} ({reason})"
    log.info(msg)
    try:
        from ws_hub import hub
        hub.emit_log({"level": "I", "source": "api", "tag": "enc0", "msg": msg})
    except Exception:  # noqa: BLE001 — best-effort UI mirror
        pass
