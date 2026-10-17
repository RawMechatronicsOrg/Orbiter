"""Encoder-zero bookkeeping.

The rig's encoder zeros are set **manually** from the UI (Machine config →
Step 2 · Zero encoders): level the arm and press "Zero EL", and the firmware
records the current elevation as 0; "Zero AZ" does the same for the platform
heading. The zeros live in the firmware's NVS (`POST /calibrate`), and the
firmware reports them in `GET /state` → `calibration`.

`model.encoder_zero_initialized` says whether the **elevation** zero is set,
read from those reported offsets: an EL zero still at the firmware default
has never been set (or was reset, or the NVS was erased). Only EL matters —
the calibration sweep and every scan move to absolute elevations, and an
unzeroed EL encoder reads an arbitrary angle (the arm would be driven toward
the cross rail). The azimuth zero only names the platform heading 0°.
"""

from __future__ import annotations

from typing import Any

#: The EL zero the firmware uses without a stored one, and the one
#: `/calibrate mode=reset` restores (motion.c, CAL_DEFAULT_EL_RAW).
FIRMWARE_DEFAULT_EL_ZERO_RAW = 138.296


def el_zero_set(encoder_zero: dict[str, Any] | None) -> bool | None:
    """Whether the firmware holds an EL zero of its own; None when it has
    not reported its offsets (yet)."""
    value = (encoder_zero or {}).get("el_zero_raw_deg")
    if not isinstance(value, (int, float)):
        return None
    # The firmware reports the offsets with 3 decimals.
    return abs(float(value) - FIRMWARE_DEFAULT_EL_ZERO_RAW) > 0.0005
