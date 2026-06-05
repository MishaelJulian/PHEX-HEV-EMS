"""
ems_modes.py — Phase 2: Canonical EMS Mode Taxonomy
Author: Antigravity
Date: 2026-06-01

Defines the single authoritative EMSMode enum for the entire project.
Every module that references an EMS operating mode must import from this
file — no module may define its own mode strings.
"""

from enum import Enum


class EMSMode(Enum):
    """Enumeration of all valid EMS operating modes.

    Each member's value is the canonical string representation used in
    logging, serialisation, and display.
    """

    EV = "EV"                          # Full electric drive, ICE off
    ICE = "ICE"                        # ICE only, motor off
    HYBRID_ASSIST = "HYBRID_ASSIST"    # ICE + motor, maximum power available
    REGEN = "REGEN"                    # Regenerative braking, motor as generator
    CHARGE_SUSTAIN = "CHARGE_SUSTAIN"  # ICE maintains current SOC
    CHARGE_DEPLETING = "CHARGE_DEPLETING"  # Battery depletes toward target SOC
