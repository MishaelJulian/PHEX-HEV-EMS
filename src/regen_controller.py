"""
regen_controller.py — Phase 2: Regenerative Braking Power Controller
Author: Antigravity
Date: 2026-06-01

Calculates the recommended regenerative braking power given the current
vehicle state.  Power is linearly scaled with deceleration and tapered
by SOC to protect the battery near saturation.
"""

import logging
from dataclasses import dataclass

from src.config import (
    REGEN_MIN_DECEL_MS2,
    REGEN_MAX_POWER_KW,
    REGEN_SOC_SATURATION,
    REGEN_SOC_MIN_FOR_FULL,
)

logger = logging.getLogger(__name__)


@dataclass
class RegenResult:
    """Output of the regenerative braking controller.

    Attributes:
        regen_power_kw: Recommended regen power in kW (>= 0).
        is_active: True if regen is recommended over friction brakes.
    """

    regen_power_kw: float
    is_active: bool


def calculate_regen_power(
    speed_kmh: float,
    deceleration_ms2: float,
    soc: float,
) -> RegenResult:
    """Calculates the recommended regenerative braking power.

    Behaviour (all thresholds from config.py):
    - If deceleration_ms2 < REGEN_MIN_DECEL_MS2: return RegenResult(0.0, False)
    - Base power scales linearly with deceleration up to REGEN_MAX_POWER_KW.
    - If soc >= REGEN_SOC_SATURATION: regen_power = 0.0 (battery full).
    - If soc <= REGEN_SOC_MIN_FOR_FULL: no tapering, full base power.
    - Between REGEN_SOC_MIN_FOR_FULL and REGEN_SOC_SATURATION: linearly taper.
    - Clamp final value to [0.0, REGEN_MAX_POWER_KW].

    Args:
        speed_kmh: Current vehicle speed.  Must be >= 0.
        deceleration_ms2: Magnitude of deceleration (positive = decelerating).
        soc: State of charge as fraction [0.0, 1.0].

    Returns:
        RegenResult with recommended power and active flag.
    """
    # Minimum deceleration gate
    if deceleration_ms2 < REGEN_MIN_DECEL_MS2:
        return RegenResult(regen_power_kw=0.0, is_active=False)

    # Base power: linear from 0 at min-decel to REGEN_MAX_POWER_KW at a
    # reference deceleration of 5 m/s² (arbitrary engineering reference).
    _REFERENCE_DECEL_MS2 = 5.0
    base_power = (deceleration_ms2 / _REFERENCE_DECEL_MS2) * REGEN_MAX_POWER_KW
    base_power = min(base_power, REGEN_MAX_POWER_KW)

    # SOC-based tapering
    if soc >= REGEN_SOC_SATURATION:
        taper = 0.0
    elif soc <= REGEN_SOC_MIN_FOR_FULL:
        taper = 1.0
    else:
        # Linear taper between REGEN_SOC_MIN_FOR_FULL and REGEN_SOC_SATURATION
        taper = (REGEN_SOC_SATURATION - soc) / (REGEN_SOC_SATURATION - REGEN_SOC_MIN_FOR_FULL)

    regen_power = base_power * taper
    regen_power = max(0.0, min(regen_power, REGEN_MAX_POWER_KW))

    logger.debug(
        "Regen: decel=%.2f m/s², base=%.1f kW, taper=%.3f, final=%.1f kW",
        deceleration_ms2,
        base_power,
        taper,
        regen_power,
    )

    return RegenResult(regen_power_kw=regen_power, is_active=regen_power > 0.0)
