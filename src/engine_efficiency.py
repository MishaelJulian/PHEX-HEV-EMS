"""
engine_efficiency.py — Phase 2: ICE Operating Point Efficiency Scorer
Author: Antigravity
Date: 2026-06-01

Provides a Gaussian efficiency model centred at the engine's peak-efficiency
RPM.  Used by the decision engine to decide whether running the ICE at the
current operating point is worthwhile.
"""

import logging
import math

from src.config import (
    ENGINE_PEAK_EFFICIENCY_RPM,
    ENGINE_EFFICIENCY_HALF_WIDTH_RPM,
)

logger = logging.getLogger(__name__)


def efficiency_score(rpm: float) -> float:
    """Returns a Gaussian efficiency score centred at ENGINE_PEAK_EFFICIENCY_RPM.

    Score is 1.0 at peak RPM, approaches 0.0 far from peak.
    Clipped to [0.0, 1.0].

    Args:
        rpm: Engine rotational speed in RPM.  Must be >= 0.

    Returns:
        Float in [0.0, 1.0].

    Raises:
        ValueError: If rpm < 0.
    """
    if rpm < 0:
        raise ValueError(f"RPM must be non-negative, got {rpm}")

    exponent = -0.5 * ((rpm - ENGINE_PEAK_EFFICIENCY_RPM) / ENGINE_EFFICIENCY_HALF_WIDTH_RPM) ** 2
    score = math.exp(exponent)
    return max(0.0, min(1.0, score))


def is_efficient_operating_point(rpm: float, threshold: float = 0.75) -> bool:
    """Returns True if efficiency_score(rpm) >= threshold.

    Args:
        rpm: Engine RPM.
        threshold: Minimum acceptable efficiency score.

    Returns:
        bool
    """
    return efficiency_score(rpm) >= threshold
