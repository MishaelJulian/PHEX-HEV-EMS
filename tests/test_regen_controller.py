"""
test_regen_controller.py — Phase 2: Tests for regen_controller module.
"""

import pytest

from src.regen_controller import calculate_regen_power, RegenResult
from src.config import (
    REGEN_MIN_DECEL_MS2,
    REGEN_MAX_POWER_KW,
    REGEN_SOC_SATURATION,
    REGEN_SOC_MIN_FOR_FULL,
)


class TestCalculateRegenPower:
    """Tests for calculate_regen_power()."""

    def test_below_min_decel_returns_inactive(self) -> None:
        """decel < REGEN_MIN_DECEL_MS2 → regen_power_kw == 0, is_active == False."""
        result = calculate_regen_power(
            speed_kmh=50.0,
            deceleration_ms2=REGEN_MIN_DECEL_MS2 - 0.1,
            soc=0.50,
        )
        assert result.regen_power_kw == 0.0
        assert result.is_active is False

    def test_high_soc_returns_zero_power(self) -> None:
        """High SOC (>= REGEN_SOC_SATURATION) → regen_power_kw == 0."""
        result = calculate_regen_power(
            speed_kmh=60.0,
            deceleration_ms2=3.0,
            soc=REGEN_SOC_SATURATION,
        )
        assert result.regen_power_kw == 0.0

    def test_low_soc_high_decel_full_power(self) -> None:
        """Low SOC (<= REGEN_SOC_MIN_FOR_FULL) + high decel → full power."""
        result = calculate_regen_power(
            speed_kmh=80.0,
            deceleration_ms2=5.0,
            soc=REGEN_SOC_MIN_FOR_FULL,
        )
        # At 5.0 m/s² reference decel, base power = REGEN_MAX_POWER_KW, taper = 1.0
        assert result.regen_power_kw == pytest.approx(REGEN_MAX_POWER_KW)
        assert result.is_active is True

    def test_mid_soc_tapered_power(self) -> None:
        """Mid SOC → tapered power between 0 and max."""
        mid_soc = (REGEN_SOC_MIN_FOR_FULL + REGEN_SOC_SATURATION) / 2.0
        result = calculate_regen_power(
            speed_kmh=60.0,
            deceleration_ms2=3.0,
            soc=mid_soc,
        )
        assert 0.0 < result.regen_power_kw < REGEN_MAX_POWER_KW
        assert result.is_active is True

    def test_power_never_exceeds_max(self) -> None:
        """regen_power_kw never exceeds REGEN_MAX_POWER_KW."""
        result = calculate_regen_power(
            speed_kmh=100.0,
            deceleration_ms2=10.0,  # very high decel
            soc=0.10,
        )
        assert result.regen_power_kw <= REGEN_MAX_POWER_KW

    def test_power_always_non_negative(self) -> None:
        """regen_power_kw is always >= 0."""
        for soc in [0.0, 0.2, 0.5, 0.8, 1.0]:
            for decel in [0.0, 0.5, 2.0, 5.0]:
                result = calculate_regen_power(
                    speed_kmh=50.0,
                    deceleration_ms2=decel,
                    soc=soc,
                )
                assert result.regen_power_kw >= 0.0
