"""
test_engine_efficiency.py — Phase 2: Tests for engine_efficiency module.
"""

import pytest

from src.engine_efficiency import efficiency_score, is_efficient_operating_point
from src.config import ENGINE_PEAK_EFFICIENCY_RPM


class TestEfficiencyScore:
    """Tests for efficiency_score()."""

    def test_peak_rpm_returns_one(self) -> None:
        """efficiency_score(ENGINE_PEAK_EFFICIENCY_RPM) == 1.0 (within 1e-6)."""
        score = efficiency_score(ENGINE_PEAK_EFFICIENCY_RPM)
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_zero_rpm_very_low(self) -> None:
        """efficiency_score(0) < 0.01."""
        score = efficiency_score(0.0)
        assert score < 0.01

    def test_negative_rpm_raises_value_error(self) -> None:
        """efficiency_score(negative) raises ValueError."""
        with pytest.raises(ValueError):
            efficiency_score(-100.0)

    def test_monotonically_decreases_away_from_peak(self) -> None:
        """Score monotonically decreases as RPM moves away from peak."""
        peak = ENGINE_PEAK_EFFICIENCY_RPM
        offsets = [0, 100, 200, 500, 1000, 2000]
        scores_above = [efficiency_score(peak + off) for off in offsets]
        scores_below = [efficiency_score(peak - off) for off in offsets if peak - off >= 0]

        for i in range(1, len(scores_above)):
            assert scores_above[i] <= scores_above[i - 1]

        for i in range(1, len(scores_below)):
            assert scores_below[i] <= scores_below[i - 1]


class TestIsEfficientOperatingPoint:
    """Tests for is_efficient_operating_point()."""

    def test_at_peak_is_efficient(self) -> None:
        """is_efficient_operating_point at peak == True."""
        assert is_efficient_operating_point(ENGINE_PEAK_EFFICIENCY_RPM) is True

    def test_far_from_peak_not_efficient(self) -> None:
        """is_efficient_operating_point far from peak == False."""
        assert is_efficient_operating_point(500.0) is False
