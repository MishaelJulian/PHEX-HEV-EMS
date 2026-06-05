"""
test_battery_health.py — Phase 2: Tests for battery_health module.
"""

import pytest

from src.battery_health import BatteryHealthState
from src.config import (
    BATTERY_TEMP_CRITICAL_C,
    BATTERY_REGEN_FREQ_WARN,
)


class TestBatteryHealthState:
    """Tests for BatteryHealthState."""

    def test_fresh_state_full_health(self) -> None:
        """Fresh state → health_score() == 1.0."""
        state = BatteryHealthState()
        assert state.health_score() == pytest.approx(1.0)

    def test_critical_temperature_near_zero(self) -> None:
        """Temperature above BATTERY_TEMP_CRITICAL_C → health_score() approaches 0."""
        state = BatteryHealthState()
        state.update(soc=0.50, temperature_c=BATTERY_TEMP_CRITICAL_C + 5.0, regen_occurred=False)
        score = state.health_score()
        assert score == pytest.approx(0.0)

    def test_cycle_count_increments(self) -> None:
        """Cycle count increments correctly across a full charge-discharge."""
        state = BatteryHealthState()
        # Rise through 0.5
        state.update(soc=0.45, temperature_c=25.0, regen_occurred=False)
        state.update(soc=0.55, temperature_c=25.0, regen_occurred=False)
        # Fall through 0.5
        state.update(soc=0.45, temperature_c=25.0, regen_occurred=False)
        # Rise again — should trigger cycle count
        state.update(soc=0.55, temperature_c=25.0, regen_occurred=False)

        assert state.cycle_count >= 1

    def test_regen_frequency_penalty(self) -> None:
        """Regen frequency penalty triggers when above BATTERY_REGEN_FREQ_WARN."""
        state = BatteryHealthState()
        # All steps are regen → frequency = 1.0 > BATTERY_REGEN_FREQ_WARN
        for _ in range(10):
            state.update(soc=0.50, temperature_c=25.0, regen_occurred=True)

        regen_fraction = state.regen_event_count / state.total_steps
        assert regen_fraction > BATTERY_REGEN_FREQ_WARN
        # Health should be < 1.0 due to regen penalty
        assert state.health_score() < 1.0

    def test_health_score_always_in_range(self) -> None:
        """health_score() always in [0.0, 1.0]."""
        state = BatteryHealthState()
        # Stress it with various conditions
        for i in range(100):
            temp = 20.0 + i * 0.3
            soc = 0.1 + (i % 10) * 0.08
            state.update(soc=soc, temperature_c=temp, regen_occurred=(i % 3 == 0))
            score = state.health_score()
            assert 0.0 <= score <= 1.0

    def test_update_accumulates_history(self) -> None:
        """update() accumulates history correctly."""
        state = BatteryHealthState()
        state.update(soc=0.80, temperature_c=25.0, regen_occurred=False)
        state.update(soc=0.75, temperature_c=26.0, regen_occurred=True)
        state.update(soc=0.70, temperature_c=27.0, regen_occurred=False)

        assert len(state.soc_history) == 3
        assert state.total_steps == 3
        assert state.regen_event_count == 1
        assert state.temperature_c == 27.0
