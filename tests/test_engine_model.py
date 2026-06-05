"""
test_engine_model.py — Phase 3: Engine Model Unit Tests
"""

import math
import pytest
from src.engine_model import (
    EngineState, PEAK_EFFICIENCY, MIN_EFFICIENCY,
    PEAK_EFFICIENCY_RPM_LOW, PEAK_EFFICIENCY_RPM_HIGH, IDLE_RPM,
)


class TestEngineModel:
    """Tests for the EngineState ICE model."""

    def test_efficiency_at_2500_rpm_is_peak(self):
        """Efficiency at 2500 RPM must be PEAK_EFFICIENCY."""
        engine = EngineState(rpm=2500.0, torque=120.0, is_running=True)
        eff = engine.calculate_efficiency()
        assert eff == pytest.approx(PEAK_EFFICIENCY, rel=1e-6)

    def test_efficiency_at_idle_is_minimum(self):
        """Efficiency at idle RPM must be MIN_EFFICIENCY."""
        engine = EngineState(rpm=IDLE_RPM, torque=50.0, is_running=True)
        eff = engine.calculate_efficiency()
        assert eff == pytest.approx(MIN_EFFICIENCY, rel=1e-6)

    def test_efficiency_different_at_800_and_2500(self):
        """Efficiency at 800 RPM must differ from efficiency at 2500 RPM."""
        engine_idle = EngineState(rpm=800.0, torque=50.0, is_running=True)
        engine_peak = EngineState(rpm=2500.0, torque=50.0, is_running=True)
        eff_idle = engine_idle.calculate_efficiency()
        eff_peak = engine_peak.calculate_efficiency()
        assert eff_idle != eff_peak
        assert eff_peak > eff_idle

    def test_fuel_rate_at_zero_power(self):
        """Fuel rate must be zero when power is zero."""
        engine = EngineState(rpm=0.0, torque=0.0, is_running=True)
        engine.calculate_power()
        engine.calculate_efficiency()
        fuel = engine.calculate_fuel_consumption()
        assert fuel == 0.0

    def test_fuel_rate_when_engine_off(self):
        """Fuel rate must be zero when engine is not running."""
        engine = EngineState(rpm=2500.0, torque=120.0, is_running=False)
        engine.calculate_power()
        engine.calculate_efficiency()
        fuel = engine.calculate_fuel_consumption()
        assert fuel == 0.0

    def test_power_calculation(self):
        """Power must match P = torque * rpm * 2π / 60000."""
        engine = EngineState(rpm=3000.0, torque=150.0, is_running=True)
        power = engine.calculate_power()
        expected = (150.0 * 3000.0 * 2.0 * math.pi) / (60.0 * 1000.0)
        assert power == pytest.approx(expected, rel=1e-6)

    def test_power_zero_when_off(self):
        """Power must be zero when engine is off."""
        engine = EngineState(rpm=2500.0, torque=120.0, is_running=False)
        power = engine.calculate_power()
        assert power == 0.0

    def test_efficiency_in_peak_band(self):
        """Efficiency at any RPM in 2000–3000 must be PEAK_EFFICIENCY."""
        for rpm in [2000.0, 2200.0, 2500.0, 2800.0, 3000.0]:
            engine = EngineState(rpm=rpm, torque=100.0, is_running=True)
            eff = engine.calculate_efficiency()
            assert eff == pytest.approx(PEAK_EFFICIENCY, rel=1e-6), \
                f"Expected peak efficiency at {rpm} RPM"

    def test_efficiency_falls_above_3000(self):
        """Efficiency above 3000 RPM must be less than PEAK_EFFICIENCY."""
        engine = EngineState(rpm=4500.0, torque=100.0, is_running=True)
        eff = engine.calculate_efficiency()
        assert eff < PEAK_EFFICIENCY
        assert eff > MIN_EFFICIENCY
