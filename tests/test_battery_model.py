"""
test_battery_model.py — Phase 3: Battery Model Unit Tests
"""

import pytest
from src.battery_model import (
    BatteryState, SOC_MINIMUM, SOC_CRITICAL, SOC_MAXIMUM, SOC_UPPER_BUFFER,
)


class TestBatteryModel:
    """Tests for the BatteryState electrochemical model."""

    def test_soc_clamped_at_maximum(self):
        """SOC must not exceed SOC_MAXIMUM after charging."""
        bat = BatteryState(soc=0.98, capacity_kwh=13.8)
        bat.charge(50.0, 3600.0)  # 50 kW for 1 hour — way too much
        assert bat.soc == SOC_MAXIMUM

    def test_soc_clamped_at_critical(self):
        """SOC must not drop below SOC_CRITICAL after discharge."""
        bat = BatteryState(soc=0.15, capacity_kwh=13.8)
        bat.update_soc(-100.0)  # Massive negative delta
        assert bat.soc == SOC_CRITICAL

    def test_discharge_refused_at_minimum(self):
        """Discharge must be refused when SOC <= SOC_MINIMUM."""
        bat = BatteryState(soc=SOC_MINIMUM, capacity_kwh=13.8)
        energy = bat.discharge(10.0, 60.0)
        assert energy == 0.0

    def test_charge_returns_actual_energy(self):
        """Charge must return actual energy accepted, not requested."""
        bat = BatteryState(soc=0.99, capacity_kwh=13.8)
        headroom = (SOC_MAXIMUM - 0.99) * 13.8
        energy = bat.charge(100.0, 3600.0)  # Request way more than headroom
        assert energy == pytest.approx(headroom, abs=0.01)

    def test_regen_tracking(self):
        """Regen energy must be tracked in recovered_energy_kwh."""
        bat = BatteryState(soc=0.50, capacity_kwh=13.8)
        initial_regen = bat.recovered_energy_kwh
        bat.charge(10.0, 60.0, from_regen=True)
        assert bat.recovered_energy_kwh > initial_regen

    def test_regen_not_tracked_without_flag(self):
        """Regular charging should not increment recovered_energy_kwh."""
        bat = BatteryState(soc=0.50, capacity_kwh=13.8)
        initial_regen = bat.recovered_energy_kwh
        bat.charge(10.0, 60.0, from_regen=False)
        assert bat.recovered_energy_kwh == initial_regen

    def test_check_soc_critical(self):
        """SOC at critical level should return CRITICAL status."""
        bat = BatteryState(soc=0.08)
        assert bat.check_soc_limits() == "CRITICAL"

    def test_check_soc_low(self):
        """SOC between CRITICAL and MINIMUM should return LOW."""
        bat = BatteryState(soc=0.15)
        assert bat.check_soc_limits() == "LOW"

    def test_check_soc_normal(self):
        """SOC in normal range should return NORMAL."""
        bat = BatteryState(soc=0.60)
        assert bat.check_soc_limits() == "NORMAL"

    def test_check_soc_high(self):
        """SOC at upper buffer should return HIGH."""
        bat = BatteryState(soc=0.96)
        assert bat.check_soc_limits() == "HIGH"

    def test_estimate_available_energy(self):
        """Available energy must be (soc - SOC_MINIMUM) * capacity."""
        bat = BatteryState(soc=0.60, capacity_kwh=13.8)
        expected = (0.60 - SOC_MINIMUM) * 13.8
        assert bat.estimate_available_energy() == pytest.approx(expected, rel=1e-6)

    def test_estimate_energy_at_minimum(self):
        """Available energy must be zero at SOC_MINIMUM."""
        bat = BatteryState(soc=SOC_MINIMUM)
        assert bat.estimate_available_energy() == 0.0
