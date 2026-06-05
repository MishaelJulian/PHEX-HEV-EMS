"""
test_motor_model.py — Phase 3: Motor Model Unit Tests
"""

import pytest
from src.motor_model import (
    MotorState, MAX_MOTOR_TORQUE_NM, MAX_REGEN_POWER_KW,
)


class TestMotorModel:
    """Tests for the MotorState PMSM model."""

    def test_traction_delivery_normal(self):
        """Motor should deliver requested torque when below max."""
        motor = MotorState()
        actual = motor.deliver_torque(100.0, 50.0)
        assert actual == pytest.approx(100.0, rel=1e-6)

    def test_traction_torque_clamped(self):
        """Motor torque must be clamped to MAX_MOTOR_TORQUE_NM."""
        motor = MotorState()
        actual = motor.deliver_torque(300.0, 60.0)
        assert actual == pytest.approx(MAX_MOTOR_TORQUE_NM, rel=1e-6)

    def test_ev_mode_set(self):
        """Operating mode should be EV when engine not contributing."""
        motor = MotorState()
        motor.deliver_torque(80.0, 40.0, engine_contributing=False)
        assert motor.operating_mode == "EV"

    def test_hybrid_mode_set(self):
        """Operating mode should be HYBRID when engine contributing."""
        motor = MotorState()
        motor.deliver_torque(80.0, 40.0, engine_contributing=True)
        assert motor.operating_mode == "HYBRID"

    def test_regen_recovery(self):
        """Regenerate must return recovered power > 0."""
        motor = MotorState()
        recovered = motor.regenerate(30.0)
        assert recovered > 0.0
        assert motor.operating_mode == "REGEN"

    def test_regen_clamped_to_max(self):
        """Regen power must be clamped to MAX_REGEN_POWER_KW before efficiency."""
        motor = MotorState(motor_efficiency=1.0)  # Unity efficiency for easy check
        recovered = motor.regenerate(100.0)  # Way above max
        assert recovered == pytest.approx(MAX_REGEN_POWER_KW, rel=1e-6)

    def test_regen_efficiency_applied(self):
        """Recovered power must be braking_power * motor_efficiency."""
        motor = MotorState(motor_efficiency=0.90)
        recovered = motor.regenerate(20.0)
        assert recovered == pytest.approx(20.0 * 0.90, rel=1e-6)

    def test_regen_energy_tracked(self):
        """Cumulative regen energy must increase on each regenerate call."""
        motor = MotorState()
        initial = motor.total_regen_energy_kwh
        motor.regenerate(25.0)
        assert motor.total_regen_energy_kwh > initial

    def test_power_positive_on_traction(self):
        """Motor power must be positive during traction."""
        motor = MotorState()
        motor.deliver_torque(100.0, 60.0)
        assert motor.motor_power > 0.0
