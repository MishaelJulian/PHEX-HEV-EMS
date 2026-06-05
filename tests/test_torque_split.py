"""
test_torque_split.py — Phase 3: Torque Split Controller Unit Tests
"""

import pytest
from src.engine_model import EngineState
from src.motor_model import MotorState
from src.battery_model import BatteryState
from src.torque_split import (
    TorqueSplitController, TorqueSplitMode, TorqueSplitCommand,
)


class TestTorqueSplit:
    """Tests for all 6 decision branches of the TorqueSplitController."""

    def setup_method(self):
        """Create shared powertrain instances for each test."""
        self.engine = EngineState(rpm=2500.0, torque=120.0, is_running=True)
        self.motor = MotorState()
        self.battery = BatteryState(soc=0.60)
        self.controller = TorqueSplitController(
            self.engine, self.motor, self.battery,
        )

    def test_priority_1_regen_braking(self):
        """Negative demand < -5 Nm must trigger REGEN."""
        cmd = self.controller.decide(-50.0, 60.0, 0.60, "urban")
        assert cmd.mode == TorqueSplitMode.REGEN
        assert cmd.regen_power_kw > 0.0
        assert cmd.engine_torque_nm == 0.0

    def test_priority_2_max_acceleration(self):
        """Demand >= 85% combined max torque must trigger MAX_ACCEL."""
        cmd = self.controller.decide(400.0, 50.0, 0.50, "highway")
        assert cmd.mode == TorqueSplitMode.MAX_ACCEL

    def test_priority_3_ev_only(self):
        """Low speed, urban, good SOC, low demand → EV_ONLY."""
        cmd = self.controller.decide(60.0, 30.0, 0.65, "urban")
        assert cmd.mode == TorqueSplitMode.EV_ONLY
        assert cmd.engine_torque_nm == 0.0

    def test_priority_4_charge_sustain(self):
        """Low SOC + highway speed → CHARGE_SUSTAIN."""
        cmd = self.controller.decide(100.0, 90.0, 0.20, "highway")
        assert cmd.mode == TorqueSplitMode.CHARGE_SUSTAIN
        assert cmd.charge_power_kw > 0.0

    def test_priority_5_ice_only(self):
        """Highway cruise + high SOC + moderate demand → ICE_ONLY."""
        cmd = self.controller.decide(100.0, 100.0, 0.80, "highway")
        assert cmd.mode == TorqueSplitMode.ICE_ONLY
        assert cmd.motor_torque_nm == 0.0

    def test_priority_6_hybrid_default(self):
        """Default case → HYBRID with 60/40 split."""
        cmd = self.controller.decide(150.0, 60.0, 0.50, "suburban")
        assert cmd.mode == TorqueSplitMode.HYBRID
        assert cmd.engine_torque_nm > 0.0
        assert cmd.motor_torque_nm > 0.0
        # Check approximate 60/40 split
        total = cmd.engine_torque_nm + cmd.motor_torque_nm
        engine_fraction = cmd.engine_torque_nm / total
        assert engine_fraction == pytest.approx(0.60, abs=0.01)

    def test_ev_not_triggered_at_high_speed(self):
        """EV_ONLY should NOT trigger when speed >= 50 kph."""
        cmd = self.controller.decide(60.0, 55.0, 0.65, "urban")
        assert cmd.mode != TorqueSplitMode.EV_ONLY

    def test_regen_sets_mode_on_motor(self):
        """After regen decision, motor operating_mode should be REGEN."""
        self.controller.decide(-50.0, 60.0, 0.60, "urban")
        assert self.motor.operating_mode == "REGEN"

    def test_command_has_reason(self):
        """Every decision must include a non-empty reason."""
        cmd = self.controller.decide(100.0, 70.0, 0.50, "arterial")
        assert len(cmd.reason) > 0
