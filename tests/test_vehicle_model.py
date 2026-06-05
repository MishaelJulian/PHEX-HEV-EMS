"""
test_vehicle_model.py — Phase 3: Vehicle Model Unit Tests
"""

import math
import pytest
from src.vehicle_model import VehicleState, AIR_DENSITY, GRAVITY


class TestVehicleModel:
    """Tests for the VehicleState road load model."""

    def test_zero_speed_no_aero_drag(self):
        """At zero speed, aerodynamic drag must be zero."""
        state = VehicleState(speed_kph=0.0)
        assert state.calculate_aero_drag() == 0.0

    def test_zero_speed_no_rolling_force(self):
        """At zero speed, rolling resistance must be zero."""
        state = VehicleState(speed_kph=0.0)
        assert state.calculate_rolling_force() == 0.0

    def test_aero_drag_increases_with_speed(self):
        """Aerodynamic drag must increase with speed squared."""
        state_40 = VehicleState(speed_kph=40.0)
        state_80 = VehicleState(speed_kph=80.0)
        drag_40 = state_40.calculate_aero_drag()
        drag_80 = state_80.calculate_aero_drag()
        # Speed doubles → drag quadruples (v² relationship)
        assert drag_80 == pytest.approx(drag_40 * 4.0, rel=1e-6)

    def test_high_speed_aero_drag_value(self):
        """Check aero drag at 120 kph against hand calculation."""
        state = VehicleState(speed_kph=120.0)
        v_ms = 120.0 / 3.6
        expected = 0.5 * AIR_DENSITY * 0.22 * 2.2 * v_ms * v_ms
        assert state.calculate_aero_drag() == pytest.approx(expected, rel=1e-6)

    def test_uphill_grade_positive_force(self):
        """Uphill grade must produce positive (resistive) grade force."""
        state = VehicleState(speed_kph=60.0, road_grade=5.0)
        f_grade = state.calculate_grade_force()
        assert f_grade > 0.0

    def test_downhill_grade_negative_force(self):
        """Downhill grade must produce negative (assistive) grade force."""
        state = VehicleState(speed_kph=60.0, road_grade=-3.0)
        f_grade = state.calculate_grade_force()
        assert f_grade < 0.0

    def test_flat_road_zero_grade_force(self):
        """Zero grade must produce zero grade force."""
        state = VehicleState(speed_kph=60.0, road_grade=0.0)
        f_grade = state.calculate_grade_force()
        assert f_grade == pytest.approx(0.0, abs=1e-10)

    def test_rolling_force_at_grade(self):
        """Rolling force should include cos(grade) factor."""
        state_flat = VehicleState(speed_kph=60.0, road_grade=0.0)
        state_grade = VehicleState(speed_kph=60.0, road_grade=10.0)
        roll_flat = state_flat.calculate_rolling_force()
        roll_grade = state_grade.calculate_rolling_force()
        # cos(10°) < 1 so rolling force on grade should be slightly less
        assert roll_grade < roll_flat

    def test_total_load_includes_inertia(self):
        """Total load must account for acceleration (inertia term)."""
        state_coast = VehicleState(speed_kph=60.0, acceleration=0.0)
        state_accel = VehicleState(speed_kph=60.0, acceleration=2.0)
        load_coast = state_coast.calculate_total_load()
        load_accel = state_accel.calculate_total_load()
        # Acceleration adds m*a = 1800 * 2 = 3600 N
        assert load_accel > load_coast
        assert (load_accel - load_coast) == pytest.approx(1800.0 * 2.0, rel=1e-6)

    def test_deceleration_negative_power(self):
        """Strong deceleration should yield negative wheel power (regen)."""
        state = VehicleState(speed_kph=80.0, acceleration=-3.0, road_grade=-2.0)
        p_wheel = state.calculate_wheel_power_demand()
        assert p_wheel < 0.0

    def test_wheel_power_at_zero_speed(self):
        """At zero speed, wheel power demand must be zero regardless of forces."""
        state = VehicleState(speed_kph=0.0, acceleration=0.0, road_grade=5.0)
        p_wheel = state.calculate_wheel_power_demand()
        assert p_wheel == 0.0
