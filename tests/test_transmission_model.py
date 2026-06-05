"""
test_transmission_model.py — Unit tests verifying TransmissionModel behavior.
"""

import pytest
import math
from src.transmission_model import TransmissionModel
from src.vehicle_config import WHEEL_RADIUS

def test_transmission_gear_ratios():
    """Assert transmission correctly select gears based on speed."""
    tx = TransmissionModel()
    assert tx.gear_select(0.0, 0.0) == 1
    assert tx.gear_select(25.0 / 3.6, 10.0) == 2
    assert tx.gear_select(80.0 / 3.6, 20.0) == 5
    assert tx.gear_select(120.0 / 3.6, 50.0) == 6

def test_wheel_torque_calculation():
    """Verify wheel torque calculations using gear ratios and final drive."""
    tx = TransmissionModel(gear_ratios=[3.5, 2.2, 1.5], final_drive_ratio=3.8, efficiency=0.97)
    # Gear 1: ratio = 3.5. Engine torque = 100 Nm.
    # Expected wheel torque = 100 * 3.5 * 3.8 * 0.97 = 1290.1
    expected = 100.0 * 3.5 * 3.8 * 0.97
    assert tx.wheel_torque(100.0, 1) == pytest.approx(expected)

def test_engine_rpm_calculation():
    """Verify engine RPM calculation from vehicle speed in m/s."""
    tx = TransmissionModel(gear_ratios=[3.5, 2.2, 1.5], final_drive_ratio=3.8)
    speed_ms = 10.0 # m/s
    # Wheel RPM = 10 * 60 / (2 * pi * 0.31) = 308.06
    wheel_rpm = (speed_ms * 60.0) / (2.0 * math.pi * WHEEL_RADIUS)
    # Gear 2: ratio = 2.2
    # Engine RPM = wheel_rpm * 2.2 * 3.8 = wheel_rpm * 8.36
    expected_rpm = wheel_rpm * 2.2 * 3.8
    assert tx.engine_rpm(speed_ms, 2) == pytest.approx(expected_rpm)
