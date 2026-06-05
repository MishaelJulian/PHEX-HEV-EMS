"""
test_vehicle_config.py — Unit tests verifying vehicle reference model constants.
"""

from src import vehicle_config

def test_vehicle_config_constants():
    """Assert every constant in vehicle_config matches the required specifications for Mercedes-Benz A220e."""
    assert vehicle_config.VEHICLE_MASS == 1800.0
    assert vehicle_config.BATTERY_CAPACITY == 15.0
    assert vehicle_config.INITIAL_SOC == 0.70
    assert vehicle_config.MIN_SOC == 0.20
    assert vehicle_config.MAX_SOC == 1.00
    assert vehicle_config.MOTOR_PEAK_POWER == 80.0
    assert vehicle_config.ICE_PEAK_POWER == 120.0
    assert 160.0 <= vehicle_config.COMBINED_PEAK_POWER <= 180.0
    assert vehicle_config.WHEEL_RADIUS == 0.31
    assert vehicle_config.AERO_DRAG_COEFF == 0.22
    assert vehicle_config.FRONTAL_AREA == 2.2
    assert vehicle_config.ROLLING_RESISTANCE_COEFF == 0.012
    assert vehicle_config.TOP_SPEED == 210.0
    assert vehicle_config.AIR_DENSITY == 1.225
    assert vehicle_config.GRAVITY == 9.81
