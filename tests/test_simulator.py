"""
test_simulator.py — Unit tests verifying the physical simulator execution.
"""

import pytest
import numpy as np
import pandas as pd
from src.simulator import run_physics_simulation
from src.drive_cycles import get_wltp_urban, get_wltp_mixed, get_bangalore_urban

def test_wltp_urban_simulation():
    """Verify that the simulator executes on the WLTP Urban cycle with correct shape and no NaNs."""
    cycle = get_wltp_urban()
    df = run_physics_simulation(cycle, initial_soc=0.70)
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == len(cycle)
    
    # Check expected columns
    expected_cols = [
        "step", "speed_kmh", "acceleration_ms2", "power_demand_kw",
        "actual_battery_power_kw", "fuel_rate_lh", "engine_efficiency",
        "engine_running", "motor_power_kw", "engine_power_kw", "rpm",
        "motor_torque_nm", "engine_torque_nm", "regen_power_kw",
        "position_km", "soc", "soc_target", "predicted_speed_kmh",
        "current_segment", "ems_mode", "battery_health_score",
        "throttle", "temperature_c"
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing column: {col}"
        
    # Check for NaN values
    assert df.isna().sum().sum() == 0, "Simulation result contains NaN values"
    
    # Check SOC boundaries
    assert df["soc"].min() >= 10.0, "SOC dropped below critical limit of 10%"
    assert df["soc"].max() <= 100.0, "SOC exceeded hard maximum limit of 100%"

def test_wltp_mixed_simulation():
    """Verify simulation runs on WLTP Mixed cycle."""
    cycle = get_wltp_mixed()
    df = run_physics_simulation(cycle, initial_soc=0.70)
    assert len(df) == len(cycle)
    assert df.isna().sum().sum() == 0

def test_bangalore_urban_simulation():
    """Verify simulation runs on Bangalore Urban cycle."""
    cycle = get_bangalore_urban()
    df = run_physics_simulation(cycle, initial_soc=0.70)
    assert len(df) == len(cycle)
    assert df.isna().sum().sum() == 0
