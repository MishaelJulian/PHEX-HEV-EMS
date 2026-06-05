import pytest
import pandas as pd
from src.preprocessing import engineer_features, engineer_temporal_features
from src.config import HIGH_POWER_KW

def _make_trip_df(n=20):
    """Create a small fake trip for testing temporal features."""
    import numpy as np
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "speed": rng.uniform(10, 80, n),
        "acceleration": rng.uniform(-2, 2, n),
        "power_required_kw": rng.uniform(5, 50, n),
        "torque_required_nm": rng.uniform(50, 200, n),
        "battery_soc": np.linspace(80, 60, n),
        "battery_temp": rng.uniform(20, 35, n),
        "aux_load_kw": rng.uniform(0.5, 3, n),
        "grade_angle": rng.uniform(-2, 2, n),
        "regen_available": [False] * n,
        "braking": [False] * n,
        "traffic_condition": ["medium"] * n,
        "current_segment": ["urban"] * n,
        "next_segment": ["arterial"] * n,
        "trip_id": [0] * n,
    })

def test_engineer_features():
    df = _make_trip_df()
    df_eng = engineer_features(df)
    
    assert "high_power_flag" in df_eng.columns
    assert "is_urban" in df_eng.columns
    assert "power_per_speed" in df_eng.columns

def test_engineer_temporal_features():
    df = _make_trip_df()
    df_temp = engineer_temporal_features(df)
    
    assert "speed_ma_5" in df_temp.columns
    assert "accel_var_5" in df_temp.columns
    assert "soc_delta_5" in df_temp.columns
    assert "braking_freq_5" in df_temp.columns
    assert "idle_count_5" in df_temp.columns
    assert "regen_count_5" in df_temp.columns
    
    # No NaNs should remain after filling
    temporal_cols = ["speed_ma_5", "accel_var_5", "soc_delta_5"]
    for col in temporal_cols:
        assert not df_temp[col].isnull().any(), f"NaN found in {col}"
