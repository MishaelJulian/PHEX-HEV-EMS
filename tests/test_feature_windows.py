import pytest
import pandas as pd
import numpy as np
from src.feature_windows import create_lags

def test_create_lags_basic():
    df = pd.DataFrame({
        "speed": [10.0, 20.0, 30.0, 40.0],
        "power": [5.0, 10.0, 15.0, 20.0]
    })
    
    # 2 lags
    df_lagged = create_lags(df, ["speed", "power"], n_lags=2)
    
    assert "speed_lag_1" in df_lagged.columns
    assert "speed_lag_2" in df_lagged.columns
    assert "power_lag_1" in df_lagged.columns
    assert "power_lag_2" in df_lagged.columns
    
    # Check shift logic
    # Row 2 (index 2): original speed=30.0, lag_1=20.0, lag_2=10.0
    assert df_lagged.loc[2, "speed_lag_1"] == 20.0
    assert df_lagged.loc[2, "speed_lag_2"] == 10.0
    
    # Check no NaN values remain
    assert not df_lagged.isnull().any().any()

def test_create_lags_with_trip_id():
    df = pd.DataFrame({
        "trip_id": [1, 1, 1, 2, 2, 2],
        "speed": [10.0, 20.0, 30.0, 100.0, 200.0, 300.0]
    })
    
    df_lagged = create_lags(df, ["speed"], n_lags=1)
    
    # Verify no leakage between trip 1 and trip 2
    # First row of trip 2 (index 3) should lag-shift to trip 2's start (bfill-ed to 100.0), NOT trip 1's end (30.0)
    assert df_lagged.loc[3, "speed_lag_1"] == 100.0
    assert df_lagged.loc[3, "speed_lag_1"] != 30.0
    
    # Checking within trips
    assert df_lagged.loc[1, "speed_lag_1"] == 10.0
    assert df_lagged.loc[2, "speed_lag_1"] == 20.0
    assert df_lagged.loc[4, "speed_lag_1"] == 100.0
    assert df_lagged.loc[5, "speed_lag_1"] == 200.0
