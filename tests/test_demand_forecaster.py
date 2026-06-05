import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.demand_forecaster import train_demand_forecaster, DemandForecaster

def test_demand_forecaster_train_and_predict(tmp_path):
    # Create dummy synthetic data representing sequential telemetry
    np.random.seed(42)
    n_rows = 120
    df = pd.DataFrame({
        "trip_id": [1] * 60 + [2] * 60,
        "speed": np.random.uniform(0, 100, n_rows),
        "acceleration": np.random.uniform(-3, 3, n_rows),
        "power_required_kw": np.random.uniform(-20, 100, n_rows),
        "battery_soc": np.random.uniform(20, 80, n_rows),
        "battery_temp": np.random.uniform(25, 45, n_rows),
        "aux_load_kw": np.random.uniform(0.5, 2.0, n_rows),
        "grade_angle": np.random.uniform(-5, 5, n_rows),
        "regen_available": np.random.choice([True, False], n_rows),
        "braking": np.random.choice([True, False], n_rows),
        "traffic_condition": np.random.choice(["light", "medium", "heavy"], n_rows),
        "current_segment": np.random.choice(["urban", "highway", "arterial", "stop_go"], n_rows),
        "next_segment": np.random.choice(["urban", "highway", "arterial", "traffic"], n_rows),
    })
    
    csv_path = tmp_path / "dummy_synthetic.csv"
    df.to_csv(csv_path, index=False)
    
    # Train forecasters
    metrics = train_demand_forecaster(csv_path, tmp_path, n_lags=5, forecast_step=5)
    
    assert "power" in metrics
    assert "speed" in metrics
    assert "accel" in metrics
    
    assert (tmp_path / "power_forecaster.pkl").exists()
    assert (tmp_path / "speed_forecaster.pkl").exists()
    assert (tmp_path / "accel_forecaster.pkl").exists()
    
    # Instantiate forecaster and point to temp models
    forecaster = DemandForecaster()
    forecaster.power_model_path = tmp_path / "power_forecaster.pkl"
    forecaster.speed_model_path = tmp_path / "speed_forecaster.pkl"
    forecaster.accel_model_path = tmp_path / "accel_forecaster.pkl"
    
    loaded = forecaster.load_models()
    assert loaded is True
    
    # Create history DataFrame (at least 5 rows required for lags)
    history_df = df.iloc[:10].copy()
    pred_power, pred_speed, pred_accel = forecaster.predict_future(history_df)
    
    assert isinstance(pred_power, float)
    assert isinstance(pred_speed, float)
    assert isinstance(pred_accel, float)
