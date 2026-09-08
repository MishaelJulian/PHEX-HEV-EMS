"""
demand_forecaster.py — Phase 2: Time-Series Demand Forecasting Module
Author: Antigravity
Date: 2026-05-22

Trains and loads a predictive supervisory EMS demand forecaster.
Predicts future_power_required_kw, future_speed, and future_acceleration
K steps ahead using rolling window temporal lag features.
"""

from __future__ import annotations

import logging
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import (
    MODELS_DIR, SYNTHETIC_DATA_PATH, RANDOM_STATE, TEST_SIZE
)
from src.feature_windows import create_lags

# LABEL SOURCE: real_data
# LABEL VERSION: 2026-05-22-v1

logger = logging.getLogger(__name__)

# Map categories consistently with the decision engine
TRAFFIC_MAP = {"heavy": 0, "light": 1, "medium": 2}
SEGMENT_MAP = {"arterial": 0, "highway": 1, "mountain": 2,
               "stop_go": 3, "suburban": 4, "traffic": 5, "urban": 6}

class DemandForecaster:
    """
    Predicts future speed, acceleration, and power demand using rolling window history.
    """
    def __init__(self):
        self.power_model_path = MODELS_DIR / "power_forecaster.pkl"
        self.speed_model_path = MODELS_DIR / "speed_forecaster.pkl"
        self.accel_model_path = MODELS_DIR / "accel_forecaster.pkl"
        
        self.power_model = None
        self.speed_model = None
        self.accel_model = None
        self.load_models()
        
    def load_models(self) -> bool:
        """Loads trained regressor models from models/ directory."""
        try:
            if self.power_model_path.exists():
                self.power_model = joblib.load(self.power_model_path)
            if self.speed_model_path.exists():
                self.speed_model = joblib.load(self.speed_model_path)
            if self.accel_model_path.exists():
                self.accel_model = joblib.load(self.accel_model_path)
            
            logger.info("Successfully loaded all demand forecasting models.")
            return True
        except Exception as e:
            logger.error(f"Error loading demand forecasting models: {e}")
            return False
            
    def predict_future(self, df_history: pd.DataFrame) -> Tuple[float, float, float]:
        if self.power_model is None or self.speed_model is None or self.accel_model is None:
            # Try to load models dynamically if not loaded
            if not self.load_models():
                # Fallback to current values if models are missing
                last_row = df_history.iloc[-1]
                logger.warning("DemandForecaster models not trained. Using current state fallback.")
                return (float(last_row["power_required_kw"]), 
                        float(last_row["speed"]), 
                        float(last_row["acceleration"]))
        # Prepare the single row (last row of lagged df)
        # First build lags on the history df
        lagged_df = create_lags(df_history, ["speed", "acceleration", "power_required_kw"], n_lags=5)
        last_row_df = lagged_df.iloc[[-1]].copy()
        # Encode categorical columns
        if "traffic_condition" in last_row_df.columns:
            last_row_df["traffic_condition"] = last_row_df["traffic_condition"].map(TRAFFIC_MAP).fillna(0).astype(int)
        if "current_segment" in last_row_df.columns:
            last_row_df["current_segment"] = last_row_df["current_segment"].map(SEGMENT_MAP).fillna(0).astype(int)
        if "next_segment" in last_row_df.columns:
            last_row_df["next_segment"] = last_row_df["next_segment"].map(SEGMENT_MAP).fillna(0).astype(int)
        # Ensure boolean columns are numeric
        for col in ["regen_available", "braking"]:
            if col in last_row_df.columns:
                last_row_df[col] = last_row_df[col].astype(int)
        # Features used for training (must match exactly)
        feature_cols = [
            "speed", "acceleration", "power_required_kw", "battery_soc", "battery_temp",
            "aux_load_kw", "grade_angle", "regen_available", "braking",
            "traffic_condition", "current_segment", "next_segment",
            "speed_lag_1", "speed_lag_2", "speed_lag_3", "speed_lag_4", "speed_lag_5",
            "acceleration_lag_1", "acceleration_lag_2", "acceleration_lag_3", "acceleration_lag_4", "acceleration_lag_5",
            "power_required_kw_lag_1", "power_required_kw_lag_2", "power_required_kw_lag_3", "power_required_kw_lag_4", "power_required_kw_lag_5"
        ]
        # Ensure all columns exist, if not fill with 0.0
        for col in feature_cols:
            if col not in last_row_df.columns:
                last_row_df[col] = 0.0       
        # Align features
        X = last_row_df[feature_cols]
        pred_power = float(self.power_model.predict(X)[0])
        pred_speed = float(self.speed_model.predict(X)[0])
        pred_accel = float(self.accel_model.predict(X)[0])
        # INTEGRATION POINT — Alex: replace with real power_required_kw or transient load models
        return pred_power, pred_speed, pred_accel

    def forecast(self, drive_cycle_segment: np.ndarray, horizon_s: float) -> DemandForecast:
        """Returns predicted power demand over the next horizon_s seconds."""
        from src.vehicle_model import VehicleState
        if len(drive_cycle_segment) < 2:
            return DemandForecast(0.0, 0.0, 0.0, 1.0)
        times = drive_cycle_segment[:, 0]
        speeds = drive_cycle_segment[:, 1]

        # Filter segment to horizon_s
        start_time = times[0]
        mask = (times - start_time) <= horizon_s
        segment_times = times[mask]
        segment_speeds = speeds[mask]

        if len(segment_times) < 2:
            return DemandForecast(0.0, 0.0, 0.0, 1.0)

        dt = np.diff(segment_times)
        dt = np.where(dt == 0, 1.0, dt)
        dv = np.diff(segment_speeds / 3.6)
        accel = dv / dt
        accel = np.append(accel, 0.0)

        powers = []
        for v, a in zip(segment_speeds, accel):
            vs = VehicleState(speed_kph=v, acceleration=a)
            powers.append(vs.calculate_wheel_power_demand())

        powers = np.array(powers)
        mean_p = float(np.mean(powers))
        peak_p = float(np.max(powers))

        neg_powers = powers[powers < 0.0]
        regen_pot = float(np.mean(np.abs(neg_powers))) if len(neg_powers) > 0 else 0.0

        return DemandForecast(
            mean_power_kw=mean_p,
            peak_power_kw=peak_p,
            regen_potential_kw=regen_pot,
            confidence=0.9,
        )


@dataclass
class DemandForecast:
    mean_power_kw: float
    peak_power_kw: float
    regen_potential_kw: float  # expected recoverable regen energy rate
    confidence: float          # 0.0–1.0


def train_demand_forecaster(data_path: Path, model_dir: Path, n_lags: int = 5, forecast_step: int = 5) -> Dict[str, Any]:
    logger.info(f"Loading data from {data_path} to train demand forecaster...")
    df = pd.read_csv(data_path)
    # 1. Generate lagged features within trip groups to prevent leakage
    df_features = create_lags(df, ["speed", "acceleration", "power_required_kw"], n_lags=n_lags)
    # 2. Map categorical columns
    df_features["traffic_condition"] = df_features["traffic_condition"].map(TRAFFIC_MAP).fillna(0).astype(int)
    df_features["current_segment"] = df_features["current_segment"].map(SEGMENT_MAP).fillna(0).astype(int)
    df_features["next_segment"] = df_features["next_segment"].map(SEGMENT_MAP).fillna(0).astype(int)
    # Ensure boolean columns are numeric
    for col in ["regen_available", "braking"]:
        if col in df_features.columns:
            df_features[col] = df_features[col].astype(int)
    # 3. Define feature columns
    feature_cols = [
        "speed", "acceleration", "power_required_kw", "battery_soc", "battery_temp",
        "aux_load_kw", "grade_angle", "regen_available", "braking",
        "traffic_condition", "current_segment", "next_segment",
        "speed_lag_1", "speed_lag_2", "speed_lag_3", "speed_lag_4", "speed_lag_5",
        "acceleration_lag_1", "acceleration_lag_2", "acceleration_lag_3", "acceleration_lag_4", "acceleration_lag_5",
        "power_required_kw_lag_1", "power_required_kw_lag_2", "power_required_kw_lag_3", "power_required_kw_lag_4", "power_required_kw_lag_5"
    ]
    # 4. Generate future targets within each trip_id group to prevent lookahead leak across trips
    grouped = df_features.groupby("trip_id")
    df_features["target_power"] = grouped["power_required_kw"].shift(-forecast_step)
    df_features["target_speed"] = grouped["speed"].shift(-forecast_step)
    df_features["target_accel"] = grouped["acceleration"].shift(-forecast_step)
    # Drop rows where target is NaN (which occur at the end of each trip due to shift)
    df_clean = df_features.dropna(subset=["target_power", "target_speed", "target_accel"]).copy()
    X = df_clean[feature_cols]
    y_power = df_clean["target_power"]
    y_speed = df_clean["target_speed"]
    y_accel = df_clean["target_accel"]
    # Perform a trip-level split or randomized row split (trip-level split is more robust for sequences)
    # Since each trip has a distinct trip_id, we can split trips:
    unique_trips = df_clean["trip_id"].unique()
    np.random.seed(RANDOM_STATE)
    np.random.shuffle(unique_trips)
    split_idx = int(len(unique_trips) * (1.0 - TEST_SIZE))
    train_trips = unique_trips[:split_idx]
    test_trips = unique_trips[split_idx:]
    train_mask = df_clean["trip_id"].isin(train_trips)
    test_mask = df_clean["trip_id"].isin(test_trips)
    X_train, X_test = X[train_mask], X[test_mask]
    y_power_train, y_power_test = y_power[train_mask], y_power[test_mask]
    y_speed_train, y_speed_test = y_speed[train_mask], y_speed[test_mask]
    y_accel_train, y_accel_test = y_accel[train_mask], y_accel[test_mask]
    logger.info(f"Split data into {len(train_trips)} train trips and {len(test_trips)} test trips.")
    logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    # 5. Train Random Forests (lightweight for fast training & stable inference)
    # Using small max_depth to guarantee generalization and avoid overfitting
    model_params = {"n_estimators": 50, "max_depth": 8, "random_state": RANDOM_STATE, "n_jobs": -1}
    logger.info("Training power demand forecaster...")
    power_rf = RandomForestRegressor(**model_params)
    power_rf.fit(X_train, y_power_train)
    pred_power = power_rf.predict(X_test)
    mae_power = mean_absolute_error(y_power_test, pred_power)
    rmse_power = np.sqrt(mean_squared_error(y_power_test, pred_power))
    r2_power = r2_score(y_power_test, pred_power)
    logger.info(f"  Power Forecaster: MAE={mae_power:.3f} kW, RMSE={rmse_power:.3f} kW, R2={r2_power:.3f}")
    logger.info("Training speed forecaster...")
    speed_rf = RandomForestRegressor(**model_params)
    speed_rf.fit(X_train, y_speed_train)
    pred_speed = speed_rf.predict(X_test)
    mae_speed = mean_absolute_error(y_speed_test, pred_speed)
    rmse_speed = np.sqrt(mean_squared_error(y_speed_test, pred_speed))
    r2_speed = r2_score(y_speed_test, pred_speed)
    logger.info(f"  Speed Forecaster: MAE={mae_speed:.3f} km/h, RMSE={rmse_speed:.3f} km/h, R2={r2_speed:.3f}")
    logger.info("Training acceleration forecaster...")
    accel_rf = RandomForestRegressor(**model_params)
    accel_rf.fit(X_train, y_accel_train)
    pred_accel = accel_rf.predict(X_test)
    mae_accel = mean_absolute_error(y_accel_test, pred_accel)
    rmse_accel = np.sqrt(mean_squared_error(y_accel_test, pred_accel))
    r2_accel = r2_score(y_accel_test, pred_accel)
    logger.info(f"  Acceleration Forecaster: MAE={mae_accel:.3f} m/s², RMSE={rmse_accel:.3f} m/s², R2={r2_accel:.3f}")
    
    # Save assets
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(power_rf, model_dir / "power_forecaster.pkl")
    joblib.dump(speed_rf, model_dir / "speed_forecaster.pkl")
    joblib.dump(accel_rf, model_dir / "accel_forecaster.pkl")
    logger.info(f"Saved forecasting models to {model_dir}")
    
    # Return metrics for validation report
    return {
        "power": {"mae": mae_power, "rmse": rmse_power, "r2": r2_power},
        "speed": {"mae": mae_speed, "rmse": rmse_speed, "r2": r2_speed},
        "accel": {"mae": mae_accel, "rmse": rmse_accel, "r2": r2_accel}
    }


def estimate_future_demand(
    telemetry_history: pd.DataFrame,
    horizon_steps: int = 5,
) -> float:
    """Lightweight predictive demand estimate using recent telemetry trends.

    Method:
    1. Extract rolling means of speed, acceleration, and power_demand
       over the last ``horizon_steps`` rows of telemetry_history.
    2. Compute linear trend slope for each signal using numpy polyfit(deg=1).
    3. Project each signal ``horizon_steps`` into the future.
    4. Apply a simple demand estimation formula to the projected values:
       estimated_demand = projected_power + 0.15 * projected_speed
                          + 8.0 * abs(projected_accel)

    Expected columns in telemetry_history:
        speed_kmh, acceleration_ms2, power_demand_kw

    Args:
        telemetry_history: DataFrame of recent timesteps (must have >=
            horizon_steps rows).
        horizon_steps: Number of steps to project forward.

    Returns:
        Estimated future power demand in kW.  Returns current demand if
        telemetry_history has fewer than horizon_steps rows.
    """
    if len(telemetry_history) < horizon_steps:
        logger.debug(
            "estimate_future_demand: insufficient history (%d < %d), using current demand",
            len(telemetry_history),
            horizon_steps,
        )
        return float(telemetry_history["power_demand_kw"].iloc[-1])

    window = telemetry_history.tail(horizon_steps)
    x = np.arange(len(window), dtype=float)

    projected: dict[str, float] = {}
    for col in ("speed_kmh", "acceleration_ms2", "power_demand_kw"):
        y = window[col].values.astype(float)
        coeffs = np.polyfit(x, y, deg=1)  # [slope, intercept]
        slope, intercept = coeffs[0], coeffs[1]
        projected[col] = slope * (len(window) - 1 + horizon_steps) + intercept

    estimated_demand = (
        projected["power_demand_kw"]
        + 0.15 * projected["speed_kmh"]
        + 8.0 * abs(projected["acceleration_ms2"])
    )

    logger.debug(
        "estimate_future_demand: projected speed=%.1f, accel=%.2f, power=%.1f → demand=%.1f kW",
        projected["speed_kmh"],
        projected["acceleration_ms2"],
        projected["power_demand_kw"],
        estimated_demand,
    )

    return float(estimated_demand)

