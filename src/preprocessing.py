"""
EMS AIML Layer — Phase 3: Cleaning & Preprocessing
====================================================
Reusable, modular preprocessing pipeline for all three datasets.

This module handles:
  - Missing values, duplicates, invalid entries
  - Datatype corrections
  - Categorical encoding (Label + One-Hot)
  - Feature scaling (StandardScaler)
  - Outlier detection (IQR method)
  - NASA battery: flatten nested .mat → tabular DataFrame
  - Telemetry IMU: engineer acc_magnitude, gyro_magnitude, jerk
  - Driver Behavior: encode labels, generate synthetic efficiency target

Design:
  Each preprocessing function is independent and reusable.
  All functions accept raw data, return cleaned DataFrames.
  No side effects — pure data transformations.
"""

import sys
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# SECTION 1: GENERIC CLEANING UTILITIES
# =============================================================================

def remove_duplicates(df: pd.DataFrame, name: str = "dataset") -> pd.DataFrame:
    """
    Remove exact duplicate rows.
    
    Args:
        df:   Input DataFrame.
        name: Dataset name for logging.
    
    Returns:
        DataFrame with duplicates removed.
    """
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    n_removed = n_before - len(df)
    if n_removed > 0:
        logger.info(f"  [{name}] Removed {n_removed:,} duplicate rows ({n_before:,} -> {len(df):,})")
    else:
        logger.info(f"  [{name}] No duplicate rows found")
    return df


def handle_missing_values(
    df: pd.DataFrame,
    numeric_strategy: str = "median",
    categorical_strategy: str = "mode",
    name: str = "dataset",
) -> pd.DataFrame:
    """
    Handle missing values with configurable strategies.
    
    Args:
        df:                    Input DataFrame.
        numeric_strategy:      'median', 'mean', or 'zero' for numeric columns.
        categorical_strategy:  'mode' or 'unknown' for categorical columns.
        name:                  Dataset name for logging.
    
    Returns:
        DataFrame with nulls handled.
    """
    total_nulls = df.isnull().sum().sum()
    if total_nulls == 0:
        logger.info(f"  [{name}] No missing values")
        return df

    logger.info(f"  [{name}] Handling {total_nulls:,} missing values")

    # Numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            if numeric_strategy == "median":
                fill_val = df[col].median()
            elif numeric_strategy == "mean":
                fill_val = df[col].mean()
            else:
                fill_val = 0
            df[col] = df[col].fillna(fill_val)
            logger.info(f"    {col}: filled with {numeric_strategy} = {fill_val:.4f}")

    # Categorical columns
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        if df[col].isnull().any():
            if categorical_strategy == "mode":
                fill_val = df[col].mode()[0]
            else:
                fill_val = "UNKNOWN"
            df[col] = df[col].fillna(fill_val)
            logger.info(f"    {col}: filled with '{fill_val}'")

    return df


def detect_outliers_iqr(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    factor: float = 1.5,
    name: str = "dataset",
) -> pd.DataFrame:
    """
    Detect and report outliers using the IQR method.
    Does NOT remove them — just flags and reports.
    
    The IQR method defines outliers as values below Q1 - factor*IQR
    or above Q3 + factor*IQR. factor=1.5 is standard.
    
    Args:
        df:      Input DataFrame.
        columns: Specific columns to check (None = all numeric).
        factor:  IQR multiplier (1.5 = standard, 3.0 = extreme only).
        name:    Dataset name for logging.
    
    Returns:
        Same DataFrame (unchanged — outlier info is logged only).
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    logger.info(f"  [{name}] Outlier detection (IQR × {factor}):")
    for col in columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr
        n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        pct = n_outliers / len(df) * 100
        if n_outliers > 0:
            logger.info(f"    {col:<25} → {n_outliers:>6,} outliers ({pct:.1f}%) "
                        f"[{lower:.3f}, {upper:.3f}]")

    return df


def clip_outliers(
    df: pd.DataFrame,
    columns: List[str],
    factor: float = 3.0,
    name: str = "dataset",
) -> pd.DataFrame:
    """
    Clip extreme outliers to IQR boundaries.
    Uses factor=3.0 by default (only clip truly extreme values).
    
    Args:
        df:      Input DataFrame.
        columns: Columns to clip.
        factor:  IQR multiplier for boundaries.
        name:    Dataset name for logging.
    
    Returns:
        DataFrame with extreme values clipped.
    """
    for col in columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr
        n_clipped = ((df[col] < lower) | (df[col] > upper)).sum()
        if n_clipped > 0:
            df[col] = df[col].clip(lower=lower, upper=upper)
            logger.info(f"    [{name}] Clipped {n_clipped} extreme values in {col}")

    return df


# =============================================================================
# SECTION 2: DRIVER BEHAVIOR PREPROCESSING
# =============================================================================

def generate_synthetic_efficiency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a synthetic energy efficiency target using physics-based logic.
    
    Engineering rationale:
    ─────────────────────
    Real EMS efficiency depends on multiple factors. Since we don't have
    actual fuel consumption data, we create a realistic synthetic target
    using established automotive engineering principles:
    
    1. Speed penalty: Aerodynamic drag scales with v². Optimal efficiency
       is around 50-70 km/h. Below and above this, efficiency drops.
       
    2. Throttle penalty: Higher throttle = more fuel injected = lower
       efficiency. Linear relationship assumed.
       
    3. Braking penalty: Hard braking wastes kinetic energy as heat.
       Represents energy that could have been recovered (regen braking).
       
    4. Acceleration penalty: High acceleration demands peak power,
       which is inherently less efficient than steady-state cruising.
       
    5. Steering penalty: Aggressive steering causes tire scrub and
       lateral friction losses.
       
    6. Behavior bonus/penalty: Safe driving is more efficient than
       aggressive/distracted driving.
    
    Output range: 0-100 (percentage), with noise for realism.
    
    Args:
        df: Driver Behavior DataFrame with operational columns.
    
    Returns:
        DataFrame with 'efficiency_score' column added.
    """
    logger.info("  Generating synthetic efficiency target...")

    # Normalize features to 0-1 range for formula
    speed_norm = (df["speed_kmph"] - 20) / (120 - 20)       # 20-120 km/h
    throttle_norm = (df["throttle"] - 20) / (100 - 20)       # 20-100%
    brake_norm = df["brake_pressure"] / 100                   # 0-100
    accel_norm = (df["accel_x"] - df["accel_x"].min()) / (df["accel_x"].max() - df["accel_x"].min())
    steer_norm = df["steering_angle"].abs() / 60              # 0-60 deg

    # Speed penalty: quadratic — penalize extremes, optimal around 0.4 (56 km/h)
    speed_penalty = 4 * (speed_norm - 0.4) ** 2  # min at ~56 km/h

    # Behavior encoding: Safe=0.0, Distracted=0.3, Aggressive=0.5
    behavior_penalty = df["behavior_label"].map({
        "Safe": 0.0,
        "Distracted": 0.3,
        "Aggressive": 0.5,
    }).fillna(0.3)

    # Composite efficiency formula
    efficiency_raw = (
        100
        - 15 * speed_penalty          # speed contribution (0-15)
        - 20 * throttle_norm           # throttle contribution (0-20)
        - 15 * brake_norm              # braking contribution (0-15)
        - 12 * accel_norm              # acceleration contribution (0-12)
        - 8 * steer_norm               # steering contribution (0-8)
        - 15 * behavior_penalty        # behavior contribution (0-7.5)
    )

    # Add realistic noise (±3%)
    np.random.seed(42)
    noise = np.random.normal(0, 3, size=len(df))
    efficiency_raw += noise

    # Clip to valid range
    df["efficiency_score"] = np.clip(efficiency_raw, 5, 100).round(2)

    logger.info(f"    efficiency_score range: [{df['efficiency_score'].min():.1f}, "
                f"{df['efficiency_score'].max():.1f}], "
                f"mean={df['efficiency_score'].mean():.1f}")

    return df


def preprocess_behavior_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, LabelEncoder, StandardScaler]:
    logger.info("\n" + "=" * 60)
    logger.info("  PREPROCESSING: Driver Behavior Dataset")
    logger.info("=" * 60)
    name = "Behavior"
    # 1. Duplicates
    df = remove_duplicates(df, name)
    # 2. Missing values
    df = handle_missing_values(df, name=name)
    # 3. Outlier detection
    numeric_cols = ["speed_kmph", "accel_x", "accel_y", "brake_pressure",
                    "steering_angle", "throttle", "lane_deviation",
                    "headway_distance", "reaction_time"]
    detect_outliers_iqr(df, columns=numeric_cols, name=name)
    # 4. Clip extreme outliers (only truly extreme ones, 3×IQR)
    df = clip_outliers(df, columns=numeric_cols, factor=3.0, name=name)
    # 5. Generate synthetic efficiency target
    df = generate_synthetic_efficiency(df)
    # 6. Encode behavior labels
    le = LabelEncoder()
    df["behavior_encoded"] = le.fit_transform(df["behavior_label"])
    logger.info(f"  [{name}] Label encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    # 7. Scale numeric features
    feature_cols = numeric_cols.copy()
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[feature_cols] = scaler.fit_transform(df[feature_cols])
    # Keep both scaled and unscaled versions
    # The unscaled version is useful for interpretability
    for col in feature_cols:
        df[f"{col}_scaled"] = df_scaled[col]
    logger.info(f"  [{name}] Scaled {len(feature_cols)} numeric features (StandardScaler)")
    logger.info(f"  [{name}] Final shape: {df.shape}")
    return df, le, scaler


# =============================================================================
# SECTION 3: TELEMETRY IMU PREPROCESSING
# =============================================================================

def engineer_imu_features(df: pd.DataFrame) -> pd.DataFrame:
    # Acceleration magnitude (total g-force)
    df["acc_magnitude"] = np.sqrt(df["AccX"]**2 + df["AccY"]**2 + df["AccZ"]**2)
    # Gyroscope magnitude (total angular velocity)
    df["gyro_magnitude"] = np.sqrt(df["GyroX"]**2 + df["GyroY"]**2 + df["GyroZ"]**2)
    # Jerk (rate of acceleration change) — computed as difference
    # High jerk = sudden driving inputs = aggressive behavior
    df["jerk_x"] = df["AccX"].diff().fillna(0)
    df["jerk_y"] = df["AccY"].diff().fillna(0)
    df["jerk_z"] = df["AccZ"].diff().fillna(0)
    df["jerk_magnitude"] = np.sqrt(df["jerk_x"]**2 + df["jerk_y"]**2 + df["jerk_z"]**2)
    logger.info("  [Telemetry] Engineered features: acc_magnitude, gyro_magnitude, jerk_x/y/z, jerk_magnitude")
    return df


def preprocess_telemetry_dataset(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, LabelEncoder, StandardScaler]:
    """
    Full preprocessing pipeline for the Vehicle Telemetry (IMU) dataset.
    
    Steps:
      1. Remove duplicates
      2. Handle missing values
      3. Engineer IMU features (magnitude, jerk)
      4. Encode driving class labels
      5. Detect outliers
      6. Scale features
    
    Args:
        train_df: Raw training DataFrame.
        test_df:  Raw testing DataFrame.
    
    Returns:
        Tuple of (processed_train, processed_test, label_encoder, scaler)
    """
    logger.info("\n" + "=" * 60)
    logger.info("  PREPROCESSING: Vehicle Telemetry (IMU) Dataset")
    logger.info("=" * 60)

    name = "Telemetry"

    # Process both splits
    for label, df in [("train", train_df), ("test", test_df)]:
        df = remove_duplicates(df, f"{name}-{label}")
        df = handle_missing_values(df, name=f"{name}-{label}")

    # Engineer features on both
    train_df = engineer_imu_features(train_df.copy())
    test_df = engineer_imu_features(test_df.copy())

    # Encode labels
    le = LabelEncoder()
    train_df["class_encoded"] = le.fit_transform(train_df["Class"])
    test_df["class_encoded"] = le.transform(test_df["Class"])
    logger.info(f"  [{name}] Label encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # Scale features
    feature_cols = ["AccX", "AccY", "AccZ", "GyroX", "GyroY", "GyroZ",
                    "acc_magnitude", "gyro_magnitude", "jerk_magnitude"]
    scaler = StandardScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])  # use train statistics!

    logger.info(f"  [{name}] Scaled {len(feature_cols)} features")
    logger.info(f"  [{name}] Train shape: {train_df.shape}, Test shape: {test_df.shape}")

    # Outlier detection (on scaled data)
    detect_outliers_iqr(train_df, columns=feature_cols, name=f"{name}-train")

    return train_df, test_df, le, scaler


# =============================================================================
# SECTION 4: NASA BATTERY PREPROCESSING
# =============================================================================

def flatten_nasa_battery(
    battery_data: Dict[str, Any],
    max_steps: Optional[int] = None,
) -> pd.DataFrame:
    logger.info("\n" + "=" * 60)
    logger.info("  PREPROCESSING: NASA Battery Dataset")
    logger.info("=" * 60)
    rows = []
    for batt_name, mat_data in battery_data.items():
        steps = mat_data["data"]["step"]
        n_steps = len(steps) if max_steps is None else min(len(steps), max_steps)
        logger.info(f"  [{batt_name}] Processing {n_steps:,} / {len(steps):,} steps...")
        for i in range(n_steps):
            step = steps[i]
            # Extract arrays — handle scalar edge cases
            voltage = np.atleast_1d(step["voltage"]).flatten()
            current = np.atleast_1d(step["current"]).flatten()
            temperature = np.atleast_1d(step["temperature"]).flatten()
            rel_time = np.atleast_1d(step["relativeTime"]).flatten()
            # Skip steps with insufficient data (< 3 data points)
            if len(voltage) < 3:
                continue
            # Compute aggregated features
            row = {
                "battery": batt_name,
                "step_index": i,
                "step_type": step["type"],           # C, D, or R
                "comment": step.get("comment", ""),
                # Voltage features
                "voltage_mean": voltage.mean(),
                "voltage_min": voltage.min(),
                "voltage_max": voltage.max(),
                "voltage_std": voltage.std(),
                "voltage_delta": voltage[-1] - voltage[0],  # drop during step
                # Current features
                "current_mean": current.mean(),
                "current_min": current.min(),
                "current_max": current.max(),
                "current_std": current.std(),
                "current_abs_mean": np.abs(current).mean(),
                # Temperature features
                "temp_mean": temperature.mean(),
                "temp_min": temperature.min(),
                "temp_max": temperature.max(),
                "temp_delta": temperature[-1] - temperature[0],
                # Duration
                "duration_sec": rel_time[-1] if len(rel_time) > 0 else 0,
                "n_datapoints": len(voltage),
                # Derived: discharge capacity proxy
                # capacity ≈ |mean_current| × duration (Ampere-seconds)
                "capacity_proxy": np.abs(current.mean()) * (rel_time[-1] if len(rel_time) > 0 else 0),
            }
            rows.append(row)
    df = pd.DataFrame(rows)

    # Encode step type
    type_map = {"C": 0, "D": 1, "R": 2}
    df["step_type_encoded"] = df["step_type"].map(type_map).fillna(-1).astype(int)

    # Add cycle number (approximate — every C+D+R sequence is one cycle)
    # Group by battery and assign sequential cycle numbers
    for batt in df["battery"].unique():
        mask = df["battery"] == batt
        # Every 'C' step that follows a non-C step marks a new cycle
        charge_mask = (df.loc[mask, "step_type"] == "C")
        df.loc[mask, "cycle_number"] = charge_mask.cumsum()

    logger.info(f"  [NASA] Flattened to DataFrame: {df.shape}")
    logger.info(f"  [NASA] Batteries: {df['battery'].unique().tolist()}")
    logger.info(f"  [NASA] Step types: {df['step_type'].value_counts().to_dict()}")

    # Handle any remaining nulls
    df = handle_missing_values(df, name="NASA")

    # Outlier detection on key features
    detect_outliers_iqr(
        df,
        columns=["voltage_mean", "current_mean", "temp_mean", "capacity_proxy", "duration_sec"],
        name="NASA",
    )

    return df


# =============================================================================
# SECTION 5: MASTER PREPROCESSING RUNNER
# =============================================================================

def run_full_preprocessing() -> Dict[str, Any]:
    """
    Execute the complete preprocessing pipeline for all datasets.
    
    Returns:
        Dict with all processed data and fitted transformers.
    """
    sys.stdout.reconfigure(encoding="utf-8")
    
    from config import NASA_DATA_DIR, TELEMETRY_DATA_DIR, BEHAVIOR_DATA_DIR, DATA_PROCESSED_DIR
    from data_loader import load_nasa_battery_data, load_telemetry_data, load_behavior_data

    print("\n" + "=" * 70)
    print("  EMS AIML — PHASE 3: CLEANING & PREPROCESSING")
    print("=" * 70)

    # ── Load raw data ─────────────────────────────────────────────────────
    logger.info("Loading raw datasets...")
    nasa_raw = load_nasa_battery_data(NASA_DATA_DIR)
    telemetry_raw = load_telemetry_data(TELEMETRY_DATA_DIR)
    behavior_raw = load_behavior_data(BEHAVIOR_DATA_DIR)

    results = {}

    # ── Preprocess Driver Behavior ────────────────────────────────────────
    behavior_df, behavior_le, behavior_scaler = preprocess_behavior_dataset(behavior_raw)
    results["behavior"] = {
        "df": behavior_df,
        "label_encoder": behavior_le,
        "scaler": behavior_scaler,
    }

    # ── Preprocess Telemetry IMU ──────────────────────────────────────────
    train_tel, test_tel, tel_le, tel_scaler = preprocess_telemetry_dataset(
        telemetry_raw["train"], telemetry_raw["test"]
    )
    results["telemetry"] = {
        "train": train_tel,
        "test": test_tel,
        "label_encoder": tel_le,
        "scaler": tel_scaler,
    }

    # ── Preprocess NASA Battery ───────────────────────────────────────────
    # Use max_steps=None for full data (takes ~30-60 sec per battery)
    # Use max_steps=5000 for quick prototyping
    nasa_df = flatten_nasa_battery(nasa_raw, max_steps=5000)
    results["nasa"] = {
        "df": nasa_df,
    }

    # ── Save processed datasets ──────────────────────────────────────────
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    behavior_df.to_csv(DATA_PROCESSED_DIR / "behavior_processed.csv", index=False)
    train_tel.to_csv(DATA_PROCESSED_DIR / "telemetry_train_processed.csv", index=False)
    test_tel.to_csv(DATA_PROCESSED_DIR / "telemetry_test_processed.csv", index=False)
    nasa_df.to_csv(DATA_PROCESSED_DIR / "nasa_battery_processed.csv", index=False)

    logger.info(f"\n  Saved processed datasets to: {DATA_PROCESSED_DIR}")
    logger.info(f"    behavior_processed.csv:         {behavior_df.shape}")
    logger.info(f"    telemetry_train_processed.csv:   {train_tel.shape}")
    logger.info(f"    telemetry_test_processed.csv:    {test_tel.shape}")
    logger.info(f"    nasa_battery_processed.csv:      {nasa_df.shape}")

    print("\n" + "=" * 70)
    print("  PHASE 3 COMPLETE — All datasets cleaned and preprocessed")
    print("=" * 70)

    return results


# =============================================================================
# SECTION 6: EMS FEATURE ENGINEERING (NEW)
# =============================================================================

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from src.config import (
    HIGH_POWER_KW, SOC_LOW, SOC_CRITICAL, REGEN_MIN_SPEED,
    BATTERY_TEMP_MAX, BATTERY_TEMP_MIN, TEST_SIZE, RANDOM_STATE
)

def build_preprocessing_pipeline(
    categorical_cols: list[str],
    numeric_cols: list[str]
) -> ColumnTransformer:
    """
    Build sklearn ColumnTransformer for EMS features.
    
    Numeric cols → StandardScaler
    Categorical cols → OneHotEncoder (sparse=False, handle_unknown='ignore')
    
    Returns fitted-ready transformer.
    """
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )

    return preprocessor

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Engineering EMS features...")
    # 1. Flags
    df["high_power_flag"] = (df["power_required_kw"] > HIGH_POWER_KW).astype(int)
    df["low_soc_flag"] = (df["battery_soc"] < SOC_LOW).astype(int)
    df["critical_soc_flag"] = (df["battery_soc"] < SOC_CRITICAL).astype(int)
    df["is_urban"] = (df["current_segment"] == 'urban').astype(int)
    df["is_highway"] = (df["current_segment"] == 'highway').astype(int)
    df["is_stop_go"] = (df["current_segment"] == 'stop_go').astype(int)
    df["regen_candidate"] = ((df["braking"] == True) & 
                             (df["regen_available"] == True) & 
                             (df["speed"] > REGEN_MIN_SPEED)).astype(int)
    df["next_is_highway"] = (df["next_segment"] == 'highway').astype(int)
    df["next_is_traffic"] = df["next_segment"].isin(['traffic', 'stop_go']).astype(int)
    # 2. Continuous features
    df["load_intensity_score"] = (df["power_required_kw"] + df["aux_load_kw"]) / 120.0
    # Thermal and grade
    df["thermal_stress_flag"] = ((df["battery_temp"] > BATTERY_TEMP_MAX) | 
                                 (df["battery_temp"] < BATTERY_TEMP_MIN)).astype(int)
    df["uphill_flag"] = (df["grade_angle"] > 3.0).astype(int)
    df["downhill_flag"] = (df["grade_angle"] < -3.0).astype(int)
    # Ratios and interactions
    df["power_per_speed"] = df["power_required_kw"] / (df["speed"] + 1e-3)
    df["soc_x_power"] = df["battery_soc"] * df["power_required_kw"]
    logger.info(f"Engineered features added. New shape: {df.shape}")
    return df


# =============================================================================
# SECTION 7: TEMPORAL ROLLING-WINDOW FEATURES (Phase 7)
# =============================================================================

from src.config import WINDOW_SIZE_SHORT, WINDOW_SIZE_LONG

def engineer_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add rolling-window temporal features grouped by trip_id.
    
    These features give ML models temporal context without requiring
    sequence models (LSTM/GRU). Computed per-trip so rolling windows
    don't leak across trip boundaries.
    
    NaN values at the start of each trip are forward-filled, then 
    back-filled for any remaining NaNs.
    """
    logger.info("Engineering temporal features...")
    
    has_trips = "trip_id" in df.columns
    
    if has_trips:
        groups = df.groupby("trip_id")
    else:
        # Treat entire dataset as one trip
        df["_temp_trip"] = 0
        groups = df.groupby("_temp_trip")
    
    ws = WINDOW_SIZE_SHORT  # 5
    wl = WINDOW_SIZE_LONG   # 10
    
    # --- Short window features (window=5) ---
    df["speed_ma_5"] = groups["speed"].transform(
        lambda x: x.rolling(ws, min_periods=1).mean()
    )
    df["accel_var_5"] = groups["acceleration"].transform(
        lambda x: x.rolling(ws, min_periods=1).var()
    )
    df["power_trend_5"] = groups["power_required_kw"].transform(
        lambda x: x.rolling(ws, min_periods=1).mean()
    )
    df["soc_delta_5"] = groups["battery_soc"].transform(
        lambda x: x.diff(ws)
    )
    df["braking_freq_5"] = groups["braking"].transform(
        lambda x: x.astype(int).rolling(ws, min_periods=1).sum()
    )
    df["temp_trend_5"] = groups["battery_temp"].transform(
        lambda x: x.rolling(ws, min_periods=1).mean()
    )
    
    # --- Long window features (window=10) ---
    df["speed_ma_10"] = groups["speed"].transform(
        lambda x: x.rolling(wl, min_periods=1).mean()
    )
    df["speed_std_10"] = groups["speed"].transform(
        lambda x: x.rolling(wl, min_periods=1).std()
    )
    df["soc_delta_10"] = groups["battery_soc"].transform(
        lambda x: x.diff(wl)
    )
    df["power_std_10"] = groups["power_required_kw"].transform(
        lambda x: x.rolling(wl, min_periods=1).std()
    )
    
    # --- Derived temporal features ---
    # Idle duration: rolling count of near-zero speed steps
    df["idle_count_5"] = groups["speed"].transform(
        lambda x: (x < 2.0).astype(int).rolling(ws, min_periods=1).sum()
    )
    # Stop-go frequency: count of speed sign changes in window
    df["stop_go_freq_5"] = groups["speed"].transform(
        lambda x: ((x < 5.0) & (x.shift(1) >= 5.0)).astype(int).rolling(ws, min_periods=1).sum()
    )
    # Recent regen opportunity count
    df["regen_count_5"] = groups["regen_available"].transform(
        lambda x: x.astype(int).rolling(ws, min_periods=1).sum()
    )
    
    # Fill NaNs from rolling windows
    temporal_cols = [
        "speed_ma_5", "accel_var_5", "power_trend_5", "soc_delta_5",
        "braking_freq_5", "temp_trend_5", "speed_ma_10", "speed_std_10",
        "soc_delta_10", "power_std_10", "idle_count_5", "stop_go_freq_5",
        "regen_count_5"
    ]
    for col in temporal_cols:
        df[col] = df[col].ffill().bfill().fillna(0)
    
    # Clean up temp column
    if "_temp_trip" in df.columns:
        df = df.drop(columns=["_temp_trip"])
    
    logger.info(f"Temporal features added. New shape: {df.shape}")
    return df


def prepare_train_test(
    df: pd.DataFrame,
    target_col: str = "label",
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split into train/test sets based on trip_id to prevent sequence leakage.
    Drops metadata columns (trip_id, route_template, driver_profile, etc.)
    Print class distribution in both splits before returning.
    """
    if "trip_id" not in df.columns:
        # Fallback to standard split if trip_id is not present
        metadata_cols = [target_col, "confidence", "reason", "rule_triggered",
                         "trip_id", "route_template", "driver_profile", "timestamp"]
        drop_cols = [col for col in metadata_cols if col in df.columns]
        drop_cols += [col for col in df.columns if col.startswith("adv_")]
        X = df.drop(columns=drop_cols)
        y = df[target_col]
        from sklearn.model_selection import train_test_split
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    unique_trips = df["trip_id"].unique()
    np.random.seed(random_state)
    np.random.shuffle(unique_trips)

    split_idx = int(len(unique_trips) * (1.0 - test_size))
    train_trips = unique_trips[:split_idx]
    test_trips = unique_trips[split_idx:]

    train_df = df[df["trip_id"].isin(train_trips)]
    test_df = df[df["trip_id"].isin(test_trips)]

    metadata_cols = [target_col, "confidence", "reason", "rule_triggered",
                     "trip_id", "route_template", "driver_profile", "timestamp"]
    drop_cols = [col for col in metadata_cols if col in df.columns]
    drop_cols += [col for col in df.columns if col.startswith("adv_")]

    X_train = train_df.drop(columns=drop_cols)
    X_test = test_df.drop(columns=drop_cols)
    y_train = train_df[target_col]
    y_test = test_df[target_col]

    logger.info(f"Split data by trip_id: {len(train_trips)} train trips, {len(test_trips)} test trips.")
    
    logger.info("  Train Class Distribution:")
    for label, count in y_train.value_counts(normalize=True).items():
        logger.info(f"    {label:<20} {count*100:5.1f}%")
        
    logger.info("  Test Class Distribution:")
    for label, count in y_test.value_counts(normalize=True).items():
        logger.info(f"    {label:<20} {count*100:5.1f}%")
        
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    run_full_preprocessing()

