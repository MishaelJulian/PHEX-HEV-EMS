"""
ems_model.py — Phase 5: Retrain EMS ML Models with Forecasting & Advisories
Author: Antigravity
Date: 2026-05-22

Trains multiple classifiers on temporally and predictively enriched data.
Features include demand forecasts, Bangalore traffic advisories, and lags.
"""

import sys
import logging
import time
from pathlib import Path

import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score
)
from sklearn.preprocessing import LabelEncoder

from src.config import (
    SYNTHETIC_DATA_PATH, MODELS_DIR, REPORTS_DIR,
    RF_MODEL_PATH, GB_MODEL_PATH, ENCODER_PATH, SCALER_PATH,
    RF_N_ESTIMATORS, RF_MAX_DEPTH, RF_MIN_SAMPLES,
    GBM_N_ESTIMATORS, GBM_MAX_DEPTH, GBM_LEARNING_RATE,
    RANDOM_STATE, TEST_SIZE
)
from src.preprocessing import engineer_features, engineer_temporal_features, prepare_train_test
from src.demand_forecaster import DemandForecaster, TRAFFIC_MAP, SEGMENT_MAP
from src.predictive_controller import PredictiveEMSController
from src.feature_windows import create_lags
from src.rule_ems import VehicleState
from src.ems_modes import EMSMode  # Phase 2: canonical mode enum for label encoding

# LABEL SOURCE: ems_model
# LABEL VERSION: 2026-05-22-v1

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")

MODELS_TO_TRAIN = {
    "DecisionTree": DecisionTreeClassifier(
        max_depth=8,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=RANDOM_STATE
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        min_samples_leaf=RF_MIN_SAMPLES,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=GBM_N_ESTIMATORS,
        max_depth=GBM_MAX_DEPTH,
        learning_rate=GBM_LEARNING_RATE,
        random_state=RANDOM_STATE
    ),
    "XGBoost": xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        eval_metric="mlogloss",
        random_state=RANDOM_STATE
    )
}

def add_predictive_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enriches the DataFrame with demand forecasts and predictive advisory features.
    
    Args:
        df: Input DataFrame.
        
    Returns:
        Enriched DataFrame with prediction and advisory columns.
    """
    logger.info("Enriching dataset with demand forecasts and predictive advisory features...")
    df_enriched = df.copy()
    
    # 1. Add demand forecasts using vectorized predictions
    forecaster = DemandForecaster()
    if not forecaster.load_models():
        raise RuntimeError("Failed to load demand forecasting models. Run demand_forecaster.py training first!")
        
    df_lags = create_lags(df_enriched, ["speed", "acceleration", "power_required_kw"], n_lags=5)
    
    # Map categoricals for forecasting model input
    df_lags["traffic_condition_mapped"] = df_lags["traffic_condition"].map(TRAFFIC_MAP).fillna(0).astype(int)
    df_lags["current_segment_mapped"] = df_lags["current_segment"].map(SEGMENT_MAP).fillna(0).astype(int)
    df_lags["next_segment_mapped"] = df_lags["next_segment"].map(SEGMENT_MAP).fillna(0).astype(int)
    
    # Map boolean columns
    for col in ["regen_available", "braking"]:
        if col in df_lags.columns:
            df_lags[col] = df_lags[col].astype(int)
            
    feature_cols = [
        "speed", "acceleration", "power_required_kw", "battery_soc", "battery_temp",
        "aux_load_kw", "grade_angle", "regen_available", "braking",
        "traffic_condition_mapped", "current_segment_mapped", "next_segment_mapped",
        "speed_lag_1", "speed_lag_2", "speed_lag_3", "speed_lag_4", "speed_lag_5",
        "acceleration_lag_1", "acceleration_lag_2", "acceleration_lag_3", "acceleration_lag_4", "acceleration_lag_5",
        "power_required_kw_lag_1", "power_required_kw_lag_2", "power_required_kw_lag_3", "power_required_kw_lag_4", "power_required_kw_lag_5"
    ]
    
    # Rename mapped columns temporarily to match features exactly
    rename_dict = {
        "traffic_condition_mapped": "traffic_condition",
        "current_segment_mapped": "current_segment",
        "next_segment_mapped": "next_segment"
    }
    X = df_lags[feature_cols].rename(columns=rename_dict)
    
    df_enriched["pred_future_power_required_kw"] = forecaster.power_model.predict(X)
    df_enriched["pred_future_speed"] = forecaster.speed_model.predict(X)
    df_enriched["pred_future_acceleration"] = forecaster.accel_model.predict(X)
    
    # 2. Add predictive advisories
    controller = PredictiveEMSController()
    advisories = []
    
    # Group by trip_id to correctly look ahead within each trip without data leakage
    for trip_id, group in df_enriched.groupby("trip_id"):
        group = group.copy()
        n_steps = len(group)
        current_segments = group["current_segment"].tolist()
        grade_angles = group["grade_angle"].tolist()
        
        # We can construct VehicleState for each row in the group
        for idx in range(n_steps):
            row = group.iloc[idx]
            
            # Construct a VehicleState
            state = VehicleState(
                speed=float(row["speed"]),
                acceleration=float(row["acceleration"]),
                power_required_kw=float(row["power_required_kw"]),
                torque_required_nm=float(row.get("torque_required_nm", 0.0)),
                battery_soc=float(row["battery_soc"]),
                battery_temp=float(row["battery_temp"]),
                aux_load_kw=float(row["aux_load_kw"]),
                grade_angle=float(row["grade_angle"]),
                regen_available=bool(row["regen_available"]),
                braking=bool(row["braking"]),
                traffic_condition=str(row["traffic_condition"]),
                current_segment=str(row["current_segment"]),
                next_segment=str(row["next_segment"])
            )
            
            # Extract lookahead (next 3 segments to match online simulator)
            lookahead = current_segments[idx + 1: idx + 4]
            # Pad with current segment if not enough future segments
            while len(lookahead) < 3:
                lookahead.append(current_segments[-1])
                
            # Remaining distance: assume 100 meters (0.1 km) per row step
            distance_remaining = max(0.1, (n_steps - 1 - idx) * 0.1)
            
            # Grade ahead: mean of next 3 grade angles
            grade_ahead_list = grade_angles[idx + 1: idx + 4]
            grade_ahead = float(np.mean(grade_ahead_list)) if grade_ahead_list else float(row["grade_angle"])
            
            adv = controller.advise(state, lookahead, distance_remaining, grade_ahead)
            advisories.append(adv.to_feature_dict())
            
    # Convert advisory dicts to DataFrame
    df_adv = pd.DataFrame(advisories)
    
    # Conjoin the columns
    for col in df_adv.columns:
        df_enriched[col] = df_adv[col].values
        
    logger.info(f"Enrichment complete. Features added: {df_adv.columns.tolist() + ['pred_future_power_required_kw', 'pred_future_speed', 'pred_future_acceleration']}")
    return df_enriched

def train_and_evaluate_all(X_train, X_test, y_train, y_test) -> dict:
    """
    Train all models. Print comparison table.
    Return dict of {name: (model, metrics)}.
    """
    results = {}

    header = f"| {'Model':<16} | {'Accuracy':<8} | {'F1 Macro':<8} | {'Train(s)':<8} | {'Pred(ms)':<9} |"
    sep = "+" + "-"*18 + "+" + "-"*10 + "+" + "-"*10 + "+" + "-"*10 + "+" + "-"*11 + "+"
    print("\n" + sep)
    print(header)
    print(sep)

    for name, model in MODELS_TO_TRAIN.items():
        t0 = time.time()

        try:
            model.fit(X_train, y_train)
        except TypeError as e:
            if name == "XGBoost" and "use_label_encoder" in str(e):
                model = xgb.XGBClassifier(
                    n_estimators=200, max_depth=6, learning_rate=0.05,
                    eval_metric="mlogloss", random_state=RANDOM_STATE
                )
                model.fit(X_train, y_train)
            else:
                raise e

        train_time = time.time() - t0

        t1 = time.time()
        y_pred = model.predict(X_test)
        pred_time_ms = (time.time() - t1) * 1000

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")

        print(f"| {name:<16} | {acc:<8.3f} | {f1:<8.3f} | {train_time:<8.2f} | {pred_time_ms:<9.1f} |")

        results[name] = {
            "model": model,
            "metrics": {
                "accuracy": acc,
                "f1_macro": f1
            }
        }

    print(sep)
    return results

def plot_confusion_matrix(model, X_test, y_test, label_encoder, model_name: str) -> None:
    """Seaborn heatmap confusion matrix."""
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.title(f"Confusion Matrix -- {model_name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(REPORTS_DIR / f"confusion_matrix_{model_name.lower()}.png", dpi=300)
    plt.close()

def plot_feature_importance(model, feature_names: list, model_name: str) -> None:
    """Sorted horizontal bar chart top-20 features."""
    if not hasattr(model, 'feature_importances_'):
        return

    importances = model.feature_importances_
    indices = np.argsort(importances)[-20:]

    plt.figure(figsize=(10, 10))
    plt.barh(range(len(indices)), importances[indices], color='steelblue')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.title(f"Top 20 Feature Importances -- {model_name}")
    plt.xlabel("Relative Importance")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / f"feature_importance_{model_name.lower()}.png", dpi=300)
    plt.close()

def select_and_save_best_model(results: dict, label_encoder, feature_names: list) -> str:
    """Select model with best macro F1 score and save to models/."""
    best_name = max(results.keys(), key=lambda k: results[k]["metrics"]["f1_macro"])
    best_score = results[best_name]["metrics"]["f1_macro"]

    logger.info(f"Best model: {best_name} -- F1 Macro: {best_score:.4f}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Save best model as the default
    joblib.dump(results[best_name]["model"], MODELS_DIR / "ems_best_model.pkl")

    # Save explicitly requested models
    if "RandomForest" in results:
        joblib.dump(results["RandomForest"]["model"], RF_MODEL_PATH)
    if "GradientBoosting" in results:
        joblib.dump(results["GradientBoosting"]["model"], GB_MODEL_PATH)

    joblib.dump(label_encoder, ENCODER_PATH)
    joblib.dump(feature_names, MODELS_DIR / "ems_feature_names.pkl")
    joblib.dump(None, SCALER_PATH)

    return best_name

def main():
    if not SYNTHETIC_DATA_PATH.exists():
        logger.error(f"Data not found: {SYNTHETIC_DATA_PATH}. Run generate_synthetic_data.py.")
        return

    logger.info("Loading dataset...")
    df = pd.read_csv(SYNTHETIC_DATA_PATH)
    logger.info(f"  Loaded {len(df)} rows.")

    # Phase 7: Engineer static features
    df = engineer_features(df)

    # Phase 7: Engineer temporal features (rolling windows per trip)
    df = engineer_temporal_features(df)

    # Phase 5: Enrich with demand forecasts and predictive route advisories
    df = add_predictive_features(df)

    # Encode categorical columns
    cat_cols = ["traffic_condition", "current_segment", "next_segment"]
    cat_encoders = {}
    for col in cat_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            cat_encoders[col] = le

    # Convert boolean columns to int
    bool_cols = ["regen_available", "braking"]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)

    # Remove timestamp if present
    if "timestamp" in df.columns:
        df = df.drop(columns=["timestamp"])

    # Train/test split (prepare_train_test drops metadata columns internally)
    X_train, X_test, y_train, y_test = prepare_train_test(df, target_col="label")

    # Encode target labels
    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    y_test_enc = label_encoder.transform(y_test)

    feature_names = X_train.columns.tolist()
    logger.info(f"  Feature count: {len(feature_names)}")

    logger.info("Training and evaluating models...")
    results = train_and_evaluate_all(X_train, X_test, y_train_enc, y_test_enc)

    best_model_name = select_and_save_best_model(results, label_encoder, feature_names)
    best_model = results[best_model_name]["model"]

    logger.info(f"Generating plots for {best_model_name}...")
    plot_confusion_matrix(best_model, X_test, y_test_enc, label_encoder, best_model_name)
    plot_feature_importance(best_model, feature_names, best_model_name)

    # Print classification report for best model
    y_pred = best_model.predict(X_test)
    print(f"\n=== Classification Report ({best_model_name}) ===")
    print(classification_report(y_test_enc, y_pred,
                                target_names=label_encoder.classes_))

    logger.info("Model training pipeline complete.")

if __name__ == "__main__":
    main()
