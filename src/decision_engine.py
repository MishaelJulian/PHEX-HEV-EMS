"""
decision_engine.py — Phase 7: Intelligent Hybrid Decision Engine
Author: Antigravity
Date: 2026-05-22

The core AI module. Integrates:
1. ML model for primary EMS decisions
2. PredictiveEMSController for route-aware advisory signals
3. SafetyOverrideLayer for hard battery/thermal bounds
"""

import logging
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from collections import deque

from src.config import (
    MODELS_DIR, SOC_CRITICAL, SOC_LOW, BATTERY_TEMP_MAX, BATTERY_TEMP_MIN
)
from src.rule_ems import RuleBasedEMS, VehicleState, EMSDecision
from src.preprocessing import engineer_features, engineer_temporal_features
from src.predictive_controller import PredictiveEMSController, PredictiveAdvisory
from src.safety_override import SafetyOverrideLayer
from src.demand_forecaster import DemandForecaster

logger = logging.getLogger(__name__)


class HybridDecisionEngine:
    def __init__(self, model_name: str = "ems_best_model.pkl"):
        self.rule_engine = RuleBasedEMS()
        self.predictive_controller = PredictiveEMSController()
        self.safety_layer = SafetyOverrideLayer()
        self.demand_forecaster = DemandForecaster()
        # Load ML Assets
        model_path = MODELS_DIR / model_name
        encoder_path = MODELS_DIR / "ems_label_encoder.pkl"
        if not model_path.exists() or not encoder_path.exists():
            raise FileNotFoundError(
                f"Missing ML models in {MODELS_DIR}. Run run_ems.py --train first."
            )
        self.model = joblib.load(model_path)
        self.label_encoder = joblib.load(encoder_path)
        # Load expected feature names (saved during training)
        feature_path = MODELS_DIR / "ems_feature_names.pkl"
        if feature_path.exists():
            self.expected_features = joblib.load(feature_path)
        else:
            self.expected_features = None
        # State history for online temporal feature engineering & demand forecasting
        self.state_history = deque(maxlen=15)
    def _update_history(self, state: VehicleState):
        """Append state to history, padding with duplicates if history is empty."""
        if len(self.state_history) == 0:
            for _ in range(15):
                self.state_history.append(state)
        else:
            self.state_history.append(state)
    def _get_history_dataframe(self) -> pd.DataFrame:
        """Convert deque of VehicleState to DataFrame."""
        rows = []
        for s in self.state_history:
            row_dict = {k: v for k, v in s.__dict__.items() if k != "timestamp"}
            rows.append(row_dict)
        return pd.DataFrame(rows)
    def _prepare_features(self, df_history: pd.DataFrame,
                          advisory: Optional[PredictiveAdvisory] = None) -> pd.DataFrame:
        """Apply the exact feature engineering used in training."""
        # 1. Static features
        df_eng = engineer_features(df_history.copy())

        # 2. Temporal features (computed over history window)
        df_eng = engineer_temporal_features(df_eng)

        # 3. Demand forecasting predictions
        pred_power, pred_speed, pred_accel = self.demand_forecaster.predict_future(df_history)
        df_eng["pred_future_power_required_kw"] = pred_power
        df_eng["pred_future_speed"] = pred_speed
        df_eng["pred_future_acceleration"] = pred_accel

        # Add predictive advisory features if available
        if advisory is not None:
            for key, val in advisory.to_feature_dict().items():
                df_eng[key] = val

        # Encode categorical columns
        traffic_map = {"heavy": 0, "light": 1, "medium": 2}
        segment_map = {"arterial": 0, "highway": 1, "mountain": 2,
                       "stop_go": 3, "suburban": 4, "traffic": 5, "urban": 6}

        if "traffic_condition" in df_eng.columns:
            df_eng["traffic_condition"] = df_eng["traffic_condition"].map(
                traffic_map).fillna(0).astype(int)
        if "current_segment" in df_eng.columns:
            df_eng["current_segment"] = df_eng["current_segment"].map(
                segment_map).fillna(0).astype(int)
        if "next_segment" in df_eng.columns:
            df_eng["next_segment"] = df_eng["next_segment"].map(
                segment_map).fillna(0).astype(int)

        # Ensure boolean columns are numeric
        for col in ["regen_available", "braking"]:
            if col in df_eng.columns:
                df_eng[col] = df_eng[col].astype(int)

        # Align with expected features from training
        if self.expected_features is not None:
            for col in self.expected_features:
                if col not in df_eng.columns:
                    df_eng[col] = 0
            df_eng = df_eng[self.expected_features]

        # Return only the last row representing the current timestep
        return df_eng.iloc[[-1]]

    def _safety_check(self, ml_mode: str, state: VehicleState) -> Tuple[bool, str]:
        """Backward compatible wrapper for safety checks."""
        return self.safety_layer.check_safety(ml_mode, state)

    def decide(self, state: VehicleState,
               lookahead: list = None,
               distance_remaining: float = 10.0,
               grade_ahead: float = 0.0) -> EMSDecision:
        # 1. Update history
        self._update_history(state)
        # 2. Predictive Advisory
        if lookahead is None:
            lookahead = [state.next_segment]
        advisory = self.predictive_controller.advise(
            state, lookahead, distance_remaining, grade_ahead
        )
        # 3. Prepare ML Input from history
        df_history = self._get_history_dataframe()
        # Suppress logging for row-by-row simulation
        prep_logger = logging.getLogger("src.preprocessing")
        prev_level = prep_logger.getEffectiveLevel()
        prep_logger.setLevel(logging.WARNING)
        X = self._prepare_features(df_history, advisory)
        prep_logger.setLevel(prev_level)
        # 4. ML Prediction
        pred_idx = self.model.predict(X)[0]
        ml_mode = self.label_encoder.inverse_transform([pred_idx])[0]
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X)[0]
            confidence = float(probs[pred_idx])
        else:
            confidence = 1.0
        # 5. Safety Override Check using SafetyOverrideLayer
        is_safe, violation = self.safety_layer.check_safety(ml_mode, state)
        if is_safe:
            # Build advisory reason string
            adv_notes = []
            if advisory.preserve_battery:
                adv_notes.append("preserve_battery")
            if advisory.prepare_regen:
                adv_notes.append("prepare_regen")
            if advisory.prefer_engine:
                adv_notes.append("prefer_engine")
            if advisory.charge_sustain:
                adv_notes.append("charge_sustain")
            adv_str = f" [Advisory: {', '.join(adv_notes)}]" if adv_notes else ""
            return EMSDecision(
                mode=ml_mode,
                confidence=confidence,
                rule_triggered="ML_PRIMARY",
                reason=(f"ML selected {ml_mode} ({confidence*100:.1f}% conf)."
                        f"{adv_str}"),
                source="ML",
                input_state=state
            )
        else:
            # 6. Fallback to Deterministic Rules
            logger.debug(f"SAFETY OVERRIDE: {violation}")
            rule_decision = self.rule_engine.decide(state)
            rule_decision.source = "HYBRID_OVERRIDE"
            rule_decision.reason = (f"OVERRIDE: {violation} | "
                                    f"Fallback -> {rule_decision.reason}")
            return rule_decision

# ══════════════════════════════════════════════════════════════════
# Phase 2: Standalone supervisor mode-selection function
# ══════════════════════════════════════════════════════════════════

from src.ems_modes import EMSMode
from src.traffic_predictor import TrafficState
from src.soc_planner import SOCPlan
from src.battery_health import BatteryHealthState
from src.regen_controller import RegenResult
from src.config import (
    CRITICAL_SOC_THRESHOLD,
    MAX_ACCEL_POWER_DEMAND_THRESHOLD_KW,
    MAX_ACCEL_THROTTLE_THRESHOLD,
    IDLE_SPEED_THRESHOLD_KMH,
    IDLE_SOC_MIN_FOR_EV,
    STOP_AND_GO_SPEED_THRESHOLD_KMH,
    STOP_AND_GO_MIN_SOC,
    LOW_SOC_THRESHOLD,
    ENGINE_CHARGE_POWER_DEMAND_MAX_KW,
    HIGHWAY_SPEED_THRESHOLD_KMH,
)


from src.vehicle_config import (
    MIN_SOC,
    MAX_SOC,
    ICE_PEAK_POWER,
    MOTOR_PEAK_POWER,
    COMBINED_PEAK_POWER,
)


def select_mode(
    telemetry: dict,
    traffic_state: TrafficState,
    soc_plan: SOCPlan,
    battery_health: BatteryHealthState,
    predicted_demand_kw: float,
    regen_result: RegenResult,
) -> EMSMode:
    speed = telemetry.get("speed_kmh", 0.0)
    soc = telemetry.get("soc", 0.0)
    soc_frac = soc / 100.0 if soc > 1.0 else soc
    power_demand = telemetry.get("power_demand_kw", 0.0)
    throttle = telemetry.get("throttle", 0.0)
    current_segment = telemetry.get("current_segment", "urban")
    next_segment = telemetry.get("next_segment", "urban")
    position = telemetry.get("position_km", 0.0)
    rpm = telemetry.get("rpm", 0.0)
    # 1. Regenerative Braking 
    if power_demand < 0.0:
        if soc_frac >= MAX_SOC:
            logger.info("RULE6_REGEN: Regen disabled - battery full (SOC=%.2f)", soc_frac)
        else:
            clamped_regen = min(abs(power_demand), MOTOR_PEAK_POWER)
            # 1 Wh = 1/3600 kWh
            recovered_wh = clamped_regen * (1.0 / 3600.0) * 1000.0
            logger.info("RULE6_REGEN: Regenerative braking active. Recovered Wh: %.4f", recovered_wh)
            return EMSMode.REGEN
    # 2. Maximum Acceleration 
    if throttle >= 0.90:
        logger.info("RULE7_MAX_ACCEL: Maximum acceleration requested.")
        return EMSMode.HYBRID_ASSIST
    # 3. Low SOC Recovery 
    if soc_frac <= 0.25:
        logger.info("RULE8_SOC_RECOVERY: Low SOC (SOC=%.2f) recovery active.", soc_frac)
        return EMSMode.CHARGE_SUSTAIN
    # 4. Urban Stop-and-Go EV Priority 
    if speed < 50.0 and (traffic_state == TrafficState.CONGESTED or traffic_state.value == "CONGESTED") and current_segment.lower() == "urban":
        if soc_frac <= (MIN_SOC + 0.05):
            logger.info("RULE1_SUPPRESSED_LOW_SOC: Urban EV Priority suppressed due to low SOC (SOC=%.2f)", soc_frac)
        else:
            logger.info("RULE1_EV_PRIORITY: Urban stop-and-go prioritising EV operation.")
            return EMSMode.EV
    # 5. SOC Preservation Before Urban Zones 
    upcoming_urban = next_segment.lower() in ("urban", "stop_go")
    if upcoming_urban and soc_frac < 0.80 and current_segment.lower() in ("highway", "mixed", "arterial"):
        logger.info("RULE2_SOC_PRESERVE: Upcoming urban zone - preserving SOC (SOC=%.2f).", soc_frac)
        return EMSMode.CHARGE_SUSTAIN
    # 6. Traffic-Aware Mode Planning 
    if traffic_state == TrafficState.CONGESTED or traffic_state.value == "CONGESTED":
        logger.info("RULE3_TRAFFIC_AWARE: High congestion forecast - biasing EV/Hybrid mode.")
        return EMSMode.HYBRID_ASSIST
    # 7. GPS Route Preview Mode Planning 
    if current_segment.lower() == "urban":
        logger.info("RULE4_ROUTE_PREVIEW: Urban segment - favouring EV mode.")
        return EMSMode.EV
    elif current_segment.lower() == "highway":
        logger.info("RULE4_ROUTE_PREVIEW: Highway segment - favouring ICE efficiency mode.")
        return EMSMode.ICE
    # 8. ICE Efficiency Sweet Spot (C.RULE5)
    if rpm > 0.0 and (rpm < 2000.0 or rpm > 3000.0):
        logger.info("RULE5_RPM_SWEET_SPOT: Engine outside peak band (%d RPM). Adjusting split.", int(rpm))
        if rpm > 3000.0:
            # Shed load
            return EMSMode.HYBRID_ASSIST
        else:
            # Deactivate or increase load
            return EMSMode.EV
    # Default fallback
    if soc_frac > MIN_SOC:
        return EMSMode.EV
    else:
        return EMSMode.CHARGE_SUSTAIN


