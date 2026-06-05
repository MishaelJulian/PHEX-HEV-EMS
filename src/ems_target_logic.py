"""
ems_target_logic.py — Phase 4: Heuristic Smart Target Generator with Stochastic Noise
Author: Antigravity
Date: 2026-05-22

Generates heuristic target labels based on lookahead states and physical constraints,
injecting stochasticity to model real-world driver variations and controller approximations.
"""

import pandas as pd
import numpy as np
import logging
from src.config import (
    SOC_CRITICAL, SOC_LOW, SOC_HIGH, HIGH_POWER_KW, REGEN_MIN_SPEED, IDLE_SPEED_KMH,
    BATTERY_TEMP_MAX
)

# LABEL SOURCE: ems_target_logic
# LABEL VERSION: 2026-05-22-v3

logger = logging.getLogger(__name__)

def determine_smart_mode(row: pd.Series, next_segment: str) -> str:
    """
    Heuristically determine the optimal power management mode for a given state row,
    incorporating lookahead context to avoid simple rule cloning.
    """
    speed = float(row["speed"])
    power = float(row["power_required_kw"])
    soc = float(row["battery_soc"])
    braking = bool(row["braking"])
    regen_avail = bool(row["regen_available"])
    curr_seg = str(row["current_segment"])
    traffic = str(row["traffic_condition"])
    grade = float(row.get("grade_angle", 0.0))
    
    # 1. Stopped / Stationary state
    if speed <= IDLE_SPEED_KMH and not braking:
        return "IDLE_STOP"
        
    # 2. Regenerative Braking state
    if braking and regen_avail and speed > REGEN_MIN_SPEED and soc < SOC_HIGH:
        return "REGEN"
        
    # 3. Critical SOC safety state
    if soc < SOC_CRITICAL:
        return "ENGINE_CHARGE"
        
    # 4. Urban Congestion / Stop-Go ahead (Lookahead adaptation)
    city_ahead = next_segment in ["urban", "stop_go", "traffic"]
    
    if city_ahead and soc < SOC_LOW and curr_seg not in ["urban", "stop_go"]:
        return "ENGINE_CHARGE"
        
    if curr_seg in ["urban", "stop_go"] or traffic == "heavy":
        if soc > SOC_LOW:
            return "BATTERY_ONLY"
        else:
            return "ENGINE_CHARGE"
            
    # 5. Downhill ahead (Lookahead adaptation)
    if grade < -2.0 and soc >= SOC_HIGH - 15.0:
        return "BATTERY_ONLY"
        
    # 6. High power spikes
    if power > HIGH_POWER_KW:
        if soc > SOC_LOW:
            return "HYBRID_ASSIST"
        else:
            return "ENGINE_ONLY"
            
    # 7. Highway Cruise
    if curr_seg == "highway" and power > 15.0:
        return "ENGINE_ONLY"
        
    # Default fallbacks
    if soc > SOC_LOW:
        return "BATTERY_ONLY"
    else:
        return "ENGINE_ONLY"

def perturb_label(true_mode: str, row: pd.Series, rng: np.random.Generator) -> str:
    """
    Perturb the mode to simulate decision uncertainty or driver variability.
    Strictly constrains perturbations to plausible alternative decisions:
      - BATTERY_ONLY <-> HYBRID_ASSIST
      - ENGINE_ONLY <-> HYBRID_ASSIST
    Enforces physical safety constraints to never generate physically invalid labels.
    """
    soc = float(row["battery_soc"])
    temp = float(row["battery_temp"])
    speed = float(row["speed"])
    
    # 1. Enforce physical invariants (never perturb safety-critical states)
    if true_mode in ["IDLE_STOP", "REGEN", "ENGINE_CHARGE"]:
        return true_mode
        
    # 2. Prevent physical safety violations under any perturbation
    # - critical SOC -> cannot discharge (must not be BATTERY_ONLY or HYBRID_ASSIST)
    # - overheated battery -> cannot do EV-heavy (must not be BATTERY_ONLY or HYBRID_ASSIST)
    # - full battery -> cannot regen (handled by determine_smart_mode, but reinforced here)
    if true_mode == "REGEN" and soc >= SOC_HIGH:
        return "BATTERY_ONLY" if soc >= SOC_LOW else "ENGINE_ONLY"
        
    # Allowed alternatives
    choices = {
        "BATTERY_ONLY": ["HYBRID_ASSIST"],
        "ENGINE_ONLY": ["HYBRID_ASSIST"],
        "HYBRID_ASSIST": ["BATTERY_ONLY", "ENGINE_ONLY"]
    }
    
    candidates = choices.get(true_mode, [])
    valid_candidates = []
    
    for cand in candidates:
        # Critical SOC cannot support discharge
        if cand in ["BATTERY_ONLY", "HYBRID_ASSIST"] and soc < SOC_CRITICAL:
            continue
        # Overheated battery cannot support EV-heavy discharge
        if cand in ["BATTERY_ONLY", "HYBRID_ASSIST"] and temp >= BATTERY_TEMP_MAX:
            continue
        # Low SOC cannot support BATTERY_ONLY discharge
        if cand == "BATTERY_ONLY" and soc < SOC_LOW:
            continue
            
        valid_candidates.append(cand)
        
    if not valid_candidates:
        return true_mode
        
    return rng.choice(valid_candidates)

def generate_smart_targets(df: pd.DataFrame, uncertainty_rate: float = 0.12, seed: int = 42) -> pd.DataFrame:
    """
    Processes the synthetic dataset to assign smart targets using heuristic rules
    and optional stochastic perturbations.
    """
    logger.info(f"Generating smart EMS targets (uncertainty_rate={uncertainty_rate})...")
    df_targets = df.copy()
    rng = np.random.default_rng(seed)
    
    labels = []
    for idx, row in df_targets.iterrows():
        # Get next segment lookahead (if at end of trip, use current segment)
        if idx < len(df_targets) - 1 and df_targets.loc[idx, "trip_id"] == df_targets.loc[idx + 1, "trip_id"]:
            next_seg = df_targets.loc[idx + 1, "current_segment"]
        else:
            next_seg = row["current_segment"]
            
        true_mode = determine_smart_mode(row, next_seg)
        
        # Apply stochastic perturbation
        if rng.random() < uncertainty_rate:
            final_mode = perturb_label(true_mode, row, rng)
        else:
            final_mode = true_mode
            
        labels.append(final_mode)
        
    df_targets["label"] = labels
    
    dist = df_targets["label"].value_counts(normalize=True).to_dict()
    logger.info(f"Smart targets generated. Distribution: {dist}")
    
    return df_targets
