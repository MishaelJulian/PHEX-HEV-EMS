import pytest
import pandas as pd
from src.ems_target_logic import determine_smart_mode, generate_smart_targets, perturb_label
from src.config import SOC_CRITICAL, SOC_LOW, BATTERY_TEMP_MAX
import numpy as np

def test_determine_smart_mode():
    # 1. Idle Stop
    row_idle = pd.Series({
        "speed": 0.0, "acceleration": 0.0, "power_required_kw": 0.0,
        "battery_soc": 50.0, "braking": False, "regen_available": True,
        "current_segment": "urban", "traffic_condition": "medium", "grade_angle": 0.0
    })
    assert determine_smart_mode(row_idle, "urban") == "IDLE_STOP"
    
    # 2. Regen Braking
    row_regen = pd.Series({
        "speed": 30.0, "acceleration": -1.5, "power_required_kw": -10.0,
        "battery_soc": 50.0, "braking": True, "regen_available": True,
        "current_segment": "urban", "traffic_condition": "medium", "grade_angle": 0.0
    })
    assert determine_smart_mode(row_regen, "urban") == "REGEN"

    # 3. Critical SOC Safety
    row_crit = pd.Series({
        "speed": 40.0, "acceleration": 0.5, "power_required_kw": 12.0,
        "battery_soc": 12.0, "braking": False, "regen_available": True,
        "current_segment": "urban", "traffic_condition": "medium", "grade_angle": 0.0
    })
    assert determine_smart_mode(row_crit, "urban") == "ENGINE_CHARGE"

    # 4. Pre-emptive charging (highway, city ahead, low SOC)
    row_pre_charge = pd.Series({
        "speed": 85.0, "acceleration": 0.0, "power_required_kw": 20.0,
        "battery_soc": 22.0, "braking": False, "regen_available": True,
        "current_segment": "highway", "traffic_condition": "light", "grade_angle": 0.0
    })
    assert determine_smart_mode(row_pre_charge, "urban") == "ENGINE_CHARGE"

    # 5. Downhill headroom prep (high SOC, negative grade)
    row_downhill = pd.Series({
        "speed": 60.0, "acceleration": 0.0, "power_required_kw": 10.0,
        "battery_soc": 75.0, "braking": False, "regen_available": True,
        "current_segment": "mountain", "traffic_condition": "light", "grade_angle": -3.0
    })
    assert determine_smart_mode(row_downhill, "mountain") == "BATTERY_ONLY"

def test_generate_smart_targets():
    df = pd.DataFrame([
        {
            "trip_id": 1, "speed": 10.0, "acceleration": 0.5, "power_required_kw": 5.0,
            "battery_soc": 40.0, "braking": False, "regen_available": True,
            "current_segment": "urban", "traffic_condition": "medium", "grade_angle": 0.0
        },
        {
            "trip_id": 1, "speed": 80.0, "acceleration": 0.0, "power_required_kw": 25.0,
            "battery_soc": 20.0, "braking": False, "regen_available": True,
            "current_segment": "highway", "traffic_condition": "light", "grade_angle": 0.0
        }
    ])
    df_res = generate_smart_targets(df)
    assert "label" in df_res.columns
    assert len(df_res) == 2

def test_perturb_label_safety_constraints():
    rng = np.random.default_rng(42)
    
    # Critical SOC case - should NEVER return BATTERY_ONLY or HYBRID_ASSIST
    row_crit = pd.Series({"battery_soc": SOC_CRITICAL - 2.0, "battery_temp": 25.0, "speed": 40.0})
    perturbed = perturb_label("ENGINE_ONLY", row_crit, rng)
    assert perturbed == "ENGINE_ONLY"  # HYBRID_ASSIST is disallowed due to critical SOC
    
    perturbed_bat = perturb_label("BATTERY_ONLY", row_crit, rng)
    assert perturbed_bat == "BATTERY_ONLY"  # true mode is preserved since candidates are blocked

    # Overheated battery case - should NEVER return BATTERY_ONLY or HYBRID_ASSIST
    row_hot = pd.Series({"battery_soc": 50.0, "battery_temp": BATTERY_TEMP_MAX + 5.0, "speed": 40.0})
    perturbed_hot = perturb_label("ENGINE_ONLY", row_hot, rng)
    assert perturbed_hot == "ENGINE_ONLY"

    # Safe case - BATTERY_ONLY should only perturb to HYBRID_ASSIST
    row_safe = pd.Series({"battery_soc": 50.0, "battery_temp": 25.0, "speed": 40.0})
    perturbed_safe = perturb_label("BATTERY_ONLY", row_safe, rng)
    assert perturbed_safe in ["BATTERY_ONLY", "HYBRID_ASSIST"]
