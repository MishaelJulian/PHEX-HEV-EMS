"""
test_decision_engine.py — Unit tests verifying select_mode and HybridDecisionEngine safety overrides.
"""

import pytest
from src.decision_engine import select_mode, HybridDecisionEngine
from src.rule_ems import VehicleState
from src.ems_modes import EMSMode
from src.traffic_predictor import TrafficState
from src.soc_planner import SOCPlan
from src.battery_health import BatteryHealthState
from src.regen_controller import RegenResult
from src.config import SOC_CRITICAL, BATTERY_TEMP_MAX

def get_base_telemetry():
    return {
        "speed_kmh": 40.0,
        "soc": 0.50,
        "power_demand_kw": 10.0,
        "throttle": 0.5,
        "current_segment": "urban",
        "next_segment": "urban",
        "rpm": 2200.0,
    }

def get_base_state():
    return VehicleState(
        speed=50.0, acceleration=0.0, power_required_kw=20.0,
        torque_required_nm=100.0, battery_soc=50.0, battery_temp=25.0,
        aux_load_kw=1.0, grade_angle=0.0, regen_available=True,
        braking=False, traffic_condition="medium", current_segment="urban",
        next_segment="arterial"
    )

def test_safety_check_soc():
    try:
        engine = HybridDecisionEngine()
    except FileNotFoundError:
        pytest.skip("ML model not found, skipping decision engine test.")
        
    state = get_base_state()
    state.battery_soc = SOC_CRITICAL - 5.0
    
    is_safe, reason = engine._safety_check("BATTERY_ONLY", state)
    assert not is_safe
    assert "Critical SOC" in reason

def test_safety_check_thermal():
    try:
        engine = HybridDecisionEngine()
    except FileNotFoundError:
        pytest.skip("ML model not found, skipping decision engine test.")
        
    state = get_base_state()
    state.battery_temp = BATTERY_TEMP_MAX + 5.0
    
    is_safe, reason = engine._safety_check("BATTERY_ONLY", state)
    assert not is_safe
    assert "Thermal violation" in reason

def test_select_mode_rule1_urban_ev():
    """Verify C.RULE1 triggers EV mode in urban congestion."""
    telemetry = get_base_telemetry()
    telemetry["speed_kmh"] = 30.0
    telemetry["soc"] = 0.60
    telemetry["current_segment"] = "urban"
    
    mode = select_mode(
        telemetry,
        TrafficState.CONGESTED,
        SOCPlan(0.70, "Target"),
        BatteryHealthState(),
        10.0,
        RegenResult(False, 0.0)
    )
    assert mode == EMSMode.EV

def test_select_mode_rule2_soc_preserve():
    """Verify C.RULE2 triggers CHARGE_SUSTAIN when approaching urban leg on highway."""
    telemetry = get_base_telemetry()
    telemetry["speed_kmh"] = 90.0
    telemetry["soc"] = 0.60
    telemetry["current_segment"] = "highway"
    telemetry["next_segment"] = "urban"
    
    mode = select_mode(
        telemetry,
        TrafficState.FREE_FLOW,
        SOCPlan(0.70, "Target"),
        BatteryHealthState(),
        20.0,
        RegenResult(False, 0.0)
    )
    assert mode == EMSMode.CHARGE_SUSTAIN

def test_select_mode_rule6_regen():
    """Verify C.RULE6 triggers REGEN on deceleration."""
    telemetry = get_base_telemetry()
    telemetry["power_demand_kw"] = -15.0
    telemetry["soc"] = 0.60
    
    mode = select_mode(
        telemetry,
        TrafficState.FREE_FLOW,
        SOCPlan(0.50, "Target"),
        BatteryHealthState(),
        -15.0,
        RegenResult(True, 15.0)
    )
    assert mode == EMSMode.REGEN

def test_select_mode_rule7_max_accel():
    """Verify C.RULE7 triggers HYBRID_ASSIST on full throttle."""
    telemetry = get_base_telemetry()
    telemetry["throttle"] = 0.95
    telemetry["soc"] = 0.60
    
    mode = select_mode(
        telemetry,
        TrafficState.FREE_FLOW,
        SOCPlan(0.55, "Target"),
        BatteryHealthState(),
        80.0,
        RegenResult(False, 0.0)
    )
    assert mode == EMSMode.HYBRID_ASSIST

def test_select_mode_rule8_low_soc_recovery():
    """Verify C.RULE8 triggers CHARGE_SUSTAIN when SOC <= 25%."""
    telemetry = get_base_telemetry()
    telemetry["soc"] = 0.22
    
    mode = select_mode(
        telemetry,
        TrafficState.FREE_FLOW,
        SOCPlan(0.55, "Target"),
        BatteryHealthState(),
        15.0,
        RegenResult(False, 0.0)
    )
    assert mode == EMSMode.CHARGE_SUSTAIN
