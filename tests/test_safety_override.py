import pytest
from src.rule_ems import VehicleState
from src.safety_override import SafetyOverrideLayer

def test_safety_override_layer():
    safety_layer = SafetyOverrideLayer(soc_critical=15.0, soc_low=25.0, temp_max=45.0, temp_min=5.0)
    
    # Base state
    base_state = VehicleState(
        speed=40.0, acceleration=0.5, power_required_kw=10.0, torque_required_nm=35.0,
        battery_soc=50.0, battery_temp=25.0, aux_load_kw=1.0, grade_angle=0.0,
        regen_available=True, braking=False, traffic_condition="medium",
        current_segment="urban", next_segment="urban"
    )
    
    # 1. Normal state should pass
    is_safe, reason = safety_layer.check_safety("BATTERY_ONLY", base_state)
    assert is_safe is True
    assert reason == ""

    # 2. Critical SOC check
    crit_state = base_state
    crit_state.battery_soc = 10.0
    is_safe, reason = safety_layer.check_safety("BATTERY_ONLY", crit_state)
    assert is_safe is False
    assert "Critical SOC" in reason

    # 3. Low SOC check
    low_state = base_state
    low_state.battery_soc = 20.0
    is_safe, reason = safety_layer.check_safety("BATTERY_ONLY", low_state)
    assert is_safe is False
    assert "Low SOC" in reason

    # 4. Over-temp check
    hot_state = base_state
    hot_state.battery_soc = 50.0
    hot_state.battery_temp = 48.0
    is_safe, reason = safety_layer.check_safety("BATTERY_ONLY", hot_state)
    assert is_safe is False
    assert "Over-temperature" in reason

    # 5. Over-charge check
    full_state = base_state
    full_state.battery_soc = 85.0 # config SOC_HIGH is typically 80%
    full_state.battery_temp = 25.0
    is_safe, reason = safety_layer.check_safety("ENGINE_CHARGE", full_state)
    assert is_safe is False
    assert "fully charged" in reason
