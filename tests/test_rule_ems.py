import pytest
from src.rule_ems import RuleBasedEMS, VehicleState
from src.config import SOC_CRITICAL, REGEN_MIN_SPEED, HIGH_POWER_KW
from src.ems_modes import EMSMode

@pytest.fixture
def rule_engine():
    return RuleBasedEMS()

def get_base_state():
    return VehicleState(
        speed=50.0, acceleration=0.0, power_required_kw=20.0,
        torque_required_nm=100.0, battery_soc=50.0, battery_temp=25.0,
        aux_load_kw=1.0, grade_angle=0.0, regen_available=True,
        braking=False, traffic_condition="medium", current_segment="urban",
        next_segment="arterial"
    )

def test_critical_soc_rule(rule_engine):
    state = get_base_state()
    state.battery_soc = SOC_CRITICAL - 5.0
    
    decision = rule_engine.decide(state)
    assert decision.mode == EMSMode.CHARGE_SUSTAIN.value
    assert "R3_CRITICAL_SOC" in decision.rule_triggered

def test_regen_braking_rule(rule_engine):
    state = get_base_state()
    state.braking = True
    state.regen_available = True
    state.speed = REGEN_MIN_SPEED + 10.0
    state.battery_soc = 50.0
    
    decision = rule_engine.decide(state)
    assert decision.mode == EMSMode.REGEN.value

def test_idle_stop_rule(rule_engine):
    state = get_base_state()
    state.speed = 0.0
    state.braking = False
    
    decision = rule_engine.decide(state)
    assert decision.mode == EMSMode.EV.value

