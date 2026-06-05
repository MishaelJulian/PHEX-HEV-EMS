import pytest
import numpy as np
import pandas as pd
from src.generate_synthetic_data import DriveCycleGenerator, ProbabilisticLabeler
from src.config import SYNTHETIC_RANDOM_SEED

def test_drive_cycle_produces_dataframe():
    rng = np.random.default_rng(SYNTHETIC_RANDOM_SEED)
    gen = DriveCycleGenerator(rng, "smooth", "commute")
    df = gen.generate()
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 50  # should produce substantial trip
    assert not df.isnull().values.any()

def test_soc_evolves_over_time():
    rng = np.random.default_rng(SYNTHETIC_RANDOM_SEED)
    gen = DriveCycleGenerator(rng, "aggressive", "highway_trip")
    df = gen.generate()
    
    # SOC should not be random independent — it should trend
    soc_diff_std = df["battery_soc"].diff().dropna().std()
    # Standard deviation of SOC changes should be small (smooth evolution)
    assert soc_diff_std < 2.0, f"SOC evolution too random: std={soc_diff_std:.3f}"

def test_probabilistic_labeler_produces_ambiguity():
    rng = np.random.default_rng(42)
    labeler = ProbabilisticLabeler(rng)
    
    from src.rule_ems import VehicleState
    # Create an ambiguous state (moderate everything)
    state = VehicleState(
        speed=45.0, acceleration=0.5, power_required_kw=20.0,
        torque_required_nm=100.0, battery_soc=55.0, battery_temp=25.0,
        aux_load_kw=1.5, grade_angle=0.0, regen_available=False,
        braking=False, traffic_condition="medium", current_segment="arterial",
        next_segment="urban"
    )
    
    # Run 100 times — should NOT always produce the same label
    labels = set()
    for _ in range(100):
        label, _, _ = labeler.label(state)
        labels.add(label)
    
    assert len(labels) > 1, "Probabilistic labeler produced only one label for ambiguous state"

def test_physical_constraints_enforced():
    rng = np.random.default_rng(SYNTHETIC_RANDOM_SEED)
    gen = DriveCycleGenerator(rng, "erratic", "city_delivery")
    df = gen.generate()
    
    assert df["speed"].between(0, 140).all()
    assert df["battery_soc"].between(2, 100).all()
    assert df["battery_temp"].between(10, 60).all()
    assert df["power_required_kw"].between(-25, 120).all()
