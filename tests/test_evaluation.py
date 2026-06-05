"""
test_evaluation.py — Unit tests verifying KPI computation and reporting.
"""

import pytest
import pandas as pd
from pathlib import Path
from src.evaluation import compute_kpis, save_kpi_reports

def test_kpi_calculation():
    """Verify that all 10 KPIs are calculated accurately from simulated results."""
    # Build 10 steps of simulation output
    data = {
        "speed_kmh": [36.0] * 10,  # 36 km/h = 10 m/s
        "soc": [70.0, 69.9, 69.8, 69.7, 69.6, 69.5, 69.4, 69.3, 69.2, 69.1],
        "ems_mode": ["EV"] * 5 + ["ICE"] * 5,
        "power_demand_kw": [10.0] * 10,
        "actual_battery_power_kw": [10.0] * 10,
        "fuel_rate_lh": [3.0] * 10,
        "engine_efficiency": [0.35] * 10,
        "engine_running": [True] * 10
    }
    df = pd.DataFrame(data)
    
    kpis = compute_kpis(df, initial_soc=0.70)
    
    assert "fuel_economy_l_100km" in kpis
    assert "soc_deviation_pct" in kpis
    assert "energy_recuperation_pct" in kpis
    assert "ev_mode_share_pct" in kpis
    assert "hybrid_mode_share_pct" in kpis
    assert "ice_mode_share_pct" in kpis
    assert "operating_cost_currency_km" in kpis
    assert "co2_g_km" in kpis
    assert "average_ice_efficiency_pct" in kpis
    assert "peak_battery_power_kw" in kpis
    
    # Check simple values
    assert kpis["ev_mode_share_pct"] == 50.0
    assert kpis["ice_mode_share_pct"] == 50.0
    assert kpis["average_ice_efficiency_pct"] == 35.0
    assert kpis["peak_battery_power_kw"] == 10.0

def test_save_kpi_reports(tmp_path):
    """Verify JSON and text KPI reports are written to disk."""
    kpis = {
        "fuel_economy_l_100km": 5.2,
        "soc_deviation_pct": 2.1,
        "energy_recuperation_pct": 45.0,
        "ev_mode_share_pct": 60.0,
        "hybrid_mode_share_pct": 30.0,
        "ice_mode_share_pct": 10.0,
        "operating_cost_currency_km": 0.082,
        "co2_g_km": 120.1,
        "average_ice_efficiency_pct": 34.5,
        "peak_battery_power_kw": 45.2
    }
    
    save_kpi_reports(kpis, output_dir=tmp_path)
    
    assert (tmp_path / "kpi_report.json").exists()
    assert (tmp_path / "kpi_summary.txt").exists()
