"""
evaluation.py — PHEX HEV EMS KPI Evaluation Pipeline
Computes and reports all spec-mandated automotive KPIs from simulation runs.
Saves outputs to outputs/kpi_report.json and outputs/kpi_summary.txt.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import List, Dict, Any, Union
import pandas as pd
import numpy as np

# Energy cost assumptions
FUEL_PRICE_PER_L: float = 1.50         # currency/L petrol
ELECTRICITY_PRICE_PER_KWH: float = 0.25  # currency/kWh electricity
CO2_G_PER_L_PETROL: float = 2310.0      # 2.31 kg CO2/L = 2310 g CO2/L

def compute_kpis(results: Union[List[Dict[str, Any]], pd.DataFrame], initial_soc: float = 0.70) -> Dict[str, Any]:
    """
    Computes all 10 required KPIs from simulation outputs.

    Expected fields in results:
    - speed_kmh: float
    - soc: float (0.0 to 1.0 or 0 to 100)
    - ems_mode: str
    - power_demand_kw: float
    - actual_battery_power_kw: float
    - fuel_rate_lh: float (L/h)
    - engine_efficiency: float (0.0 to 1.0)
    - engine_running: bool
    """
    if isinstance(results, list):
        df = pd.DataFrame(results)
    else:
        df = results.copy()

    n_steps = len(df)
    if n_steps == 0:
        return {}

    # Distance calculation: speed_kmh * dt (assume 1 second step)
    dt_s = 1.0
    speeds_kph = df["speed_kmh"].values
    distance_km = float(np.sum(speeds_kph) * (dt_s / 3600.0))
    distance_km = max(distance_km, 0.001)  # prevent division by zero

    # Fuel Economy (L/100 km)
    # Total fuel consumed = sum(fuel_rate_lh * dt_s / 3600)
    if "fuel_rate_lh" in df.columns:
        total_fuel_l = float(np.sum(df["fuel_rate_lh"].values) * (dt_s / 3600.0))
    else:
        total_fuel_l = 0.0
    fuel_economy = (total_fuel_l / distance_km) * 100.0

    # SOC Deviation (% RMS)
    # RMS deviation of SOC from initial SOC
    soc_values = df["soc"].values
    if len(soc_values) > 0 and np.max(soc_values) <= 1.0:
        # Convert to percentage
        soc_values = soc_values * 100.0
    initial_soc_pct = initial_soc * 100.0 if initial_soc <= 1.0 else initial_soc
    soc_dev_rms = float(math.sqrt(np.mean((soc_values - initial_soc_pct) ** 2)))

    # Energy Recuperation (%)
    # Total regen energy recovered / total braking energy (positive fraction)
    # Braking energy = sum of negative power demands in kW * dt
    if "power_demand_kw" in df.columns:
        demands = df["power_demand_kw"].values
        braking_energy_kwh = float(np.sum(np.abs(demands[demands < 0.0])) * (dt_s / 3600.0))
    else:
        braking_energy_kwh = 0.0

    if "actual_battery_power_kw" in df.columns:
        batt_powers = df["actual_battery_power_kw"].values
        # Negative batt_powers is charge/regen
        regen_recovered_kwh = float(np.sum(np.abs(batt_powers[batt_powers < 0.0])) * (dt_s / 3600.0))
    elif "regen_power_kw" in df.columns:
        regen_powers = df["regen_power_kw"].values
        regen_recovered_kwh = float(np.sum(regen_powers) * (dt_s / 3600.0))
    else:
        regen_recovered_kwh = 0.0

    energy_recuperation = (regen_recovered_kwh / braking_energy_kwh * 100.0) if braking_energy_kwh > 0 else 0.0

    # Mode Shares
    modes = df["ems_mode"].values
    ev_share = float(np.sum([1 for m in modes if m in ("EV", "EV_ONLY", "BATTERY_ONLY")])) / n_steps * 100.0
    hybrid_share = float(np.sum([1 for m in modes if m in ("HYBRID_ASSIST", "HYBRID")])) / n_steps * 100.0
    ice_share = float(np.sum([1 for m in modes if m in ("ICE", "ICE_ONLY", "ENGINE_ONLY", "CHARGE_SUSTAIN")])) / n_steps * 100.0

    # Operating Cost Estimate (currency/km)
    # Cost = (fuel in L * petrol price + electricity used in kWh * price) / distance
    # Electricity used = initial SOC energy - final SOC energy
    from src.vehicle_config import BATTERY_CAPACITY
    soc_start = soc_values[0] / 100.0
    soc_end = soc_values[-1] / 100.0
    electricity_used_kwh = (soc_start - soc_end) * BATTERY_CAPACITY
    # Clip negative electricity used to 0 (since it was charged by regen/ICE)
    electricity_cost = max(0.0, electricity_used_kwh) * ELECTRICITY_PRICE_PER_KWH
    fuel_cost = total_fuel_l * FUEL_PRICE_PER_L
    operating_cost = (fuel_cost + electricity_cost) / distance_km

    # CO2 Estimate (g/km)
    co2_estimate = (total_fuel_l * CO2_G_PER_L_PETROL) / distance_km

    # Average ICE Efficiency (%)
    if "engine_efficiency" in df.columns:
        effs = df["engine_efficiency"].values
        running = df["engine_running"].values if "engine_running" in df.columns else (effs > 0.0)
        active_effs = effs[running]
        avg_ice_eff = float(np.mean(active_effs) * 100.0) if len(active_effs) > 0 else 0.0
    else:
        avg_ice_eff = 0.0

    # Peak Battery Power (kW)
    # Maximum instantaneous battery charge/discharge rate
    if "actual_battery_power_kw" in df.columns:
        peak_batt_power = float(np.max(np.abs(df["actual_battery_power_kw"].values)))
    elif "power_demand_kw" in df.columns:
        # Fallback estimation
        peak_batt_power = float(np.max(np.abs(df["power_demand_kw"].values)))
    else:
        peak_batt_power = 0.0

    return {
        "fuel_economy_l_100km": round(fuel_economy, 2),
        "soc_deviation_pct": round(soc_dev_rms, 2),
        "energy_recuperation_pct": round(energy_recuperation, 1),
        "ev_mode_share_pct": round(ev_share, 1),
        "hybrid_mode_share_pct": round(hybrid_share, 1),
        "ice_mode_share_pct": round(ice_share, 1),
        "operating_cost_currency_km": round(operating_cost, 4),
        "co2_g_km": round(co2_estimate, 1),
        "average_ice_efficiency_pct": round(avg_ice_eff, 1),
        "peak_battery_power_kw": round(peak_batt_power, 2)
    }

def save_kpi_reports(kpis: Dict[str, Any], output_dir: Union[str, Path] = "outputs") -> None:
    """Saves computed KPIs to JSON and text summary files."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Save JSON report
    json_path = out_path / "kpi_report.json"
    with open(json_path, "w") as f:
        json.dump(kpis, f, indent=4)
        
    # Save human-readable summary
    summary_path = out_path / "kpi_summary.txt"
    with open(summary_path, "w") as f:
        f.write("PHEX HEV EMS Simulation - KPI Summary Report\n")
        f.write("=============================================\n\n")
        f.write(f"Fuel Economy             : {kpis.get('fuel_economy_l_100km', 0.0):.2f} L/100 km\n")
        f.write(f"SOC Deviation (RMS)      : {kpis.get('soc_deviation_pct', 0.0):.2f} %\n")
        f.write(f"Energy Recuperation      : {kpis.get('energy_recuperation_pct', 0.0):.1f} %\n")
        f.write(f"EV Mode Share            : {kpis.get('ev_mode_share_pct', 0.0):.1f} %\n")
        f.write(f"Hybrid Mode Share        : {kpis.get('hybrid_mode_share_pct', 0.0):.1f} %\n")
        f.write(f"ICE Mode Share           : {kpis.get('ice_mode_share_pct', 0.0):.1f} %\n")
        f.write(f"Operating Cost Estimate  : {kpis.get('operating_cost_currency_km', 0.0):.4f} currency/km\n")
        f.write(f"CO2 Estimate             : {kpis.get('co2_g_km', 0.0):.1f} g/km\n")
        f.write(f"Average ICE Efficiency   : {kpis.get('average_ice_efficiency_pct', 0.0):.1f} %\n")
        f.write(f"Peak Battery Power       : {kpis.get('peak_battery_power_kw', 0.0):.2f} kW\n")
