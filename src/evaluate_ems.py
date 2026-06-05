"""
evaluate_ems.py — Phase 7: Enhanced EMS Evaluation Suite
Author: Antigravity
Date: 2026-05-20

Runs structured edge-case and predictive scenarios against both
RuleBasedEMS and HybridDecisionEngine to validate AI behavior.
"""

import logging
from typing import Dict, Any

from src.rule_ems import RuleBasedEMS, VehicleState
from src.decision_engine import HybridDecisionEngine
from src.config import SEPARATOR

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

def _get_base_state() -> Dict[str, Any]:
    return {
        "speed": 50.0,
        "acceleration": 0.0,
        "power_required_kw": 20.0,
        "torque_required_nm": 100.0,
        "battery_soc": 50.0,
        "battery_temp": 25.0,
        "aux_load_kw": 1.0,
        "grade_angle": 0.0,
        "regen_available": False,
        "braking": False,
        "traffic_condition": "medium",
        "current_segment": "urban",
        "next_segment": "arterial"
    }

def build_scenarios() -> Dict[str, dict]:
    """Build test scenarios with state + route context."""
    scenarios = {}

    # === SAFETY SCENARIOS (must trigger overrides) ===

    s1 = _get_base_state()
    s1["battery_soc"] = 12.0
    s1["speed"] = 80.0
    s1["current_segment"] = "highway"
    scenarios["CRITICAL_SOC_PROTECTION"] = {
        "state": VehicleState(**s1),
        "lookahead": ["highway", "arterial", "urban"],
        "distance": 15.0, "grade": 0.0,
    }

    s2 = _get_base_state()
    s2["battery_temp"] = 45.0
    s2["speed"] = 30.0
    scenarios["THERMAL_OVERRIDE"] = {
        "state": VehicleState(**s2),
        "lookahead": ["urban", "stop_go", "urban"],
        "distance": 5.0, "grade": 0.0,
    }

    # === PREDICTIVE SCENARIOS ===

    s3 = _get_base_state()
    s3["speed"] = 45.0
    s3["battery_soc"] = 60.0
    s3["current_segment"] = "arterial"
    s3["next_segment"] = "highway"
    scenarios["HIGHWAY_APPROACHING_PRESERVE"] = {
        "state": VehicleState(**s3),
        "lookahead": ["highway", "highway", "highway"],
        "distance": 20.0, "grade": 0.0,
    }

    s4 = _get_base_state()
    s4["speed"] = 80.0
    s4["battery_soc"] = 75.0
    s4["current_segment"] = "highway"
    s4["next_segment"] = "urban"
    scenarios["URBAN_AHEAD_SAVE_BATTERY"] = {
        "state": VehicleState(**s4),
        "lookahead": ["urban", "stop_go", "urban"],
        "distance": 8.0, "grade": 0.0,
    }

    s5 = _get_base_state()
    s5["speed"] = 60.0
    s5["battery_soc"] = 55.0
    s5["current_segment"] = "mountain"
    s5["next_segment"] = "mountain"
    scenarios["DOWNHILL_REGEN_OPPORTUNITY"] = {
        "state": VehicleState(**s5),
        "lookahead": ["mountain", "mountain", "arterial"],
        "distance": 12.0, "grade": -6.0,
    }

    # === AMBIGUOUS SCENARIOS ===

    s6 = _get_base_state()
    s6["speed"] = 40.0
    s6["power_required_kw"] = 25.0
    s6["battery_soc"] = 55.0
    s6["current_segment"] = "arterial"
    scenarios["MODERATE_ARTERIAL_AMBIGUOUS"] = {
        "state": VehicleState(**s6),
        "lookahead": ["arterial", "urban", "stop_go"],
        "distance": 10.0, "grade": 1.0,
    }

    s7 = _get_base_state()
    s7["speed"] = 25.0
    s7["power_required_kw"] = 10.0
    s7["battery_soc"] = 85.0
    s7["traffic_condition"] = "heavy"
    s7["current_segment"] = "stop_go"
    scenarios["PERFECT_EV_CONDITIONS"] = {
        "state": VehicleState(**s7),
        "lookahead": ["urban", "stop_go", "urban"],
        "distance": 5.0, "grade": 0.0,
    }

    s8 = _get_base_state()
    s8["speed"] = 110.0
    s8["power_required_kw"] = 55.0
    s8["battery_soc"] = 40.0
    s8["current_segment"] = "highway"
    s8["next_segment"] = "highway"
    scenarios["HIGHWAY_HIGH_DEMAND"] = {
        "state": VehicleState(**s8),
        "lookahead": ["highway", "highway", "arterial"],
        "distance": 25.0, "grade": 0.5,
    }

    # === REGEN SCENARIO ===

    s9 = _get_base_state()
    s9["speed"] = 60.0
    s9["braking"] = True
    s9["regen_available"] = True
    s9["power_required_kw"] = -15.0
    s9["acceleration"] = -3.0
    scenarios["REGEN_BRAKING"] = {
        "state": VehicleState(**s9),
        "lookahead": ["urban", "stop_go", "urban"],
        "distance": 5.0, "grade": -2.0,
    }

    return scenarios


def run_evaluation() -> None:
    logger.info(SEPARATOR)
    logger.info(" EMS CONTROLLER SCENARIO EVALUATION SUITE (Phase 7)")
    logger.info(SEPARATOR)

    rule_engine = RuleBasedEMS()

    try:
        hybrid_engine = HybridDecisionEngine()
        hybrid_available = True
    except FileNotFoundError:
        logger.warning("HybridDecisionEngine disabled (ML models not found).")
        hybrid_available = False

    scenarios = build_scenarios()

    for name, ctx in scenarios.items():
        state = ctx["state"]
        lookahead = ctx["lookahead"]
        dist = ctx["distance"]
        grade = ctx["grade"]

        logger.info(f"\n[ SCENARIO: {name} ]")
        logger.info(f"  Speed={state.speed}km/h SOC={state.battery_soc}% "
                     f"Temp={state.battery_temp}C Power={state.power_required_kw}kW "
                     f"Seg={state.current_segment} Next={state.next_segment}")
        logger.info(f"  Lookahead={lookahead} Dist={dist}km Grade={grade}deg")

        # Rule Based
        rule_dec = rule_engine.decide(state)
        logger.info(f"  [Rule]   {rule_dec.mode:<15} | {rule_dec.rule_triggered}")

        # Hybrid AI
        if hybrid_available:
            hybrid_dec = hybrid_engine.decide(
                state, lookahead=lookahead,
                distance_remaining=dist, grade_ahead=grade
            )
            tag = "OVERRIDE" if hybrid_dec.source == "HYBRID_OVERRIDE" else "ML"
            logger.info(f"  [Hybrid] {hybrid_dec.mode:<15} | {tag:<8} | {hybrid_dec.reason}")

    logger.info("\n" + SEPARATOR)
    logger.info(" EVALUATION COMPLETE")
    logger.info(SEPARATOR)


if __name__ == "__main__":
    run_evaluation()


# ══════════════════════════════════════════════════════════════════
# Phase 3: Extended KPI Computation and Report Generation
# ══════════════════════════════════════════════════════════════════

import math
from typing import List
import pandas as pd


def compute_phase3_kpis(
    simulation_df: pd.DataFrame,
    target_soc: float = 0.70,
) -> Dict[str, Any]:
    """Compute Phase 3 KPIs from a simulation results DataFrame.

    Expected columns in simulation_df:
        - speed_kmh: Vehicle speed per timestep (km/h).
        - soc: State of charge per timestep (0.0–1.0).
        - ems_mode: EMS mode string per timestep.
        - regen_power_kw: Regen power recovered per timestep (kW).

    Optional columns (if available):
        - fuel_rate_lh: Fuel consumption rate per timestep (L/h).
        - engine_running: Boolean, True if engine was on.

    Args:
        simulation_df: DataFrame with per-step simulation results.
        target_soc: SOC target for RMS error calculation.

    Returns:
        Dict of KPI name → value.
    """
    n_steps = len(simulation_df)
    if n_steps == 0:
        logger.warning("Empty simulation DataFrame — returning zero KPIs.")
        return {k: 0.0 for k in [
            "fuel_consumption_l_per_100km", "soc_error_rms",
            "regen_recovery_kwh", "regen_efficiency_pct",
            "mode_transition_count", "battery_usage_kwh",
            "engine_runtime_s", "ev_mode_fraction", "hybrid_mode_fraction",
        ]}

    # ── Distance (km) ──────────────────────────────────────────────
    # Approximate: speed_kmh * timestep (assume 1s per step)
    timestep_s = 1.0
    speeds = simulation_df["speed_kmh"].values
    distance_km = sum(speeds) * timestep_s / 3600.0
    distance_km = max(distance_km, 0.001)  # prevent division by zero

    # ── Fuel Consumption ───────────────────────────────────────────
    if "fuel_rate_lh" in simulation_df.columns:
        total_fuel_l = sum(simulation_df["fuel_rate_lh"].values) * timestep_s / 3600.0
    else:
        # Estimate: assume 3 L/h average when engine is running
        engine_steps = sum(
            1 for m in simulation_df["ems_mode"].values
            if m in ("ICE", "HYBRID_ASSIST", "CHARGE_SUSTAIN", "CHARGE_DEPLETING",
                      "ICE_ONLY", "HYBRID", "CHARGE_SUSTAIN", "MAX_ACCEL")
        )
        total_fuel_l = engine_steps * 3.0 * timestep_s / 3600.0
    fuel_consumption_l_per_100km = (total_fuel_l / distance_km) * 100.0

    # ── SOC Error RMS ──────────────────────────────────────────────
    soc_values = simulation_df["soc"].values
    soc_errors = [(s - target_soc) ** 2 for s in soc_values]
    soc_error_rms = math.sqrt(sum(soc_errors) / n_steps) * 100.0  # as percentage

    # ── Regen Recovery ─────────────────────────────────────────────
    regen_powers = simulation_df.get("regen_power_kw", pd.Series([0.0] * n_steps)).values
    regen_recovery_kwh = sum(regen_powers) * timestep_s / 3600.0

    # ── Regen Efficiency ───────────────────────────────────────────
    # Theoretical braking energy: sum of negative power demand
    if "power_demand_kw" in simulation_df.columns:
        braking_energy = sum(
            abs(p) for p in simulation_df["power_demand_kw"].values if p < 0
        ) * timestep_s / 3600.0
    else:
        braking_energy = regen_recovery_kwh * 1.5  # Fallback estimate
    regen_efficiency_pct = (
        (regen_recovery_kwh / braking_energy * 100.0) if braking_energy > 0 else 0.0
    )

    # ── Mode Transition Count ──────────────────────────────────────
    modes = simulation_df["ems_mode"].values
    mode_transition_count = sum(
        1 for i in range(1, len(modes)) if modes[i] != modes[i - 1]
    )

    # ── Battery Usage ──────────────────────────────────────────────
    soc_start = soc_values[0]
    soc_end = soc_values[-1]
    battery_capacity_kwh = 13.8
    battery_usage_kwh = (soc_start - soc_end) * battery_capacity_kwh - regen_recovery_kwh

    # ── Engine Runtime ─────────────────────────────────────────────
    if "engine_running" in simulation_df.columns:
        engine_runtime_s = float(sum(simulation_df["engine_running"].values)) * timestep_s
    else:
        engine_modes = {"ICE", "HYBRID_ASSIST", "CHARGE_SUSTAIN", "CHARGE_DEPLETING",
                        "ICE_ONLY", "HYBRID", "MAX_ACCEL"}
        engine_runtime_s = float(sum(
            1 for m in modes if m in engine_modes
        )) * timestep_s

    # ── Mode Fractions ─────────────────────────────────────────────
    ev_modes = {"EV", "EV_ONLY"}
    hybrid_modes = {"HYBRID_ASSIST", "HYBRID"}
    ev_mode_fraction = sum(1 for m in modes if m in ev_modes) / n_steps * 100.0
    hybrid_mode_fraction = sum(1 for m in modes if m in hybrid_modes) / n_steps * 100.0

    kpis = {
        "fuel_consumption_l_per_100km": round(fuel_consumption_l_per_100km, 2),
        "soc_error_rms": round(soc_error_rms, 2),
        "regen_recovery_kwh": round(regen_recovery_kwh, 4),
        "regen_efficiency_pct": round(regen_efficiency_pct, 1),
        "mode_transition_count": mode_transition_count,
        "battery_usage_kwh": round(battery_usage_kwh, 4),
        "engine_runtime_s": round(engine_runtime_s, 1),
        "ev_mode_fraction": round(ev_mode_fraction, 1),
        "hybrid_mode_fraction": round(hybrid_mode_fraction, 1),
    }

    logger.info("Phase 3 KPIs computed: %s", kpis)
    return kpis


def generate_phase3_report(
    kpis: Dict[str, Any],
    drive_cycle: str = "UNKNOWN",
    simulation_time_s: float = 0.0,
    distance_km: float = 0.0,
    final_soc: float = 0.0,
    target_soc: float = 0.70,
) -> str:
    """Generate a formatted text report from Phase 3 KPIs.

    Args:
        kpis: Dict of KPI values from compute_phase3_kpis().
        drive_cycle: Name of the drive cycle.
        simulation_time_s: Total simulation duration in seconds.
        distance_km: Total distance driven in km.
        final_soc: Final SOC value.
        target_soc: Target SOC for reporting.

    Returns:
        Formatted multi-line report string.
    """
    engine_pct = (
        (kpis["engine_runtime_s"] / simulation_time_s * 100.0)
        if simulation_time_s > 0 else 0.0
    )

    # Compute remaining mode fractions
    ev_frac = kpis["ev_mode_fraction"]
    hybrid_frac = kpis["hybrid_mode_fraction"]
    ice_frac = max(0.0, 100.0 - ev_frac - hybrid_frac - kpis.get("regen_frac", 0.0))

    report = f"""
PHEX-HEV-EMS EVALUATION REPORT
================================
Drive Cycle     : {drive_cycle}
Simulation Time : {simulation_time_s:.0f} s
Distance        : {distance_km:.1f} km

FUEL ECONOMY
  Fuel Consumption   : {kpis['fuel_consumption_l_per_100km']:.1f} L/100km
  Engine Runtime     : {kpis['engine_runtime_s']:.0f} s ({engine_pct:.1f}% of total)

SOC MANAGEMENT
  Final SOC          : {final_soc:.2f}
  SOC Target         : {target_soc:.2f}
  SOC Error (RMS)    : {kpis['soc_error_rms']:.1f}%

ENERGY RECOVERY
  Regen Recovered    : {kpis['regen_recovery_kwh']:.4f} kWh
  Regen Efficiency   : {kpis['regen_efficiency_pct']:.1f}%

MODE DISTRIBUTION
  EV Only            : {ev_frac:.1f}%
  Hybrid             : {hybrid_frac:.1f}%
  ICE Only           : {ice_frac:.1f}%
  Mode Transitions   : {kpis['mode_transition_count']}

BATTERY
  Net Battery Usage  : {kpis['battery_usage_kwh']:.4f} kWh
"""
    return report.strip()
