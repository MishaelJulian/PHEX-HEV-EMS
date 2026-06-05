"""
predictive_controller.py — Phase 7C: Predictive Route-Aware EMS Policy
Author: Antigravity
Date: 2026-05-20

Genuine AI supervisory intelligence. Uses route lookahead, SOC trajectory,
and thermal state to produce advisory signals that influence ML decisions.
This is NOT a decision-maker — it's an advisor that augments the ML model.
"""

import logging
from dataclasses import dataclass
from typing import Optional

from src.config import (
    SOC_CRITICAL, SOC_LOW, SOC_HIGH, HIGH_POWER_KW,
    BATTERY_TEMP_MAX, BATTERY_TEMP_MIN,
    BANGALORE_CONGESTION_THRESHOLD_KMH, BANGALORE_HEAVY_GRIDLOCK_MULTIPLIER
)
from src.rule_ems import VehicleState

logger = logging.getLogger(__name__)


@dataclass
class PredictiveAdvisory:
    """Advisory signals produced by the predictive controller."""
    preserve_battery: bool      # urban/stop-go ahead — save EV energy
    prepare_regen: bool         # downhill/braking opportunity ahead
    prefer_engine: bool         # long highway cruise ahead — engine efficient
    charge_sustain: bool        # low SOC + long distance remaining
    reduce_ev_load: bool        # battery thermal stress — limit EV usage
    anticipate_stop_go: bool    # stop-and-go traffic ahead — reserve EV

    # Numeric advisory scores (0.0 to 1.0)
    battery_urgency: float      # how urgently we need to preserve/charge battery
    regen_opportunity: float    # how likely regen is in near future
    ev_suitability: float       # how suitable current conditions are for EV mode

    def to_feature_dict(self) -> dict:
        """Convert advisory signals to features for ML model input."""
        return {
            "adv_preserve_battery": int(self.preserve_battery),
            "adv_prepare_regen": int(self.prepare_regen),
            "adv_prefer_engine": int(self.prefer_engine),
            "adv_charge_sustain": int(self.charge_sustain),
            "adv_reduce_ev_load": int(self.reduce_ev_load),
            "adv_anticipate_stop_go": int(self.anticipate_stop_go),
            "adv_battery_urgency": round(self.battery_urgency, 3),
            "adv_regen_opportunity": round(self.regen_opportunity, 3),
            "adv_ev_suitability": round(self.ev_suitability, 3),
        }


class PredictiveEMSController:
    """
    Route-aware predictive advisor for the EMS decision engine.

    Uses multi-step route lookahead, SOC trajectory, and thermal state
    to produce advisory signals. These signals are appended as features
    to the ML model input, allowing the model to learn predictive behavior.
    """

    def advise(self, state: VehicleState, lookahead: list,
               distance_remaining: float, grade_ahead: float) -> PredictiveAdvisory:
        """
        Produce advisory signals based on current state and route lookahead.

        Args:
            state: Current VehicleState
            lookahead: List of upcoming segment names (e.g., ["highway", "arterial", "urban"])
            distance_remaining: km remaining on route
            grade_ahead: average grade angle of upcoming segments (degrees)
        """
        # --- Preserve Battery ---
        # If urban or stop-go segments are coming, preserve battery for EV use
        ev_segments_ahead = sum(1 for seg in lookahead if seg in ["urban", "stop_go", "suburban"])
        preserve_battery = ev_segments_ahead >= 2 and state.battery_soc < SOC_HIGH

        # Bangalore traffic adaptation: if currently in slow heavy traffic, preserve battery aggressively
        is_bangalore_gridlock = (state.traffic_condition == "heavy" and state.speed < BANGALORE_CONGESTION_THRESHOLD_KMH)
        if is_bangalore_gridlock:
            preserve_battery = True

        # --- Prepare Regen ---
        # If downhill grade or braking segments ahead
        prepare_regen = (grade_ahead < -2.0 or
                         any(seg in ["stop_go"] for seg in lookahead[:2]))

        # --- Prefer Engine ---
        # Long highway ahead — engine runs at peak efficiency
        highway_ahead = sum(1 for seg in lookahead if seg == "highway")
        prefer_engine = highway_ahead >= 2 and state.battery_soc < SOC_HIGH

        # --- Charge Sustain ---
        # Low SOC with significant distance remaining
        charge_sustain = (state.battery_soc < SOC_LOW and
                          distance_remaining > 5.0)

        # --- Reduce EV Load ---
        # Battery temperature approaching limits
        temp_margin = min(
            BATTERY_TEMP_MAX - state.battery_temp,
            state.battery_temp - BATTERY_TEMP_MIN
        )
        reduce_ev_load = temp_margin < 5.0

        # --- Anticipate Stop-Go ---
        # Heavy traffic or stop-go segment imminent
        anticipate_stop_go = (len(lookahead) > 0 and
                              lookahead[0] in ["stop_go", "urban"]) or is_bangalore_gridlock

        # --- Numeric Scores ---

        # Battery urgency: 0 = plenty of charge, 1 = critical
        if state.battery_soc > SOC_HIGH:
            battery_urgency = 0.0
        elif state.battery_soc > SOC_LOW:
            battery_urgency = 1.0 - (state.battery_soc - SOC_LOW) / (SOC_HIGH - SOC_LOW)
        else:
            battery_urgency = min(1.0, (SOC_LOW - state.battery_soc) / SOC_LOW + 0.5)

        # Scale battery preservation urgency under Bangalore heavy gridlock
        if is_bangalore_gridlock and state.battery_soc >= SOC_LOW:
            battery_urgency = min(1.0, battery_urgency * BANGALORE_HEAVY_GRIDLOCK_MULTIPLIER)

        # Regen opportunity: based on lookahead and grade
        regen_opportunity = min(1.0, max(0.0,
            (ev_segments_ahead * 0.15) +
            (0.3 if grade_ahead < -2.0 else 0.0) +
            (0.2 if prepare_regen else 0.0)
        ))

        # EV suitability: current conditions favor EV?
        ev_score = 0.0
        if state.current_segment in ["urban", "stop_go", "suburban"]:
            ev_score += 0.3
        if state.speed < 50:
            ev_score += 0.2
        if state.power_required_kw < HIGH_POWER_KW * 0.7:
            ev_score += 0.2
        if state.battery_soc > SOC_LOW:
            ev_score += 0.2
        if state.battery_temp < BATTERY_TEMP_MAX - 5:
            ev_score += 0.1
            
        # Heavy gridlock thermal penalty: low cooling airflow at low speed increases battery temperature risk
        if is_bangalore_gridlock and state.battery_temp > BATTERY_TEMP_MAX - 8:
            ev_score -= 0.2
            
        ev_suitability = min(1.0, max(0.0, ev_score))

        return PredictiveAdvisory(
            preserve_battery=preserve_battery,
            prepare_regen=prepare_regen,
            prefer_engine=prefer_engine,
            charge_sustain=charge_sustain,
            reduce_ev_load=reduce_ev_load,
            anticipate_stop_go=anticipate_stop_go,
            battery_urgency=round(battery_urgency, 3),
            regen_opportunity=round(regen_opportunity, 3),
            ev_suitability=round(ev_suitability, 3),
        )
