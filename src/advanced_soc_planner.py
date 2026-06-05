"""
advanced_soc_planner.py — Phase 3: Strategic Predictive SOC target allocation planner.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from src.traffic_predictor import TrafficPrediction
from src.demand_forecaster import DemandForecast

logger = logging.getLogger(__name__)

@dataclass
class RouteInfo:
    """Contains nav route metrics for SOC planning."""
    position_km: float
    total_distance_km: float
    next_segment_type: str                  # 'URBAN', 'MIXED', 'HIGHWAY'
    distance_to_urban_zone_km: float
    upcoming_urban_duration_s: float

@dataclass
class SOCPlan:
    """Strategic SOC target allocation recommendation output."""
    target_soc_at_urban_entry: float
    recommended_mode: str
    charge_rate_kw: float                  # recommended charging rate, 0 if not charging
    discharge_rate_kw: float               # recommended discharge rate
    confidence: float                      # 0.0 to 1.0
    reasoning: str                         # human-readable rationale

class AdvancedSOCPlanner:
    """
    Strategic SOC planner utilizing traffic forecasts, route previews,
    and power demand predictions to allocate battery state-of-charge.
    """

    def plan(
        self,
        current_soc: float,
        route_info: RouteInfo,
        traffic_prediction: TrafficPrediction,
        demand_forecast: DemandForecast,
    ) -> SOCPlan:
        """
        Returns a strategic SOC allocation plan.
        """
        # 1. Low SOC / Recovery (C.RULE8)
        if current_soc <= 0.25:
            logger.debug("AdvancedSOCPlanner: low SOC recovery triggered.")
            return SOCPlan(
                target_soc_at_urban_entry=0.35,
                recommended_mode="CHARGE_SUSTAIN",
                charge_rate_kw=12.0,
                discharge_rate_kw=0.0,
                confidence=0.95,
                reasoning="Low SOC recovery (C.RULE8): diverting ICE power to charge battery.",
            )

        # 2. SOC Preservation before Urban Zone (C.RULE2)
        if (
            route_info.distance_to_urban_zone_km <= 10.0
            and route_info.next_segment_type in ("HIGHWAY", "MIXED")
            and current_soc < 0.80
        ):
            logger.debug("AdvancedSOCPlanner: SOC preservation triggered.")
            return SOCPlan(
                target_soc_at_urban_entry=0.80,
                recommended_mode="CHARGE_SUSTAIN",
                charge_rate_kw=15.0,
                discharge_rate_kw=0.0,
                confidence=0.90,
                reasoning="SOC preservation before urban zone (C.RULE2): charging battery to target 80% SOC.",
            )

        # 3. Urban Stop-and-Go EV Priority (C.RULE1 & C.RULE4)
        if (
            route_info.next_segment_type == "URBAN"
            and traffic_prediction.traffic_density in ("HIGH", "CONGESTED")
            and current_soc > 0.25
        ):
            logger.debug("AdvancedSOCPlanner: urban stop-and-go priority triggered.")
            return SOCPlan(
                target_soc_at_urban_entry=0.20,
                recommended_mode="EV",
                charge_rate_kw=0.0,
                discharge_rate_kw=max(5.0, demand_forecast.mean_power_kw),
                confidence=0.92,
                reasoning="Urban stop-and-go EV priority (C.RULE1): utilizing battery for silent urban crawl.",
            )

        # 4. Traffic-Aware Congestion Bias (C.RULE3)
        if traffic_prediction.congestion_probability > 0.6 and current_soc > 0.25:
            logger.debug("AdvancedSOCPlanner: traffic-aware congestion bias triggered.")
            return SOCPlan(
                target_soc_at_urban_entry=0.20,
                recommended_mode="EV",
                charge_rate_kw=0.0,
                discharge_rate_kw=max(5.0, demand_forecast.mean_power_kw),
                confidence=0.85,
                reasoning="Traffic-aware planning (C.RULE3): biasing towards EV mode due to predicted congestion.",
            )

        # 5. Highway segment (C.RULE4)
        if route_info.next_segment_type == "HIGHWAY":
            logger.debug("AdvancedSOCPlanner: highway cruise behavior.")
            return SOCPlan(
                target_soc_at_urban_entry=0.40,
                recommended_mode="ICE",
                charge_rate_kw=0.0,
                discharge_rate_kw=0.0,
                confidence=0.80,
                reasoning="GPS route preview (C.RULE4): highway cruising segment, prioritizing ICE engine efficiency.",
            )

        # 6. Default / Fallback
        logger.debug("AdvancedSOCPlanner: default mixed planning mode.")
        return SOCPlan(
            target_soc_at_urban_entry=0.55,
            recommended_mode="HYBRID",
            charge_rate_kw=0.0,
            discharge_rate_kw=0.0,
            confidence=0.70,
            reasoning="GPS route preview (C.RULE4): mixed driving segment, balanced hybrid mode.",
        )
