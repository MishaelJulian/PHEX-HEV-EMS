"""
soc_planner.py — Phase 2: SOC Target Planner
Author: Antigravity
Date: 2026-06-01

Rule-based SOC planner that outputs a target SOC based on upcoming route
context and traffic state.  Designed so the rule implementation can be
replaced by an ML model (e.g. RL policy) without changing callers.
"""

import logging
from dataclasses import dataclass
from typing import Optional, List

from src.traffic_predictor import TrafficState
from src.config import (
    SOC_URBAN_TARGET,
    SOC_HIGHWAY_TARGET,
    SOC_DEFAULT_TARGET,
)

logger = logging.getLogger(__name__)


@dataclass
class RouteContext:
    """Parsed route lookahead information.

    Attributes:
        upcoming_zone: One of "urban", "highway", "mixed", "unknown".
        distance_to_zone_km: How far until zone changes (km).
        total_remaining_km: Total distance remaining on the route (km).
    """

    upcoming_zone: str
    distance_to_zone_km: float
    total_remaining_km: float


@dataclass
class SOCPlan:
    """Output of the SOC planner.

    Attributes:
        target_soc: Recommended SOC target as a fraction [0.0, 1.0].
        rationale: Human-readable explanation for logging / debug.
    """

    target_soc: float
    rationale: str


class SOCPlanner:
    """Rule-based SOC planner.

    Determines optimal target SOC given route context and current traffic
    state.  The class interface is designed for ML replacement: a future
    subclass need only override ``plan()`` with a model-based
    implementation.
    """

    def plan(
        self,
        traffic_state: TrafficState,
        route_context: RouteContext,
        current_soc: float,
        predicted_speed_profile: Optional[List[float]] = None,
    ) -> SOCPlan:
        """Produces a SOC plan.

        Rules (all targets from config.py):
        1. If upcoming_zone == "urban" AND distance_to_zone_km < 10.0:
           → target = SOC_URBAN_TARGET (preserve battery for urban EV use)
        2. If upcoming_zone == "highway" AND current_soc > SOC_HIGHWAY_TARGET:
           → target = SOC_HIGHWAY_TARGET (allow highway depletion)
        3. If traffic_state == STOP_GO:
           → target = SOC_URBAN_TARGET
        4. Default:
           → target = SOC_DEFAULT_TARGET

        Args:
            traffic_state: Current classified traffic state.
            route_context: Parsed route lookahead.
            current_soc: Current battery SOC as fraction.
            predicted_speed_profile: Optional list of future speed samples (km/h).

        Returns:
            SOCPlan with target and human-readable rationale.
        """
        # Rule 1 — upcoming urban zone within 10 km
        if (
            route_context.upcoming_zone == "urban"
            and route_context.distance_to_zone_km < 10.0
        ):
            logger.debug(
                "SOCPlanner rule 1: urban zone in %.1f km → target=%.2f",
                route_context.distance_to_zone_km,
                SOC_URBAN_TARGET,
            )
            return SOCPlan(
                target_soc=SOC_URBAN_TARGET,
                rationale=(
                    f"Urban zone approaching in {route_context.distance_to_zone_km:.1f} km. "
                    f"Preserving battery at {SOC_URBAN_TARGET:.0%} for urban EV driving."
                ),
            )

        # Rule 2 — highway zone with SOC above highway target
        if (
            route_context.upcoming_zone == "highway"
            and current_soc > SOC_HIGHWAY_TARGET
        ):
            logger.debug(
                "SOCPlanner rule 2: highway zone, SOC=%.2f > target=%.2f",
                current_soc,
                SOC_HIGHWAY_TARGET,
            )
            return SOCPlan(
                target_soc=SOC_HIGHWAY_TARGET,
                rationale=(
                    f"Highway zone ahead. SOC {current_soc:.0%} above highway target "
                    f"{SOC_HIGHWAY_TARGET:.0%}. Allowing charge depletion on highway."
                ),
            )

        # Rule 3 — stop-and-go traffic
        if traffic_state == TrafficState.STOP_GO:
            logger.debug(
                "SOCPlanner rule 3: STOP_GO traffic → target=%.2f",
                SOC_URBAN_TARGET,
            )
            return SOCPlan(
                target_soc=SOC_URBAN_TARGET,
                rationale=(
                    f"Stop-and-go traffic detected. Preserving battery at "
                    f"{SOC_URBAN_TARGET:.0%} for efficient EV crawling."
                ),
            )

        # Rule 4 — default
        logger.debug("SOCPlanner rule 4: default → target=%.2f", SOC_DEFAULT_TARGET)
        return SOCPlan(
            target_soc=SOC_DEFAULT_TARGET,
            rationale=(
                f"No specific route/traffic condition. "
                f"Using default SOC target of {SOC_DEFAULT_TARGET:.0%}."
            ),
        )
