"""
test_soc_planner.py — Phase 2: Tests for soc_planner module.
"""

import pytest

from src.soc_planner import SOCPlanner, SOCPlan, RouteContext
from src.traffic_predictor import TrafficState
from src.config import (
    SOC_URBAN_TARGET,
    SOC_HIGHWAY_TARGET,
    SOC_DEFAULT_TARGET,
)


@pytest.fixture
def planner() -> SOCPlanner:
    """Shared SOCPlanner instance."""
    return SOCPlanner()


class TestSOCPlannerPlan:
    """Tests for SOCPlanner.plan()."""

    def test_urban_zone_within_10km(self, planner: SOCPlanner) -> None:
        """Urban zone within 10 km → target == SOC_URBAN_TARGET."""
        ctx = RouteContext(upcoming_zone="urban", distance_to_zone_km=5.0, total_remaining_km=20.0)
        plan = planner.plan(TrafficState.FREE_FLOW, ctx, current_soc=0.60)
        assert plan.target_soc == SOC_URBAN_TARGET

    def test_highway_zone_high_soc(self, planner: SOCPlanner) -> None:
        """Highway zone, current SOC > highway target → target == SOC_HIGHWAY_TARGET."""
        ctx = RouteContext(upcoming_zone="highway", distance_to_zone_km=15.0, total_remaining_km=30.0)
        plan = planner.plan(TrafficState.FREE_FLOW, ctx, current_soc=0.70)
        assert plan.target_soc == SOC_HIGHWAY_TARGET

    def test_stop_go_traffic(self, planner: SOCPlanner) -> None:
        """STOP_GO traffic → target == SOC_URBAN_TARGET."""
        ctx = RouteContext(upcoming_zone="unknown", distance_to_zone_km=20.0, total_remaining_km=50.0)
        plan = planner.plan(TrafficState.STOP_GO, ctx, current_soc=0.50)
        assert plan.target_soc == SOC_URBAN_TARGET

    def test_unknown_zone_free_flow_default(self, planner: SOCPlanner) -> None:
        """Unknown zone, FREE_FLOW → target == SOC_DEFAULT_TARGET."""
        ctx = RouteContext(upcoming_zone="unknown", distance_to_zone_km=20.0, total_remaining_km=50.0)
        plan = planner.plan(TrafficState.FREE_FLOW, ctx, current_soc=0.50)
        assert plan.target_soc == SOC_DEFAULT_TARGET

    def test_returns_soc_plan_with_rationale(self, planner: SOCPlanner) -> None:
        """plan() returns SOCPlan with non-empty rationale string."""
        ctx = RouteContext(upcoming_zone="urban", distance_to_zone_km=5.0, total_remaining_km=20.0)
        plan = planner.plan(TrafficState.FREE_FLOW, ctx, current_soc=0.60)
        assert isinstance(plan, SOCPlan)
        assert len(plan.rationale) > 0

    def test_works_with_none_speed_profile(self, planner: SOCPlanner) -> None:
        """plan() works with predicted_speed_profile=None."""
        ctx = RouteContext(upcoming_zone="mixed", distance_to_zone_km=10.0, total_remaining_km=30.0)
        plan = planner.plan(
            TrafficState.CONGESTED, ctx, current_soc=0.55,
            predicted_speed_profile=None,
        )
        assert isinstance(plan, SOCPlan)
        assert 0.0 <= plan.target_soc <= 1.0
