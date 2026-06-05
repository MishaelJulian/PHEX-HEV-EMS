"""
test_advanced_soc_planner.py — Unit tests verifying AdvancedSOCPlanner.
"""

import pytest
from src.advanced_soc_planner import AdvancedSOCPlanner, RouteInfo, SOCPlan
from src.traffic_predictor import TrafficPrediction
from src.demand_forecaster import DemandForecast

class TestAdvancedSOCPlanner:
    """Tests for AdvancedSOCPlanner.plan() method and strategic branches."""

    def test_low_soc_recovery_plan(self):
        """Assert low SOC triggers Low SOC Recovery plan."""
        planner = AdvancedSOCPlanner()
        # SOC = 0.20 (<= 0.25 threshold)
        route = RouteInfo(position_km=2.0, total_distance_km=20.0, next_segment_type='HIGHWAY', distance_to_urban_zone_km=15.0, upcoming_urban_duration_s=0.0)
        traffic = TrafficPrediction(predicted_avg_speed_kmh=80.0, congestion_probability=0.1, traffic_density='LOW', speed_trend='STEADY')
        forecast = DemandForecast(mean_power_kw=15.0, peak_power_kw=30.0, regen_potential_kw=0.0, confidence=0.9)

        plan = planner.plan(0.20, route, traffic, forecast)
        assert plan.recommended_mode == "CHARGE_SUSTAIN"
        assert plan.charge_rate_kw == 12.0
        assert plan.target_soc_at_urban_entry == 0.35
        assert "Low SOC recovery" in plan.reasoning

    def test_urban_zone_preserve_plan(self):
        """Assert approaching urban zone triggers SOC Preservation plan."""
        planner = AdvancedSOCPlanner()
        # SOC = 0.60, urban zone within 8.0 km (<= 10.0 km)
        route = RouteInfo(position_km=5.0, total_distance_km=20.0, next_segment_type='HIGHWAY', distance_to_urban_zone_km=8.0, upcoming_urban_duration_s=300.0)
        traffic = TrafficPrediction(predicted_avg_speed_kmh=90.0, congestion_probability=0.1, traffic_density='LOW', speed_trend='STEADY')
        forecast = DemandForecast(mean_power_kw=25.0, peak_power_kw=40.0, regen_potential_kw=0.0, confidence=0.9)

        plan = planner.plan(0.60, route, traffic, forecast)
        assert plan.recommended_mode == "CHARGE_SUSTAIN"
        assert plan.charge_rate_kw == 15.0
        assert plan.target_soc_at_urban_entry == 0.80
        assert "urban zone" in plan.reasoning.lower()

    def test_urban_stop_go_ev_priority(self):
        """Assert urban segment with congested traffic triggers EV priority."""
        planner = AdvancedSOCPlanner()
        # SOC = 0.50, currently urban/stop-go, high congestion
        route = RouteInfo(position_km=15.0, total_distance_km=20.0, next_segment_type='URBAN', distance_to_urban_zone_km=0.0, upcoming_urban_duration_s=300.0)
        traffic = TrafficPrediction(predicted_avg_speed_kmh=12.0, congestion_probability=0.9, traffic_density='CONGESTED', speed_trend='STEADY')
        forecast = DemandForecast(mean_power_kw=8.0, peak_power_kw=15.0, regen_potential_kw=3.0, confidence=0.9)

        plan = planner.plan(0.50, route, traffic, forecast)
        assert plan.recommended_mode == "EV"
        assert plan.discharge_rate_kw == 8.0
        assert plan.target_soc_at_urban_entry == 0.20
        assert "Urban stop-and-go" in plan.reasoning

    def test_highway_cruising_efficiency(self):
        """Assert highway segment triggers ICE cruising mode."""
        planner = AdvancedSOCPlanner()
        # SOC = 0.50, highway cruising, no urban zone soon
        route = RouteInfo(position_km=2.0, total_distance_km=50.0, next_segment_type='HIGHWAY', distance_to_urban_zone_km=30.0, upcoming_urban_duration_s=0.0)
        traffic = TrafficPrediction(predicted_avg_speed_kmh=100.0, congestion_probability=0.05, traffic_density='LOW', speed_trend='STEADY')
        forecast = DemandForecast(mean_power_kw=30.0, peak_power_kw=50.0, regen_potential_kw=0.0, confidence=0.9)

        plan = planner.plan(0.50, route, traffic, forecast)
        assert plan.recommended_mode == "ICE"
        assert plan.target_soc_at_urban_entry == 0.40
        assert "highway cruising" in plan.reasoning.lower()
