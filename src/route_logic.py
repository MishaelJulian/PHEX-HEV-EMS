"""
route_logic.py — Phase 7C: Enhanced Route Integration
Author: Antigravity
Date: 2026-05-20

Simulates Srishti's final route-mapping integration module.
Provides deterministic route sequences with multi-step lookahead,
distance tracking, and grade prediction for predictive EMS control.
"""

import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

# Grade profiles per segment type (mean, std)
SEGMENT_GRADE_PROFILES = {
    "urban":    (0.0, 0.5),
    "stop_go":  (0.0, 0.3),
    "arterial": (0.0, 1.0),
    "highway":  (0.0, 0.5),
    "suburban": (0.0, 0.8),
    "mountain": (4.0, 3.0),  # variable, can be up or down
}

# Approximate distance per timestep (km)
KM_PER_STEP = 0.3


class RouteManager:
    """
    Simulates a navigation system providing route segments.
    Enhanced with multi-step lookahead, distance tracking, and grade prediction.
    """
    def __init__(self, route_profile: str = "commute"):
        self.route_profile = route_profile
        self.segments = self._build_route()
        self.total_segments = len(self.segments)
        self.total_distance_km = self.total_segments * KM_PER_STEP

    def _build_route(self) -> List[Tuple[str, str]]:
        """
        Returns list of (segment_type, traffic_condition) waypoints.
        """
        if self.route_profile == "commute":
            return (
                [("urban", "light")] * 5 +
                [("arterial", "medium")] * 5 +
                [("highway", "light")] * 15 +
                [("stop_go", "heavy")] * 5 +
                [("urban", "light")] * 2
            )
        elif self.route_profile == "highway_trip":
            return (
                [("urban", "light")] * 2 +
                [("highway", "light")] * 30 +
                [("arterial", "medium")] * 2
            )
        elif self.route_profile == "city_delivery":
            return (
                [("stop_go", "heavy")] * 10 +
                [("urban", "medium")] * 10 +
                [("stop_go", "heavy")] * 10
            )
        elif self.route_profile == "suburban_loop":
            return (
                [("suburban", "light")] * 10 +
                [("arterial", "medium")] * 8 +
                [("suburban", "light")] * 8 +
                [("urban", "medium")] * 6 +
                [("suburban", "light")] * 8
            )
        elif self.route_profile == "mountain_pass":
            return (
                [("urban", "light")] * 4 +
                [("arterial", "medium")] * 5 +
                [("mountain", "light")] * 15 +
                [("mountain", "light")] * 10 +
                [("arterial", "medium")] * 3
            )
        elif self.route_profile == "mixed_commute_v2":
            return (
                [("stop_go", "heavy")] * 6 +
                [("arterial", "medium")] * 4 +
                [("highway", "light")] * 10 +
                [("stop_go", "heavy")] * 4 +
                [("urban", "medium")] * 6 +
                [("suburban", "light")] * 5 +
                [("arterial", "medium")] * 5
            )
        else:
            raise ValueError(f"Unknown route profile: {self.route_profile}")

    def get_current_segment(self, time_step: int) -> str:
        """Get the segment type for a given timestep."""
        idx = min(time_step, self.total_segments - 1)
        return self.segments[idx][0]

    def get_current_traffic(self, time_step: int) -> str:
        """Get the traffic condition for a given timestep."""
        idx = min(time_step, self.total_segments - 1)
        return self.segments[idx][1]

    def get_next_segment(self, time_step: int) -> str:
        """Lookahead 1 step for the EMS."""
        lookahead_idx = min(time_step + 1, self.total_segments - 1)
        return self.segments[lookahead_idx][0]

    def get_lookahead(self, time_step: int, depth: int = 3) -> List[str]:
        """
        Multi-step lookahead returning upcoming segment types.
        Returns up to `depth` future segments.
        """
        result = []
        for i in range(1, depth + 1):
            idx = min(time_step + i, self.total_segments - 1)
            result.append(self.segments[idx][0])
        return result

    def get_distance_remaining(self, time_step: int) -> float:
        """Approximate remaining distance in km."""
        steps_left = max(0, self.total_segments - time_step)
        return round(steps_left * KM_PER_STEP, 1)

    def get_grade_ahead(self, time_step: int, depth: int = 3) -> float:
        """
        Average expected grade angle over next `depth` segments.
        Uses segment-type grade profiles.
        """
        grades = []
        for i in range(1, depth + 1):
            idx = min(time_step + i, self.total_segments - 1)
            seg_type = self.segments[idx][0]
            mean_grade, _ = SEGMENT_GRADE_PROFILES.get(seg_type, (0.0, 0.5))
            grades.append(mean_grade)
        return round(sum(grades) / max(len(grades), 1), 2) if grades else 0.0

    def is_route_complete(self, time_step: int) -> bool:
        """Check if simulation has exhausted the route."""
        return time_step >= self.total_segments


from dataclasses import dataclass
from typing import Optional

@dataclass
class UrbanZone:
    distance_km: float
    estimated_duration_s: float
    density_class: str  # 'SUBURBAN', 'URBAN', 'CITY_CENTRE'

@dataclass
class HighwaySegment:
    distance_km: float
    estimated_duration_s: float
    speed_limit_kmh: float

class RouteLogic:
    def current_segment_type(self, position: float) -> str:
        """Returns 'URBAN', 'MIXED', or 'HIGHWAY' for the current position."""
        if position < 3.0:
            return 'URBAN'
        elif position < 10.0:
            return 'HIGHWAY'
        else:
            return 'MIXED'

    def upcoming_urban_zone(self, position: float, lookahead_km: float) -> Optional[UrbanZone]:
        """Returns the nearest upcoming urban zone within lookahead_km, or None."""
        urban_start = 15.0
        if position < urban_start and (urban_start - position) <= lookahead_km:
            return UrbanZone(
                distance_km=float(urban_start - position),
                estimated_duration_s=300.0,
                density_class='URBAN'
            )
        return None

    def upcoming_highway(self, position: float, lookahead_km: float) -> Optional[HighwaySegment]:
        """Returns the nearest upcoming highway segment within lookahead_km, or None."""
        highway_start = 5.0
        if position < highway_start and (highway_start - position) <= lookahead_km:
            return HighwaySegment(
                distance_km=float(highway_start - position),
                estimated_duration_s=600.0,
                speed_limit_kmh=100.0
            )
        return None
