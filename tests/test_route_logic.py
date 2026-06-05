"""
test_route_logic.py — Unit tests verifying RouteLogic behavior.
"""

import pytest
from src.route_logic import RouteLogic

def test_route_logic_segment_classification():
    """Verify current segment classification based on position."""
    rl = RouteLogic()
    assert rl.current_segment_type(1.0) == 'URBAN'
    assert rl.current_segment_type(5.0) == 'HIGHWAY'
    assert rl.current_segment_type(12.0) == 'MIXED'

def test_route_logic_upcoming_urban_zone():
    """Verify detection of upcoming urban zones."""
    rl = RouteLogic()
    # Urban zone is at 15.0 km. Position at 10.0 km with lookahead 10.0 should detect it.
    zone = rl.upcoming_urban_zone(10.0, 10.0)
    assert zone is not None
    assert zone.distance_km == 5.0
    assert zone.density_class == 'URBAN'

    # Position at 10.0 km with lookahead 3.0 should NOT detect it.
    assert rl.upcoming_urban_zone(10.0, 3.0) is None

def test_route_logic_upcoming_highway():
    """Verify detection of upcoming highway segments."""
    rl = RouteLogic()
    # Highway starts at 5.0 km. Position at 2.0 km with lookahead 5.0 should detect it.
    hwy = rl.upcoming_highway(2.0, 5.0)
    assert hwy is not None
    assert hwy.distance_km == 3.0

    # Position at 2.0 km with lookahead 1.0 should NOT detect it.
    assert rl.upcoming_highway(2.0, 1.0) is None
