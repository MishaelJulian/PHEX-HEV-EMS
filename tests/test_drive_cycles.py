"""
test_drive_cycles.py — Unit tests verifying drive cycle profiles.
"""

import pytest
import numpy as np
from src.drive_cycles import get_wltp_urban, get_wltp_mixed, get_bangalore_urban

def test_wltp_urban_profile():
    """Verify WLTP Urban duration, max speed, and average speed."""
    cycle = get_wltp_urban()
    assert isinstance(cycle, np.ndarray)
    assert cycle.shape[1] == 2
    assert cycle.shape[0] == 590 # 0 to 589s inclusive
    
    speeds = cycle[:, 1]
    assert np.max(speeds) == pytest.approx(56.5)
    assert 18.0 <= np.mean(speeds) <= 20.0

def test_wltp_mixed_profile():
    """Verify WLTP Mixed duration, max speed, and average speed."""
    cycle = get_wltp_mixed()
    assert isinstance(cycle, np.ndarray)
    assert cycle.shape[1] == 2
    assert cycle.shape[0] == 434 # 0 to 433s inclusive
    
    speeds = cycle[:, 1]
    assert np.max(speeds) == pytest.approx(97.0)
    assert 40.0 <= np.mean(speeds) <= 50.0

def test_bangalore_urban_profile():
    """Verify Bangalore Urban duration, average speed, and idle fraction."""
    cycle = get_bangalore_urban()
    assert isinstance(cycle, np.ndarray)
    assert cycle.shape[0] >= 601
    
    speeds = cycle[:, 1]
    assert np.mean(speeds) <= 25.0
    
    # Idle is speed == 0.0
    idle_count = np.sum(speeds == 0.0)
    idle_fraction = idle_count / len(speeds)
    assert idle_fraction >= 0.15
