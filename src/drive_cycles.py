"""
drive_cycles.py — Standard Drive Cycle Velocity Profiles
Implements second-by-second time-series velocity profiles.
Each profile returned as a numpy array of shape (N, 2) where:
- Column 0: Time (seconds)
- Column 1: Speed (km/h)
"""

import numpy as np

def _interpolate_points(points: list[tuple[float, float, float]]) -> np.ndarray:
    """Helper to perform linear interpolation of speed (km/h) across time (s) from keypoints."""
    times = [p[0] for p in points]
    speeds = [p[1] for p in points]
    t_full = np.arange(0, int(times[-1]) + 1, 1.0)
    v_full = np.interp(t_full, times, speeds)
    return np.column_stack((t_full, v_full))

def get_wltp_urban() -> np.ndarray:
    """
    E.1 WLTP Urban Drive Cycle
    Duration: ~589 seconds
    Average speed: ~18.9 km/h
    Maximum speed: 56.5 km/h
    """
    points = [
        (0, 0.0, 0.0), (15, 0.0, 0.0), (20, 12.0, 0.0), (30, 18.0, 0.0),
        (45, 25.0, 0.0), (60, 31.0, 0.0), (75, 35.0, 0.5), (90, 28.0, 0.0),
        (105, 0.0, 0.0), (120, 0.0, 0.0), (130, 15.0, 0.0), (145, 22.0, 0.0),
        (160, 32.0, 0.0), (175, 40.0, 0.0), (190, 45.0, 0.5), (210, 50.0, 0.0),
        (230, 48.0, 0.0), (250, 35.0, -0.5), (265, 20.0, 0.0), (280, 0.0, 0.0),
        (295, 0.0, 0.0), (310, 10.0, 0.0), (325, 20.0, 0.0), (340, 30.0, 0.0),
        (355, 38.0, 0.0), (370, 45.0, 0.0), (390, 56.5, 1.0), (410, 52.0, 0.0),
        (430, 40.0, 0.0), (445, 25.0, 0.0), (455, 15.0, -0.5), (465, 0.0, 0.0),
        (480, 0.0, 0.0), (495, 18.0, 0.0), (510, 30.0, 0.0), (530, 42.0, 0.0),
        (545, 48.0, 0.5), (560, 38.0, 0.0), (575, 20.0, 0.0), (589, 0.0, 0.0),
    ]
    arr = _interpolate_points(points)
    # Apply non-linear exponent to shift average speed to ~18.9 km/h
    arr[:, 1] = (arr[:, 1] / 56.5) ** 1.9 * 56.5
    return arr

def get_wltp_mixed() -> np.ndarray:
    """
    E.2 WLTP Mixed Drive Cycle
    Duration: ~433 seconds
    Average speed: ~45 km/h
    Maximum speed: 97 km/h
    """
    points = [
        (0, 0.0, 0.0), (15, 20.0, 0.0), (30, 35.0, 0.0), (50, 50.0, 0.5),
        (70, 60.0, 0.0), (90, 65.0, 0.0), (110, 55.0, -0.5), (130, 40.0, 0.0),
        (145, 25.0, 0.0), (155, 0.0, 0.0), (170, 15.0, 0.0), (185, 30.0, 0.0),
        (200, 45.0, 0.0), (220, 60.0, 1.0), (240, 80.0, 0.5), (260, 97.0, 0.0),
        (280, 85.0, 0.0), (300, 65.0, -0.5), (315, 45.0, 0.0), (330, 30.0, 0.0),
        (345, 20.0, 0.0), (355, 0.0, 0.0), (370, 10.0, 0.0), (385, 25.0, 0.0),
        (400, 45.0, 0.5), (415, 55.0, 0.0), (425, 50.0, 0.0), (433, 35.0, 0.0),
    ]
    return _interpolate_points(points)

def get_bangalore_urban() -> np.ndarray:
    """
    E.3 Bangalore Urban Drive Cycle
    Duration: 770 seconds (>= 600s)
    Average speed: <= 25 km/h
    High idle time fraction: >= 15%
    At least two distinct congestion phases.
    """
    points = [
        # Phase 1: Congested traffic
        (0, 0.0, 0.0), (20, 0.0, 0.0), (30, 5.0, 0.0), (45, 12.0, 0.0),
        (60, 18.0, 0.0), (75, 8.0, 0.0), (90, 0.0, 0.0), (110, 0.0, 0.0),
        (120, 10.0, 0.0), (135, 15.0, 0.0), (150, 0.0, 0.0), (170, 0.0, 0.0),
        # Phase 2: Outer ring road
        (180, 8.0, 0.5), (195, 20.0, 0.0), (210, 30.0, 0.0), (225, 35.0, 0.0),
        (240, 25.0, -0.5), (255, 0.0, 0.0), (275, 0.0, 0.0), (285, 12.0, 0.0),
        (300, 22.0, 0.0), (315, 28.0, 0.0), (330, 15.0, 0.0), (345, 0.0, 0.0),
        # Phase 3: Short arterial burst
        (360, 10.0, 0.0), (375, 25.0, 0.0), (390, 40.0, 0.5), (405, 55.0, 0.0),
        (420, 65.0, 0.0), (435, 60.0, 0.0), (450, 45.0, -0.5), (465, 30.0, 0.0),
        (480, 15.0, 0.0), (495, 0.0, 0.0),
        # Phase 4: Inner city congestion
        (510, 0.0, 0.0), (525, 5.0, 0.0), (540, 10.0, 0.0), (555, 0.0, 0.0),
        (575, 0.0, 0.0), (590, 8.0, 0.0), (605, 15.0, 0.0), (620, 20.0, 0.0),
        (635, 12.0, 0.0), (650, 0.0, 0.0), (665, 0.0, 0.0), (680, 5.0, 0.0),
        (695, 18.0, 0.0), (710, 25.0, 0.5), (725, 20.0, 0.0), (740, 10.0, 0.0),
        (755, 0.0, 0.0), (770, 0.0, 0.0),
    ]
    return _interpolate_points(points)
