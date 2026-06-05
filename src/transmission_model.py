"""
transmission_model.py — Transmission gear system mapping engine/motor shafts to wheel shaft.
"""

from __future__ import annotations

import math
from typing import List
from src.vehicle_config import WHEEL_RADIUS

class TransmissionModel:
    """
    Maps engine/motor shaft to wheel shaft.

    Parameters
    ----------
    gear_ratios : list[float]
        Per-gear ratios from input shaft to output shaft.
    final_drive_ratio : float
        Differential/final drive ratio.
    efficiency : float
        Transmission mechanical efficiency (typical: 0.97).
    """

    def __init__(
        self,
        gear_ratios: List[float] = None,
        final_drive_ratio: float = 3.8,
        efficiency: float = 0.97,
    ):
        if gear_ratios is None:
            # Typical 6-speed ratios for a compact hybrid
            self.gear_ratios = [3.5, 2.2, 1.5, 1.1, 0.85, 0.69]
        else:
            self.gear_ratios = gear_ratios
        self.final_drive_ratio = final_drive_ratio
        self.efficiency = efficiency
        self.wheel_radius = WHEEL_RADIUS

    def wheel_torque(self, engine_torque: float, gear: int) -> float:
        """Returns torque at the wheel given engine torque and current gear (1-indexed)."""
        if gear < 1 or gear > len(self.gear_ratios):
            raise ValueError(f"Invalid gear: {gear}")
        gear_ratio = self.gear_ratios[gear - 1]
        # Torque splits from ICE/motor pass through gear ratio, final drive, and efficiency
        return engine_torque * gear_ratio * self.final_drive_ratio * self.efficiency

    def wheel_speed_rpm(self, vehicle_speed_ms: float, gear: int) -> float:
        """Returns wheel rotational speed in RPM from vehicle speed (m/s)."""
        # Gear is ignored as wheel speed depends only on vehicle speed and wheel radius
        if self.wheel_radius <= 0:
            return 0.0
        return (vehicle_speed_ms * 60.0) / (2.0 * math.pi * self.wheel_radius)

    def engine_rpm(self, vehicle_speed_ms: float, gear: int) -> float:
        """Returns engine shaft RPM from vehicle speed (m/s) and gear selection (1-indexed)."""
        if gear < 1 or gear > len(self.gear_ratios):
            raise ValueError(f"Invalid gear: {gear}")
        wheel_rpm = self.wheel_speed_rpm(vehicle_speed_ms, gear)
        gear_ratio = self.gear_ratios[gear - 1]
        engine_rpm_val = wheel_rpm * gear_ratio * self.final_drive_ratio
        # Enforce physical constraints: engine cannot go below 0 when off,
        # but if running it has idle speed. We clamp to redline 6000 RPM.
        return min(max(engine_rpm_val, 0.0), 6000.0)

    def gear_select(self, vehicle_speed_ms: float, power_demand_kw: float) -> int:
        """Returns optimal gear based on speed (m/s) and power demand (kW)."""
        speed_kph = vehicle_speed_ms * 3.6
        if speed_kph < 15.0:
            return 1
        elif speed_kph < 30.0:
            return 2
        elif speed_kph < 50.0:
            return 3
        elif speed_kph < 70.0:
            return 4
        elif speed_kph < 90.0:
            return 5
        else:
            return 6
