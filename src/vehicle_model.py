"""
vehicle_model.py — Phase 3: Longitudinal Vehicle Road Load Model
Author: Antigravity
Date: 2026-06-03

Computes longitudinal road load forces and wheel power demand from
vehicle state.  Based on Newton's second law applied to a vehicle
moving on a road.  The total force the drivetrain must overcome is the
sum of aerodynamic drag, rolling resistance, grade (gravitational)
force, and inertial force.

Dependencies:
    - None (standalone physics module)

Outputs:
    - Aerodynamic drag force (N)
    - Rolling resistance force (N)
    - Grade resistance force (N)
    - Total road load force (N)
    - Wheel power demand (kW, negative = regen available)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

from src.vehicle_config import (
    AIR_DENSITY,
    GRAVITY,
    VEHICLE_MASS,
    ROLLING_RESISTANCE_COEFF,
    AERO_DRAG_COEFF,
    FRONTAL_AREA,
    WHEEL_RADIUS,
)


@dataclass
class VehicleState:
    """Instantaneous physical state of the vehicle for road load calculation.

    All fields carry SI-compatible units.  Speed is in km/h for convenience
    (converted to m/s internally where needed).

    Attributes:
        speed_kph: Current vehicle speed in km/h.
        acceleration: Longitudinal acceleration in m/s².
        road_grade: Road incline in degrees (positive = uphill).
        vehicle_mass: Total vehicle mass in kg.
        rolling_resistance: Tyre rolling resistance coefficient (Crr).
        drag_coefficient: Aerodynamic drag coefficient (Cd).
        frontal_area: Vehicle frontal area in m².
        wheel_radius: Wheel radius in m.
    """

    speed_kph: float = 0.0
    acceleration: float = 0.0
    road_grade: float = 0.0
    vehicle_mass: float = VEHICLE_MASS
    rolling_resistance: float = ROLLING_RESISTANCE_COEFF
    drag_coefficient: float = AERO_DRAG_COEFF
    frontal_area: float = FRONTAL_AREA
    wheel_radius: float = WHEEL_RADIUS

    # ── Helpers ────────────────────────────────────────────────────

    def _speed_ms(self) -> float:
        """Convert speed from km/h to m/s."""
        return self.speed_kph / 3.6

    def _grade_radians(self) -> float:
        """Convert road grade from degrees to radians."""
        return math.radians(self.road_grade)

    # ── Force Calculations ─────────────────────────────────────────

    def calculate_aero_drag(self) -> float:
        """Return aerodynamic drag force in Newtons.

        Formula:
            F_aero = 0.5 * rho * Cd * A * v²

        Returns:
            Aerodynamic drag force (N).  Always >= 0.
        """
        v = self._speed_ms()
        if v == 0.0:
            return 0.0
        f_aero = 0.5 * AIR_DENSITY * self.drag_coefficient * self.frontal_area * v * v
        return f_aero

    def calculate_rolling_force(self) -> float:
        """Return rolling resistance force in Newtons.

        Formula:
            F_roll = Crr * m * g * cos(grade_rad)

        Returns:
            Rolling resistance force (N).  Zero when stationary.
        """
        if self.speed_kph == 0.0:
            return 0.0
        grade_rad = self._grade_radians()
        f_roll = self.rolling_resistance * self.vehicle_mass * GRAVITY * math.cos(grade_rad)
        return f_roll

    def calculate_grade_force(self) -> float:
        """Return gravitational grade resistance force in Newtons.

        Formula:
            F_grade = m * g * sin(grade_rad)

        Positive when climbing (resists motion).  Negative when descending
        (assists motion — important for downstream regen logic).

        Returns:
            Grade resistance force (N).
        """
        grade_rad = self._grade_radians()
        f_grade = self.vehicle_mass * GRAVITY * math.sin(grade_rad)
        return f_grade

    def calculate_total_load(self) -> float:
        """Return total longitudinal road load force in Newtons.

        Formula:
            F_total = F_aero + F_roll + F_grade + F_inertia

        where F_inertia = vehicle_mass * acceleration.

        A negative result is valid and indicates that the vehicle is
        decelerating under road forces — energy available for regen.

        Returns:
            Total road load force (N).
        """
        f_aero = self.calculate_aero_drag()
        f_roll = self.calculate_rolling_force()
        f_grade = self.calculate_grade_force()
        f_inertia = self.vehicle_mass * self.acceleration
        f_total = f_aero + f_roll + f_grade + f_inertia
        return f_total

    def calculate_wheel_power_demand(self) -> float:
        """Return instantaneous wheel power demand in kilowatts.

        Formula:
            P_wheel = F_total * v / 1000

        A negative value means regenerative power is available.

        Returns:
            Wheel power demand (kW).
        """
        f_total = self.calculate_total_load()
        v = self._speed_ms()
        p_wheel = f_total * v / 1000.0
        logger.debug(
            "Wheel power demand: %.2f kW (F_total=%.1f N, v=%.2f m/s)",
            p_wheel, f_total, v,
        )
        return p_wheel


if __name__ == "__main__":
    print("=" * 60)
    print("  VEHICLE MODEL — Road Load Demonstration")
    print("=" * 60)

    state = VehicleState(
        speed_kph=80.0,
        acceleration=0.0,
        road_grade=2.0,
    )

    print(f"\nVehicle State:")
    print(f"  Speed        : {state.speed_kph} km/h ({state._speed_ms():.2f} m/s)")
    print(f"  Acceleration : {state.acceleration} m/s²")
    print(f"  Road Grade   : {state.road_grade}°")
    print(f"  Mass         : {state.vehicle_mass} kg")
    print(f"  Cd           : {state.drag_coefficient}")
    print(f"  Frontal Area : {state.frontal_area} m²")
    print(f"  Crr          : {state.rolling_resistance}")
    print()

    f_aero = state.calculate_aero_drag()
    f_roll = state.calculate_rolling_force()
    f_grade = state.calculate_grade_force()
    f_total = state.calculate_total_load()
    p_wheel = state.calculate_wheel_power_demand()

    print(f"Computed Forces:")
    print(f"  Aerodynamic Drag   : {f_aero:8.2f} N")
    print(f"  Rolling Resistance : {f_roll:8.2f} N")
    print(f"  Grade Force        : {f_grade:8.2f} N")
    print(f"  Total Road Load    : {f_total:8.2f} N")
    print(f"  Wheel Power Demand : {p_wheel:8.2f} kW")
    print()

    # Downhill example demonstrating negative grade force
    state_downhill = VehicleState(speed_kph=60.0, acceleration=-1.5, road_grade=-3.0)
    f_grade_down = state_downhill.calculate_grade_force()
    p_wheel_down = state_downhill.calculate_wheel_power_demand()
    print(f"Downhill Scenario (60 kph, -3° grade, -1.5 m/s² decel):")
    print(f"  Grade Force        : {f_grade_down:8.2f} N (negative = assists motion)")
    print(f"  Wheel Power Demand : {p_wheel_down:8.2f} kW (negative = regen available)")
    print("=" * 60)
