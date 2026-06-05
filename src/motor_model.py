"""
motor_model.py — Phase 3: PMSM Traction Motor Model
Author: Antigravity
Date: 2026-06-03

Models the Permanent Magnet Synchronous Motor (PMSM) traction and
regeneration behavior.  Unlike an ICE, an electric motor is highly
efficient (85–95%) across most of its operating range and can instantly
switch between motoring (consuming energy) and generating (recovering
energy during braking).

Dependencies:
    - None (standalone physics module)

Outputs:
    - Traction torque delivery with clamping
    - Electrical power accounting (includes efficiency losses)
    - Regenerative braking power recovery
    - Cumulative regen energy tracking
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

from src.vehicle_config import MOTOR_PEAK_POWER, WHEEL_RADIUS

# ── Motor Constants ────────────────────────────────────────────────
MAX_MOTOR_POWER_KW: float = MOTOR_PEAK_POWER
MAX_MOTOR_TORQUE_NM: float = 250.0
MAX_REGEN_POWER_KW: float = 50.0
MOTOR_EFFICIENCY_PEAK: float = 0.95
MOTOR_EFFICIENCY_LOW_LOAD: float = 0.85

# Drivetrain constants
FINAL_DRIVE_RATIO: float = 8.0
WHEEL_RADIUS_M: float = WHEEL_RADIUS


@dataclass
class MotorState:
    """Represents the instantaneous operating state of the PMSM traction motor.

    The motor can operate in traction mode (EV or HYBRID) or regen mode.
    Tracks cumulative regenerated energy for KPI reporting.

    Attributes:
        motor_power: Current electrical power demand on motor in kW
            (positive = traction).
        motor_torque: Motor torque in Nm.
        motor_efficiency: Current operating efficiency (0.0–1.0).
        regen_power: Recovered power during regenerative braking in kW.
        operating_mode: One of IDLE, EV, HYBRID, REGEN.
        total_regen_energy_kwh: Cumulative regenerated energy in kWh.
    """

    motor_power: float = 0.0
    motor_torque: float = 0.0
    motor_efficiency: float = 0.92
    regen_power: float = 0.0
    operating_mode: str = "IDLE"
    total_regen_energy_kwh: float = 0.0
    motor_rpm: float = 0.0

    def deliver_torque(
        self,
        requested_torque_nm: float,
        speed_kph: float,
        engine_contributing: bool = False,
        gear: int = 1,
        transmission: Optional["TransmissionModel"] = None,
    ) -> float:
        """Attempt to deliver the requested torque at the given vehicle speed.

        Converts speed to motor RPM via wheel radius and final drive ratio.
        Clamps torque to MAX_MOTOR_TORQUE_NM.  Calls calculate_power()
        internally and sets operating_mode.

        Args:
            requested_torque_nm: Torque requested from the motor (Nm).
            speed_kph: Current vehicle speed in km/h.
            engine_contributing: If True, sets mode to HYBRID; else EV.
            gear: Current selected transmission gear.
            transmission: TransmissionModel instance.

        Returns:
            Actual torque delivered after clamping (Nm).
        """
        # Clamp torque
        actual_torque = min(abs(requested_torque_nm), MAX_MOTOR_TORQUE_NM)
        if requested_torque_nm < 0:
            actual_torque = -actual_torque

        self.motor_torque = actual_torque

        # Convert speed to motor RPM
        speed_ms = speed_kph / 3.6
        if transmission is not None:
            self.motor_rpm = transmission.engine_rpm(speed_ms, gear)
        else:
            if WHEEL_RADIUS_M > 0.0:
                wheel_rpm = (speed_ms * 60.0) / (2.0 * math.pi * WHEEL_RADIUS_M)
                self.motor_rpm = wheel_rpm * FINAL_DRIVE_RATIO
            else:
                self.motor_rpm = 0.0

        # Calculate electrical power
        self.calculate_power(self.motor_rpm)

        # Set operating mode
        if engine_contributing:
            self.operating_mode = "HYBRID"
        else:
            self.operating_mode = "EV"

        logger.debug(
            "Motor deliver: %.1f Nm requested, %.1f Nm delivered, "
            "motor RPM=%.0f, mode=%s",
            requested_torque_nm, actual_torque, self.motor_rpm, self.operating_mode,
        )

        return actual_torque

    def calculate_power(self, motor_rpm: float = 3000.0) -> float:
        """Compute electrical power from motor torque and RPM.

        The electrical power drawn = mechanical_power / motor_efficiency.
        This accounts for motor losses.

        Updates self.motor_power in place.

        Returns:
            Electrical power in kW.
        """
        omega = (motor_rpm * 2.0 * math.pi) / 60.0
        mechanical_power_kw = abs(self.motor_torque) * omega / 1000.0
        mechanical_power_kw = min(mechanical_power_kw, MAX_MOTOR_POWER_KW)

        if self.motor_efficiency > 0.0:
            self.motor_power = mechanical_power_kw / self.motor_efficiency
        else:
            self.motor_power = mechanical_power_kw

        return self.motor_power

    def regenerate(self, braking_power_kw: float) -> float:
        """Recover energy during regenerative braking.

        Called when wheel power demand is negative (vehicle decelerating).
        Clamps to MAX_REGEN_POWER_KW and applies motor_efficiency to get
        actual electrical energy recovered.

        Args:
            braking_power_kw: Magnitude of available braking power (positive value).

        Returns:
            Actual recovered power in kW.
        """
        # Clamp to maximum regen capability
        clamped_power = min(abs(braking_power_kw), MAX_REGEN_POWER_KW)

        # Apply motor efficiency to get electrical energy recovered
        recovered = clamped_power * self.motor_efficiency
        self.regen_power = recovered
        self.operating_mode = "REGEN"

        # Update cumulative regen energy (assume 1-second timestep for power→energy)
        # Callers should use actual duration; here we track instantaneous recovery
        self.total_regen_energy_kwh += recovered / 3600.0  # kW to kWh at 1s

        logger.debug(
            "Motor regen: braking=%.1f kW, clamped=%.1f kW, "
            "recovered=%.1f kW (η=%.2f)",
            braking_power_kw, clamped_power, recovered, self.motor_efficiency,
        )

        return recovered


if __name__ == "__main__":
    print("=" * 65)
    print("  MOTOR MODEL — Traction and Regeneration Demo")
    print("=" * 65)

    motor = MotorState()

    # 5 traction steps
    print("\n--- Traction Steps ---")
    print(f"{'Step':>4} | {'Req Torque':>10} | {'Actual':>8} | {'Power':>8} | {'Mode':<8}")
    print("-" * 50)

    traction_requests = [50.0, 100.0, 180.0, 250.0, 300.0]
    speeds = [20.0, 40.0, 60.0, 80.0, 100.0]

    for i, (torque, speed) in enumerate(zip(traction_requests, speeds), 1):
        actual = motor.deliver_torque(torque, speed, engine_contributing=(i > 3))
        print(
            f"  {i:2d}  | {torque:10.1f} | {actual:8.1f} | "
            f"{motor.motor_power:8.2f} | {motor.operating_mode:<8}"
        )

    # 3 regen steps
    print("\n--- Regeneration Steps ---")
    print(f"{'Step':>4} | {'Braking kW':>10} | {'Recovered':>10} | {'Mode':<8}")
    print("-" * 45)

    regen_powers = [15.0, 35.0, 60.0]
    for i, braking in enumerate(regen_powers, 1):
        recovered = motor.regenerate(braking)
        print(
            f"  {i:2d}  | {braking:10.1f} | {recovered:10.2f} | {motor.operating_mode:<8}"
        )

    print(f"\nCumulative regen energy: {motor.total_regen_energy_kwh:.6f} kWh")
    print("=" * 65)
