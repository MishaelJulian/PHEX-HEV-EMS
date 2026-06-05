"""
engine_model.py — Phase 3: Internal Combustion Engine Thermodynamic Model
Author: Antigravity
Date: 2026-06-03

Models the ICE operating state: computes power from RPM and torque,
determines thermal efficiency based on a piecewise linear BSFC-style
curve, and calculates instantaneous fuel consumption.

Physical basis:
    Real ICE efficiency varies with RPM.  At very low RPM, friction losses
    dominate.  At very high RPM, pumping losses and heat rejection dominate.
    The "sweet spot" is 2000–3000 RPM for this engine.

Dependencies:
    - None (standalone physics module)

Outputs:
    - Engine power (kW)
    - Thermal efficiency (0.0–1.0)
    - Fuel consumption rate (L/h)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

from src.vehicle_config import ICE_PEAK_POWER

# ── Engine Constants ───────────────────────────────────────────────
PEAK_EFFICIENCY_RPM_LOW: float = 2000.0
PEAK_EFFICIENCY_RPM_HIGH: float = 3000.0
PEAK_EFFICIENCY: float = 0.38              # 38% peak thermal efficiency
MIN_EFFICIENCY: float = 0.15               # 15% at idle or very high RPM
MAX_POWER_KW: float = ICE_PEAK_POWER               # Engine rated power
MAX_TORQUE_NM: float = 180.0              # Engine peak torque
IDLE_RPM: float = 800.0
FUEL_LOWER_HEATING_VALUE_KJ_PER_L: float = 31600.0  # Gasoline LHV
MAX_RPM: float = 6000.0                   # Redline RPM for efficiency fall-off


@dataclass
class EngineState:
    """Represents the instantaneous operating state of the ICE.

    Models a gasoline engine with peak efficiency at 2000–3000 RPM and
    a piecewise linear efficiency curve.

    Attributes:
        rpm: Engine speed in RPM.
        torque: Output torque in Nm.
        power_kw: Output power in kW (computed).
        fuel_rate: Instantaneous fuel consumption in L/h (computed).
        efficiency: Thermal efficiency as fraction (0.0–1.0, computed).
        is_running: Whether engine is currently on.
    """

    rpm: float = 0.0
    torque: float = 0.0
    power_kw: float = 0.0
    fuel_rate: float = 0.0
    efficiency: float = 0.0
    is_running: bool = False

    def update_rpm(self, vehicle_speed_ms: float, gear: int, transmission: "TransmissionModel") -> float:
        """Update engine speed RPM using the transmission model."""
        if not self.is_running:
            self.rpm = 0.0
        else:
            self.rpm = max(transmission.engine_rpm(vehicle_speed_ms, gear), IDLE_RPM)
        return self.rpm

    def calculate_power(self) -> float:
        """Compute engine output power from RPM and torque.

        Formula:
            P = (torque * rpm * 2π) / (60 * 1000)  [result in kW]

        Updates self.power_kw in place.

        Returns:
            Computed power in kW.  Returns 0.0 if engine is off.
        """
        if not self.is_running:
            self.power_kw = 0.0
            return 0.0

        self.power_kw = (self.torque * self.rpm * 2.0 * math.pi) / (60.0 * 1000.0)
        return self.power_kw

    def calculate_efficiency(self) -> float:
        """Compute thermal efficiency based on RPM using a piecewise linear model.

        In the peak band (2000–3000 RPM): efficiency = PEAK_EFFICIENCY (38%).
        Below 2000 RPM: linearly falls from PEAK_EFFICIENCY at 2000 RPM to
            MIN_EFFICIENCY at IDLE_RPM (800 RPM).
        Above 3000 RPM: linearly falls from PEAK_EFFICIENCY at 3000 RPM to
            MIN_EFFICIENCY at MAX_RPM (6000 RPM).

        Updates self.efficiency in place.

        Returns:
            Efficiency value (float between 0.0 and 1.0).
            Returns 0.0 if engine is off.
        """
        if not self.is_running:
            self.efficiency = 0.0
            return 0.0

        rpm = self.rpm

        if PEAK_EFFICIENCY_RPM_LOW <= rpm <= PEAK_EFFICIENCY_RPM_HIGH:
            # Sweet spot — peak efficiency
            eff = PEAK_EFFICIENCY
        elif rpm < PEAK_EFFICIENCY_RPM_LOW:
            # Below peak band: linear interpolation from IDLE_RPM to 2000 RPM
            if rpm <= IDLE_RPM:
                eff = MIN_EFFICIENCY
            else:
                # Linear interpolation
                fraction = (rpm - IDLE_RPM) / (PEAK_EFFICIENCY_RPM_LOW - IDLE_RPM)
                eff = MIN_EFFICIENCY + fraction * (PEAK_EFFICIENCY - MIN_EFFICIENCY)
        else:
            # Above peak band: linear interpolation from 3000 RPM to MAX_RPM
            if rpm >= MAX_RPM:
                eff = MIN_EFFICIENCY
            else:
                fraction = (rpm - PEAK_EFFICIENCY_RPM_HIGH) / (MAX_RPM - PEAK_EFFICIENCY_RPM_HIGH)
                eff = PEAK_EFFICIENCY - fraction * (PEAK_EFFICIENCY - MIN_EFFICIENCY)

        self.efficiency = max(0.0, min(1.0, eff))
        return self.efficiency

    def calculate_fuel_consumption(self) -> float:
        """Compute instantaneous fuel consumption rate in L/h.

        Formula:
            fuel_rate = (power_kw * 3600) / (efficiency * LHV_kJ_per_L)

        Updates self.fuel_rate in place.

        Returns:
            Fuel rate in L/h.  Returns 0.0 if power is zero or engine is off.
        """
        if self.power_kw == 0.0 or not self.is_running:
            self.fuel_rate = 0.0
            return 0.0

        if self.efficiency <= 0.0:
            logger.warning(
                "Engine efficiency is zero/negative (%.4f) — cannot compute fuel rate.",
                self.efficiency,
            )
            self.fuel_rate = 0.0
            return 0.0

        self.fuel_rate = (self.power_kw * 3600.0) / (
            self.efficiency * FUEL_LOWER_HEATING_VALUE_KJ_PER_L
        )
        return self.fuel_rate


if __name__ == "__main__":
    print("=" * 75)
    print("  ENGINE MODEL — Efficiency, Power, and Fuel Rate Table")
    print("=" * 75)

    fixed_torque = 120.0
    rpm_values = [800, 1500, 2000, 2500, 3000, 4000, 5000]

    print(f"\nFixed Torque: {fixed_torque} Nm")
    print()
    print(f"{'RPM':>6} | {'Power (kW)':>10} | {'Efficiency':>10} | {'Fuel Rate (L/h)':>15}")
    print("-" * 52)

    for rpm in rpm_values:
        engine = EngineState(rpm=float(rpm), torque=fixed_torque, is_running=True)
        power = engine.calculate_power()
        efficiency = engine.calculate_efficiency()
        fuel_rate = engine.calculate_fuel_consumption()

        print(
            f"{rpm:6d} | {power:10.2f} | {efficiency:10.4f} | {fuel_rate:15.3f}"
        )

    print()
    print("Key observations:")
    print(f"  - Peak efficiency ({PEAK_EFFICIENCY:.0%}) at {PEAK_EFFICIENCY_RPM_LOW:.0f}–{PEAK_EFFICIENCY_RPM_HIGH:.0f} RPM")
    print(f"  - Minimum efficiency ({MIN_EFFICIENCY:.0%}) at idle ({IDLE_RPM:.0f} RPM) and redline ({MAX_RPM:.0f} RPM)")
    print(f"  - Efficiency at 2500 RPM != efficiency at 800 RPM (verified above)")
    print("=" * 75)
