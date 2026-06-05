"""
battery_model.py — Phase 3: Battery Pack Electrochemical State Model
Author: Antigravity
Date: 2026-06-03

Represents and updates the battery pack's electrochemical state.
Enforces SOC limits and supports energy flow in three directions:
regen charging, generator charging, and motor discharge.

Dependencies:
    - None (standalone model)

Outputs:
    - SOC state tracking with safety limits
    - Charge / discharge energy accounting
    - Cumulative regen energy tracking
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

from src.vehicle_config import (
    BATTERY_CAPACITY,
    INITIAL_SOC,
    MIN_SOC,
    MAX_SOC,
)

# ── SOC Limit Constants ────────────────────────────────────────────
SOC_MINIMUM: float = MIN_SOC          # Below this: force charging behavior
SOC_CRITICAL: float = 0.10         # Below this: safety override territory
SOC_TARGET_URBAN: float = 0.70     # Desired SOC entering urban zones
SOC_MAXIMUM: float = MAX_SOC          # Hard cap — never exceed
SOC_UPPER_BUFFER: float = 0.95     # Soft upper limit — stop active charging


@dataclass
class BatteryState:
    """Represents the instantaneous electrochemical state of the battery pack.

    Models a PHEV battery with 15.0 kWh usable capacity.
    Tracks SOC, enforces safety limits, and records cumulative regen energy.

    Attributes:
        soc: State of Charge as a fraction (0.0–1.0).
        capacity_kwh: Usable battery capacity in kWh.
        temperature: Battery pack temperature in °C.
        voltage: Pack terminal voltage in V (nominal).
        current: Instantaneous current in A (positive = discharge).
        recovered_energy_kwh: Cumulative energy recovered via regen charging.
        efficiency: Configurable charge/discharge efficiency (eta_batt).
        energy_throughput_kwh: Cumulative energy throughput (kWh).
    """

    soc: float = INITIAL_SOC
    capacity_kwh: float = BATTERY_CAPACITY
    temperature: float = 25.0
    voltage: float = 360.0
    current: float = 0.0
    recovered_energy_kwh: float = 0.0
    efficiency: float = 0.95
    energy_throughput_kwh: float = 0.0

    def get_charge_derating(self) -> float:
        """Derate charging power when SOC is high (above 0.90)."""
        if self.soc >= SOC_MAXIMUM:
            return 0.0
        if self.soc > 0.90:
            return (SOC_MAXIMUM - self.soc) / 0.10
        return 1.0

    def get_discharge_derating(self) -> float:
        """Derate discharge power when SOC is low (below 0.25)."""
        if self.soc <= SOC_MINIMUM:
            return 0.0
        if self.soc < 0.25:
            return (self.soc - SOC_MINIMUM) / 0.05
        return 1.0

    def step(self, power_demand_kw: float, dt_s: float) -> tuple[float, float]:
        """Accepts power demand as input, updates SOC and returns (updated_soc, actual_power_delivered_kw)."""
        if power_demand_kw >= 0.0:
            # Discharging
            derating = self.get_discharge_derating()
            clamped_power = power_demand_kw * derating
            actual_energy_kwh = self.discharge(clamped_power, dt_s)
            actual_power = actual_energy_kwh * (3600.0 / dt_s) if dt_s > 0 else 0.0
            return self.soc, actual_power
        else:
            # Charging
            derating = self.get_charge_derating()
            clamped_power = abs(power_demand_kw) * derating
            actual_energy_kwh = self.charge(clamped_power, dt_s, from_regen=True)
            actual_power = -actual_energy_kwh * (3600.0 / dt_s) if dt_s > 0 else 0.0
            return self.soc, actual_power

    def update_soc(self, delta_energy_kwh: float) -> None:
        """Update SOC based on energy delta.

        Positive delta_energy_kwh = charging.  Negative = discharging.
        Clamps result between SOC_CRITICAL and SOC_MAXIMUM.  Logs a
        WARNING if either limit is hit.

        Args:
            delta_energy_kwh: Energy added (positive) or removed (negative) in kWh.
        """
        new_soc = self.soc + (delta_energy_kwh / self.capacity_kwh)

        if new_soc >= SOC_MAXIMUM:
            logger.warning(
                "SOC clamped at MAXIMUM (%.2f → %.2f). Requested delta=%.4f kWh.",
                self.soc, SOC_MAXIMUM, delta_energy_kwh,
            )
            new_soc = SOC_MAXIMUM
        elif new_soc <= SOC_CRITICAL:
            logger.warning(
                "SOC clamped at CRITICAL (%.2f → %.2f). Requested delta=%.4f kWh.",
                self.soc, SOC_CRITICAL, delta_energy_kwh,
            )
            new_soc = SOC_CRITICAL

        self.soc = new_soc
        self.check_soc_limits()

    def charge(
        self,
        power_kw: float,
        duration_s: float,
        from_regen: bool = False,
    ) -> float:
        """Charge the battery at a given power for a given duration.

        Refuses to charge above SOC_MAXIMUM.  If charging would exceed the
        cap, charges only up to the cap and returns the actual energy accepted.

        Args:
            power_kw: Charging power in kW (positive value).
            duration_s: Charging duration in seconds.
            from_regen: If True, the charged energy is also added to
                recovered_energy_kwh (regenerative source tracking).

        Returns:
            Actual energy added in kWh.
        """
        energy_kwh = power_kw * (duration_s / 3600.0)
        chemical_energy_kwh = energy_kwh * self.efficiency

        # Check headroom before charging
        headroom_kwh = (SOC_MAXIMUM - self.soc) * self.capacity_kwh
        if headroom_kwh <= 0.0:
            logger.info(
                "Battery full (SOC=%.4f). Charge refused.", self.soc,
            )
            return 0.0

        actual_chemical_energy = min(chemical_energy_kwh, headroom_kwh)
        self.update_soc(actual_chemical_energy)

        actual_electrical_energy = actual_chemical_energy / self.efficiency
        self.energy_throughput_kwh += actual_electrical_energy

        if from_regen:
            self.recovered_energy_kwh += actual_electrical_energy

        logger.info(
            "Charge: %.2f kW × %.1f s = %.4f kWh accepted (regen=%s). SOC=%.4f",
            power_kw, duration_s, actual_electrical_energy, from_regen, self.soc,
        )
        return actual_electrical_energy

    def discharge(self, power_kw: float, duration_s: float) -> float:
        """Discharge the battery at a given power for a given duration.

        Refuses to discharge below SOC_MINIMUM.  If SOC <= SOC_MINIMUM, logs
        a WARNING and returns 0.0.

        Args:
            power_kw: Discharge power in kW (positive value).
            duration_s: Duration in seconds.

        Returns:
            Actual energy removed in kWh.
        """
        if self.soc <= SOC_MINIMUM:
            logger.warning(
                "Discharge refused: SOC=%.4f <= SOC_MINIMUM=%.2f.",
                self.soc, SOC_MINIMUM,
            )
            return 0.0

        energy_kwh = power_kw * (duration_s / 3600.0)
        chemical_energy_kwh = energy_kwh / self.efficiency

        # Limit discharge so SOC does not go below SOC_MINIMUM
        available_kwh = (self.soc - SOC_MINIMUM) * self.capacity_kwh
        actual_chemical_energy = min(chemical_energy_kwh, available_kwh)

        self.update_soc(-actual_chemical_energy)

        actual_electrical_energy = actual_chemical_energy * self.efficiency
        self.energy_throughput_kwh += actual_electrical_energy

        logger.info(
            "Discharge: %.2f kW × %.1f s = %.4f kWh removed. SOC=%.4f",
            power_kw, duration_s, actual_electrical_energy, self.soc,
        )
        return actual_electrical_energy

    def estimate_available_energy(self) -> float:
        """Return usable energy available above SOC_MINIMUM in kWh.

        Returns:
            Available energy (kWh).  Returns 0.0 if at or below minimum.
        """
        if self.soc <= SOC_MINIMUM:
            return 0.0
        return (self.soc - SOC_MINIMUM) * self.capacity_kwh

    def check_soc_limits(self) -> str:
        """Return a string status describing the current SOC condition.

        Returns exactly one of:
            - ``"CRITICAL"`` if soc <= SOC_CRITICAL
            - ``"LOW"`` if SOC_CRITICAL < soc <= SOC_MINIMUM
            - ``"NORMAL"`` if SOC_MINIMUM < soc < SOC_UPPER_BUFFER
            - ``"HIGH"`` if soc >= SOC_UPPER_BUFFER

        Returns:
            Status string.
        """
        if self.soc <= SOC_CRITICAL:
            logger.warning("Battery SOC CRITICAL: %.4f", self.soc)
            return "CRITICAL"
        elif self.soc <= SOC_MINIMUM:
            logger.warning("Battery SOC LOW: %.4f", self.soc)
            return "LOW"
        elif self.soc >= SOC_UPPER_BUFFER:
            return "HIGH"
        else:
            return "NORMAL"


if __name__ == "__main__":
    print("=" * 60)
    print("  BATTERY MODEL — Charge/Discharge/Regen Cycle Demo")
    print("=" * 60)

    bat = BatteryState(soc=0.50, capacity_kwh=13.8)

    print(f"\nInitial SOC: {bat.soc:.4f}")
    print(f"Available energy: {bat.estimate_available_energy():.2f} kWh")
    print(f"Status: {bat.check_soc_limits()}")
    print()

    # Simulate 10 steps: 5 discharge, 3 regen charge, 2 generator charge
    print("Step | Action          | Power | Duration | Energy   | SOC     | Status")
    print("-" * 78)

    for step in range(1, 11):
        if step <= 5:
            # Discharge at 10 kW for 120s
            energy = bat.discharge(10.0, 120.0)
            action = "Discharge"
            power = 10.0
            duration = 120.0
        elif step <= 8:
            # Regen charge at 15 kW for 60s
            energy = bat.charge(15.0, 60.0, from_regen=True)
            action = "Regen Charge"
            power = 15.0
            duration = 60.0
        else:
            # Generator charge at 20 kW for 90s
            energy = bat.charge(20.0, 90.0, from_regen=False)
            action = "Gen Charge"
            power = 20.0
            duration = 90.0

        status = bat.check_soc_limits()
        print(
            f"  {step:2d}  | {action:<15} | {power:5.1f} | {duration:6.0f}s  | "
            f"{energy:7.4f} | {bat.soc:.4f} | {status}"
        )

    print()
    print(f"Final SOC: {bat.soc:.4f}")
    print(f"Recovered regen energy: {bat.recovered_energy_kwh:.4f} kWh")
    print(f"Available energy: {bat.estimate_available_energy():.2f} kWh")
    print("=" * 60)
