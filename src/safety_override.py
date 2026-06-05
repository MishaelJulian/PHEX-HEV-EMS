"""
safety_override.py — Phase 6: Safety Override Layer
Author: Antigravity
Date: 2026-05-22

Enforces battery, thermal, and electrical safety boundaries on ML EMS decisions.
Integrates Sonali's simplified battery parameters.
"""

import logging
from typing import Tuple
from src.rule_ems import VehicleState
from src.ems_modes import EMSMode
from src.config import (
    SOC_CRITICAL, SOC_LOW, SOC_HIGH, BATTERY_TEMP_MAX, BATTERY_TEMP_MIN,
    BATTERY_TEMP_CRITICAL_C,
)

# SONALI INTEGRATION: Nominal battery constraints
MAX_DISCHARGE_CURRENT_A = 120.0
MAX_CHARGE_CURRENT_A = 80.0
NOMINAL_VOLTAGE_V = 350.0

logger = logging.getLogger(__name__)

class SafetyOverrideLayer:
    """
    Supervisory safety envelope. Validates ML recommendations against physical limits.
    """
    def __init__(self, 
                 soc_critical: float = SOC_CRITICAL,
                 soc_low: float = SOC_LOW,
                 temp_max: float = BATTERY_TEMP_MAX,
                 temp_min: float = BATTERY_TEMP_MIN):
        self.soc_critical = soc_critical
        self.soc_low = soc_low
        self.temp_max = temp_max
        self.temp_min = temp_min
        
    def check_safety(self, ml_mode: str, state: VehicleState) -> Tuple[bool, str]:
        """
        Validates ML action recommendation against safety thresholds.
        
        Args:
            ml_mode: The mode selected by the ML model.
            state: The current vehicle state.
            
        Returns:
            Tuple (is_safe, violation_reason)
        """
        # 1. Critical SOC Bound (Sonali constraint)
        # If SOC is critical, we MUST charge the battery.
        if state.battery_soc < self.soc_critical:
            if ml_mode != EMSMode.CHARGE_SUSTAIN.value and ml_mode != "ENGINE_CHARGE":
                return False, f"Critical SOC ({state.battery_soc:.1f}% < {self.soc_critical}%). Mode {ml_mode} rejected. CHARGE_SUSTAIN required."
                
        # 2. Low SOC EV Mode Restriction
        # Cannot run in BATTERY_ONLY / EV if SOC is below the low threshold.
        if ml_mode in (EMSMode.EV.value, "BATTERY_ONLY") and state.battery_soc < self.soc_low:
            return False, f"Low SOC ({state.battery_soc:.1f}% < {self.soc_low}%). Mode BATTERY_ONLY rejected."
            
        # 3. Thermal Bound Checks (Sonali constraint)
        # Extreme temperatures require ENGINE_ONLY mode to minimize battery stress.
        if state.battery_temp > self.temp_max:
            if ml_mode in (EMSMode.EV.value, EMSMode.HYBRID_ASSIST.value, "BATTERY_ONLY", "HYBRID_ASSIST"):
                return False, f"Thermal violation. Over-temperature ({state.battery_temp:.1f}°C > {self.temp_max}°C). Mode {ml_mode} rejected to prevent thermal runaway."
                
        if state.battery_temp < self.temp_min:
            if ml_mode in (EMSMode.EV.value, EMSMode.HYBRID_ASSIST.value, "BATTERY_ONLY", "HYBRID_ASSIST"):
                return False, f"Thermal violation. Under-temperature ({state.battery_temp:.1f}°C < {self.temp_min}°C). Mode {ml_mode} rejected due to high internal resistance."
                
        # 4. Charging limits (Sonali constraint)
        # Prevent charging (ENGINE_CHARGE / CHARGE_SUSTAIN or REGEN) if battery is already full.
        if state.battery_soc >= SOC_HIGH and ml_mode in (
            EMSMode.CHARGE_SUSTAIN.value, EMSMode.REGEN.value,
            "ENGINE_CHARGE", "REGEN",
        ):
            return False, f"Battery fully charged ({state.battery_soc:.1f}% >= {SOC_HIGH}%). Mode {ml_mode} rejected to avoid overcharging."
            
        return True, ""


def apply_battery_health_constraint(
    mode: EMSMode,
    health_state: "BatteryHealthState",
) -> EMSMode:
    """Modifies mode selection if battery health is degraded.

    Rules:
    - If health_score() < 0.5 AND mode == REGEN: downgrade to CHARGE_SUSTAIN
      (avoid stressing a degraded pack with high regen).
    - If temperature_c > BATTERY_TEMP_CRITICAL_C: force CHARGE_SUSTAIN
      regardless of requested mode.

    Args:
        mode: Proposed EMSMode from decision_engine.
        health_state: Current BatteryHealthState.

    Returns:
        Possibly modified EMSMode.
    """
    # Lazy import to avoid circular dependency
    from src.battery_health import BatteryHealthState  # noqa: F811

    # Critical temperature override
    if health_state.temperature_c > BATTERY_TEMP_CRITICAL_C:
        logger.debug(
            "Battery health constraint: temp %.1f°C > critical %.1f°C → CHARGE_SUSTAIN",
            health_state.temperature_c,
            BATTERY_TEMP_CRITICAL_C,
        )
        return EMSMode.CHARGE_SUSTAIN

    # Degraded pack regen downgrade
    if health_state.health_score() < 0.5 and mode == EMSMode.REGEN:
        logger.debug(
            "Battery health constraint: score=%.3f < 0.5 with REGEN → CHARGE_SUSTAIN",
            health_state.health_score(),
        )
        return EMSMode.CHARGE_SUSTAIN

    return mode
