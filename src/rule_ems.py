"""
rule_ems.py — Deterministic rule-based EMS controller.
Author: Antigravity
Date: 2026-05-19
"""

from dataclasses import dataclass
from typing import Optional
import logging
from src.config import *
from src.ems_modes import EMSMode

logger = logging.getLogger(__name__)

@dataclass
class VehicleState:
    """
    Complete snapshot of vehicle state for EMS decision.
    All fields from Alex + Sonali + route logic combined.
    """
    speed: float                    # km/h
    acceleration: float             # m/s²
    power_required_kw: float        # kW
    torque_required_nm: float       # Nm
    battery_soc: float              # %
    battery_temp: float             # °C
    aux_load_kw: float              # kW
    grade_angle: float              # degrees
    regen_available: bool
    braking: bool
    traffic_condition: str          # light / medium / heavy
    current_segment: str            # urban / highway / arterial / stop_go
    next_segment: str               # urban / highway / arterial / traffic
    timestamp: Optional[str] = None

@dataclass
class EMSDecision:
    """Output of EMS controller for one timestep."""
    mode: str                       # EMS_MODES label
    confidence: float               # 0.0 → 1.0
    rule_triggered: str             # which rule fired
    reason: str                     # human-readable explanation
    source: str                     # "RULE" | "ML" | "HYBRID"
    input_state: Optional[VehicleState] = None

class RuleBasedEMS:

    def decide(self, state: VehicleState) -> EMSDecision:
        """
        Apply rule chain to vehicle state.
        Returns EMSDecision with full explanation.
        """
        from src.vehicle_config import MIN_SOC, MAX_SOC, ICE_PEAK_POWER, MOTOR_PEAK_POWER, COMBINED_PEAK_POWER
        
        soc_frac = state.battery_soc / 100.0 if state.battery_soc > 1.0 else state.battery_soc

        # RULE 6 — Regenerative Braking (C.RULE6)
        if state.power_required_kw < 0.0 or (state.braking and state.regen_available):
            if soc_frac >= MAX_SOC:
                logger.info("RULE6_REGEN: Regen disabled - battery full (SOC=%.2f)", soc_frac)
            else:
                clamped_regen = min(abs(state.power_required_kw), MOTOR_PEAK_POWER)
                # 1 Wh = 1/3600 kWh
                recovered_wh = clamped_regen * (1.0 / 3600.0) * 1000.0
                logger.info("RULE6_REGEN: Regenerative braking active. Recovered Wh: %.4f", recovered_wh)
                return EMSDecision(
                    mode=EMSMode.REGEN.value,
                    confidence=0.96,
                    rule_triggered="R1_REGEN",
                    reason=f"RULE6_REGEN: Regenerative braking active. Recovered Wh: {recovered_wh:.4f}",
                    source="RULE",
                    input_state=state
                )

        # RULE 7 — Maximum Acceleration (Combined Power) (C.RULE7)
        if state.acceleration >= 3.0 or state.power_required_kw >= 80.0:
            logger.info("RULE7_MAX_ACCEL: Maximum acceleration requested.")
            return EMSDecision(
                mode=EMSMode.HYBRID_ASSIST.value,
                confidence=0.95,
                rule_triggered="R8_HIGH_DEMAND",
                reason="RULE7_MAX_ACCEL: Maximum acceleration requested. ICE + EV engaged.",
                source="RULE",
                input_state=state
            )

        # RULE 8 — Low SOC Recovery (C.RULE8)
        if soc_frac <= 0.25:
            logger.info("RULE8_SOC_RECOVERY: Low SOC (SOC=%.2f) recovery active.", soc_frac)
            return EMSDecision(
                mode=EMSMode.CHARGE_SUSTAIN.value,
                confidence=0.97,
                rule_triggered="R3_CRITICAL_SOC",
                reason=f"RULE8_SOC_RECOVERY: Low SOC (SOC={soc_frac:.2f}) recovery active.",
                source="RULE",
                input_state=state
            )

        # RULE 1 — Urban Stop-and-Go EV Priority (C.RULE1)
        if state.speed < 50.0 and state.traffic_condition in ("heavy", "congested") and state.current_segment in ("urban", "stop_go"):
            if soc_frac <= (MIN_SOC + 0.05):
                logger.info("RULE1_SUPPRESSED_LOW_SOC: Urban EV Priority suppressed due to low SOC (SOC=%.2f)", soc_frac)
            else:
                logger.info("RULE1_EV_PRIORITY: Urban stop-and-go prioritising EV operation.")
                return EMSDecision(
                    mode=EMSMode.EV.value,
                    confidence=0.90,
                    rule_triggered="R5_CITY_EV",
                    reason="RULE1_EV_PRIORITY: Urban stop-and-go prioritising EV operation.",
                    source="RULE",
                    input_state=state
                )

        # RULE 2 — SOC Preservation Before Urban Zones (C.RULE2)
        upcoming_urban = state.next_segment in ("urban", "stop_go")
        if upcoming_urban and soc_frac < 0.80 and state.current_segment in ("highway", "mixed", "arterial"):
            logger.info("RULE2_SOC_PRESERVE: Upcoming urban zone - preserving SOC (SOC=%.2f).", soc_frac)
            return EMSDecision(
                mode=EMSMode.CHARGE_SUSTAIN.value,
                confidence=0.85,
                rule_triggered="R6_ROUTE_PRESERVE",
                reason=f"RULE2_SOC_PRESERVE: Upcoming urban zone - preserving SOC (SOC={soc_frac:.2f}).",
                source="RULE",
                input_state=state
            )

        # RULE 3 — Traffic-Aware Mode Planning (C.RULE3)
        if state.traffic_condition in ("heavy", "congested"):
            logger.info("RULE3_TRAFFIC_AWARE: High congestion forecast - biasing EV/Hybrid mode.")
            return EMSDecision(
                mode=EMSMode.HYBRID_ASSIST.value,
                confidence=0.85,
                rule_triggered="R9_DEFAULT",
                reason="RULE3_TRAFFIC_AWARE: High congestion forecast - biasing EV/Hybrid mode.",
                source="RULE",
                input_state=state
            )

        # RULE 4 — GPS Route Preview Mode Planning (C.RULE4)
        if state.current_segment == "urban":
            logger.info("RULE4_ROUTE_PREVIEW: Urban segment - favouring EV mode.")
            return EMSDecision(
                mode=EMSMode.EV.value,
                confidence=0.88,
                rule_triggered="R5_CITY_EV",
                reason="RULE4_ROUTE_PREVIEW: Urban segment - favouring EV mode.",
                source="RULE",
                input_state=state
            )
        elif state.current_segment == "highway":
            logger.info("RULE4_ROUTE_PREVIEW: Highway segment - favouring ICE efficiency mode.")
            return EMSDecision(
                mode=EMSMode.ICE.value,
                confidence=0.83,
                rule_triggered="R7_HIGHWAY",
                reason="RULE4_ROUTE_PREVIEW: Highway segment - favouring ICE efficiency mode.",
                source="RULE",
                input_state=state
            )

        # Default fallback
        if state.speed <= IDLE_SPEED_KMH:
            return EMSDecision(
                mode=EMSMode.EV.value,
                confidence=0.99,
                rule_triggered="R2_IDLE",
                reason="Vehicle stopped. Engine off.",
                source="RULE",
                input_state=state
            )

        return EMSDecision(
            mode=EMSMode.HYBRID_ASSIST.value,
            confidence=0.60,
            rule_triggered="R9_DEFAULT",
            reason="Default hybrid fallback.",
            source="RULE",
            input_state=state
        )

