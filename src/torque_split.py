"""
torque_split.py — Phase 3: Torque Split Controller
Author: Antigravity
Date: 2026-06-03

Determines precisely how much torque the engine and motor each contribute
to meet driver demand.  This is the real-time decision-making core of the
hybrid EMS.

Integration:
    - Imports EngineState from engine_model.py
    - Imports MotorState from motor_model.py
    - Imports BatteryState from battery_model.py
    - Receives SOCPlan from advanced_soc_planner.py
    - Respects safety_override.py constraints

Dependencies:
    - src.engine_model
    - src.motor_model
    - src.battery_model
    - src.advanced_soc_planner
"""

from __future__ import annotations

import sys
from pathlib import Path
# Add project root to sys.path to enable running this file standalone
sys.path.append(str(Path(__file__).resolve().parent.parent))

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from src.engine_model import EngineState, MAX_TORQUE_NM as MAX_ENGINE_TORQUE_NM
from src.motor_model import MotorState, MAX_MOTOR_TORQUE_NM
from src.battery_model import BatteryState
from src.advanced_soc_planner import SOCPlan

logger = logging.getLogger(__name__)


class TorqueSplitMode(Enum):
    """Torque split operating mode.

    Defines how the powertrain divides torque demand between the ICE
    and the electric motor.
    """

    EV_ONLY = "EV_ONLY"                # Motor only, engine off
    ICE_ONLY = "ICE_ONLY"              # Engine only, motor idle
    HYBRID = "HYBRID"                  # Both contributing
    CHARGE_SUSTAIN = "CHARGE_SUSTAIN"  # Engine runs generator, may also drive wheels
    REGEN = "REGEN"                    # Braking — motor regenerates
    MAX_ACCEL = "MAX_ACCEL"            # Maximum performance — ICE + Motor at full


@dataclass
class TorqueSplitCommand:
    """Output command from the torque split controller.

    Specifies exactly how much torque each powertrain component delivers
    and what mode the system is operating in.

    Attributes:
        engine_torque_nm: Torque delivered by the ICE (Nm).
        motor_torque_nm: Torque delivered by the electric motor (Nm).
        mode: The selected TorqueSplitMode.
        regen_power_kw: Power recovered via regenerative braking (kW).
        charge_power_kw: Power directed to battery via generator (kW).
        reason: Human-readable explanation for the decision.
    """

    engine_torque_nm: float
    motor_torque_nm: float
    mode: TorqueSplitMode
    regen_power_kw: float = 0.0
    charge_power_kw: float = 0.0
    reason: str = ""


# ── Combined torque thresholds ─────────────────────────────────────
_COMBINED_MAX_TORQUE: float = MAX_ENGINE_TORQUE_NM + MAX_MOTOR_TORQUE_NM
_MAX_ACCEL_THRESHOLD: float = 0.85 * _COMBINED_MAX_TORQUE


class TorqueSplitController:
    """Real-time torque split decision controller.

    Implements a 6-priority decision tree that determines how to split
    driver torque demand between the ICE and electric motor based on
    vehicle state, SOC, and traffic conditions.

    Args:
        engine: EngineState instance representing the ICE.
        motor: MotorState instance representing the traction motor.
        battery: BatteryState instance representing the battery pack.
        soc_planner_output: Optional SOCPlan for strategic guidance.
    """

    def __init__(
        self,
        engine: EngineState,
        motor: MotorState,
        battery: BatteryState,
        soc_planner_output: Optional[SOCPlan] = None,
    ) -> None:
        """Initialize the torque split controller.

        Args:
            engine: EngineState instance.
            motor: MotorState instance.
            battery: BatteryState instance.
            soc_planner_output: Optional planner guidance.
        """
        self.engine = engine
        self.motor = motor
        self.battery = battery
        self.soc_planner_output = soc_planner_output

    def decide(
        self,
        driver_demand_nm: float,
        speed_kph: float,
        soc: float,
        traffic_state: str,
    ) -> TorqueSplitCommand:
        """Apply the torque split decision tree.

        Priority order:
            1. Regenerative braking (negative demand)
            2. Maximum acceleration (> 85% combined max)
            3. EV only (low speed + good SOC + low demand + urban)
            4. Charge sustain (low SOC + highway)
            5. ICE only (highway cruise + high SOC)
            6. Hybrid default (60/40 engine/motor split)

        Args:
            driver_demand_nm: Driver torque demand in Nm (negative = braking).
            speed_kph: Current vehicle speed in km/h.
            soc: Current battery SOC as fraction (0.0–1.0).
            traffic_state: Traffic condition string (e.g. "urban", "stop_go",
                "highway", "free_flow").

        Returns:
            TorqueSplitCommand with the decision.
        """
        # -- Priority 1: Regenerative braking -----------------------
        if driver_demand_nm < -5.0:
            braking_power = abs(driver_demand_nm) * 0.1  # Simplified torque->power
            regen_kw = self.motor.regenerate(braking_power)
            cmd = TorqueSplitCommand(
                engine_torque_nm=0.0,
                motor_torque_nm=driver_demand_nm,
                mode=TorqueSplitMode.REGEN,
                regen_power_kw=regen_kw,
                reason="Regenerative braking - recovering kinetic energy",
            )
            logger.info(
                "Torque split: REGEN - demand=%.1f Nm, recovered=%.1f kW",
                driver_demand_nm, regen_kw,
            )
            return cmd

        # -- Priority 2: Maximum acceleration -----------------------
        if driver_demand_nm >= _MAX_ACCEL_THRESHOLD:
            cmd = TorqueSplitCommand(
                engine_torque_nm=MAX_ENGINE_TORQUE_NM,
                motor_torque_nm=MAX_MOTOR_TORQUE_NM,
                mode=TorqueSplitMode.MAX_ACCEL,
                reason="Maximum acceleration - ICE + EV simultaneous",
            )
            logger.info(
                "Torque split: MAX_ACCEL - demand=%.1f Nm >= threshold=%.1f Nm",
                driver_demand_nm, _MAX_ACCEL_THRESHOLD,
            )
            return cmd

        # -- Priority 3: EV only ------------------------------------
        if (
            speed_kph < 50.0
            and soc > 0.30
            and driver_demand_nm < 80.0
            and traffic_state in ("urban", "stop_go")
        ):
            motor_torque = min(driver_demand_nm, MAX_MOTOR_TORQUE_NM)
            cmd = TorqueSplitCommand(
                engine_torque_nm=0.0,
                motor_torque_nm=motor_torque,
                mode=TorqueSplitMode.EV_ONLY,
                reason="Low speed urban - pure EV operation",
            )
            logger.info(
                "Torque split: EV_ONLY - speed=%.1f, SOC=%.4f, demand=%.1f Nm",
                speed_kph, soc, driver_demand_nm,
            )
            return cmd

        # -- Priority 4: Charge sustain -----------------------------
        if soc < 0.25 and speed_kph > 60.0:
            # Engine at efficiency peak zone, excess → generator → battery
            engine_torque = min(driver_demand_nm * 0.8, MAX_ENGINE_TORQUE_NM)
            motor_torque = max(0.0, driver_demand_nm - engine_torque)
            motor_torque = min(motor_torque, MAX_MOTOR_TORQUE_NM)
            charge_kw = 5.0  # Nominal generator charging power
            cmd = TorqueSplitCommand(
                engine_torque_nm=engine_torque,
                motor_torque_nm=motor_torque,
                mode=TorqueSplitMode.CHARGE_SUSTAIN,
                charge_power_kw=charge_kw,
                reason="SOC low on highway - engine charging via generator",
            )
            logger.info(
                "Torque split: CHARGE_SUSTAIN - SOC=%.4f, speed=%.1f, "
                "engine=%.1f Nm, charge=%.1f kW",
                soc, speed_kph, engine_torque, charge_kw,
            )
            return cmd

        # -- Priority 5: ICE only -----------------------------------
        if speed_kph > 80.0 and soc > 0.70 and driver_demand_nm < 120.0:
            engine_torque = min(driver_demand_nm, MAX_ENGINE_TORQUE_NM)
            cmd = TorqueSplitCommand(
                engine_torque_nm=engine_torque,
                motor_torque_nm=0.0,
                mode=TorqueSplitMode.ICE_ONLY,
                reason="Highway cruise - engine at peak efficiency, battery preserved",
            )
            logger.info(
                "Torque split: ICE_ONLY - speed=%.1f, SOC=%.4f, demand=%.1f Nm",
                speed_kph, soc, driver_demand_nm,
            )
            return cmd

        # -- Priority 6: Hybrid default -----------------------------
        engine_share = 0.60
        motor_share = 0.40
        engine_torque = min(driver_demand_nm * engine_share, MAX_ENGINE_TORQUE_NM)
        motor_torque = min(driver_demand_nm * motor_share, MAX_MOTOR_TORQUE_NM)
        cmd = TorqueSplitCommand(
            engine_torque_nm=engine_torque,
            motor_torque_nm=motor_torque,
            mode=TorqueSplitMode.HYBRID,
            reason="Hybrid default - 60% engine / 40% motor",
        )
        logger.info(
            "Torque split: HYBRID - demand=%.1f Nm, engine=%.1f, motor=%.1f",
            driver_demand_nm, engine_torque, motor_torque,
        )
        return cmd


if __name__ == "__main__":
    print("=" * 75)
    print("  TORQUE SPLIT CONTROLLER - All 6 Decision Branches Demo")
    print("=" * 75)

    engine = EngineState(rpm=2500.0, torque=120.0, is_running=True)
    motor = MotorState()
    battery = BatteryState(soc=0.60)

    controller = TorqueSplitController(engine, motor, battery)

    test_cases = [
        {
            "name": "P1: Regen braking",
            "demand": -50.0, "speed": 60.0, "soc": 0.60, "traffic": "urban",
        },
        {
            "name": "P2: Max acceleration",
            "demand": 400.0, "speed": 50.0, "soc": 0.50, "traffic": "highway",
        },
        {
            "name": "P3: EV only (urban, low demand)",
            "demand": 60.0, "speed": 30.0, "soc": 0.65, "traffic": "urban",
        },
        {
            "name": "P4: Charge sustain (low SOC, highway)",
            "demand": 100.0, "speed": 90.0, "soc": 0.20, "traffic": "highway",
        },
        {
            "name": "P5: ICE only (highway cruise, high SOC)",
            "demand": 100.0, "speed": 100.0, "soc": 0.80, "traffic": "highway",
        },
        {
            "name": "P6: Hybrid default",
            "demand": 150.0, "speed": 60.0, "soc": 0.50, "traffic": "suburban",
        },
    ]

    print()
    for tc in test_cases:
        cmd = controller.decide(
            tc["demand"], tc["speed"], tc["soc"], tc["traffic"],
        )
        print(f"--- {tc['name']} ---")
        print(f"  Demand    : {tc['demand']:>8.1f} Nm")
        print(f"  Speed     : {tc['speed']:>8.1f} km/h")
        print(f"  SOC       : {tc['soc']:>8.2f}")
        print(f"  Traffic   : {tc['traffic']}")
        print(f"  -> Mode    : {cmd.mode.value}")
        print(f"  -> Engine  : {cmd.engine_torque_nm:>8.1f} Nm")
        print(f"  -> Motor   : {cmd.motor_torque_nm:>8.1f} Nm")
        print(f"  -> Regen   : {cmd.regen_power_kw:>8.1f} kW")
        print(f"  -> Charge  : {cmd.charge_power_kw:>8.1f} kW")
        print(f"  -> Reason  : {cmd.reason}")
        print()

    print("=" * 75)
