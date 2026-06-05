"""
simulator.py — Phase 7: Intelligent Real-time EMS Simulator
Author: Antigravity
Date: 2026-05-20

Simulates coherent drive cycles with SOC evolution.
Feeds real-time state to the HybridDecisionEngine with predictive advisories.
"""

import time
import logging
import numpy as np
import pandas as pd
from typing import Optional
from collections import deque

from src.config import (
    IDLE_SPEED_KMH, REGEN_MIN_SPEED, WINDOW_SIZE_SHORT,
    SOC_DRAIN_RATES
)
from src.rule_ems import RuleBasedEMS, VehicleState
from src.decision_engine import HybridDecisionEngine
from src.route_logic import RouteManager
from src.predictive_controller import PredictiveEMSController

from src.vehicle_config import *
from src.transmission_model import TransmissionModel
from src.advanced_soc_planner import AdvancedSOCPlanner, RouteInfo
from src.demand_forecaster import DemandForecaster
from src.traffic_predictor import TrafficPredictor, classify_traffic, TrafficFeatures
from src.route_logic import RouteLogic
from src.engine_model import EngineState, MAX_TORQUE_NM as MAX_ENGINE_TORQUE_NM
from src.motor_model import MotorState
from src.battery_model import BatteryState
from src.generator_model import GeneratorState
from src.battery_health import BatteryHealthState
from src.regen_controller import calculate_regen_power
from src.decision_engine import select_mode
from src.ems_modes import EMSMode

logger = logging.getLogger(__name__)


def _clamp(value, lo, hi):
    return max(lo, min(hi, value))


class EMSSimulator:
    def __init__(self, mode: str = "hybrid", route_profile: str = "commute"):
        self.mode = mode
        self.rule_engine = RuleBasedEMS()
        self.route_manager = RouteManager(route_profile=route_profile)
        self.predictive_controller = PredictiveEMSController()

        self.hybrid_engine = None
        if self.mode == "hybrid":
            try:
                self.hybrid_engine = HybridDecisionEngine()
            except FileNotFoundError:
                print("WARNING: Hybrid model not found. Falling back to Rule-based. Run --train first.")
                self.mode = "rule"

    def _generate_coherent_stream(self, n_rows: int, rng: np.random.Generator) -> list:
        """
        Generate coherent telemetry stream with smooth speed/SOC transitions.
        Returns list of dicts (one per timestep).
        """
        rows = []
        speed = 0.0
        soc = rng.uniform(50, 90)
        temp = rng.uniform(20, 30)
        aux = rng.uniform(0.5, 2.5)

        for t in range(n_rows):
            t_route = t % self.route_manager.total_segments
            seg = self.route_manager.get_current_segment(t_route)
            traffic = self.route_manager.get_current_traffic(t_route)
            next_seg = self.route_manager.get_next_segment(t_route)

            # Target speed for segment
            target_speeds = {
                "urban": 35, "stop_go": 15, "arterial": 55,
                "highway": 95, "suburban": 40, "mountain": 45
            }
            target = target_speeds.get(seg, 40) + rng.normal(0, 5)

            # Smooth acceleration towards target
            speed_error = target - speed
            if abs(speed_error) < 3:
                accel = rng.normal(0, 0.3)
            elif speed_error > 0:
                accel = _clamp(rng.normal(1.5, 0.8), 0.3, 4.0)
            else:
                accel = _clamp(rng.normal(-1.5, 0.8), -5.0, -0.3)

            # Random braking events
            if seg in ["stop_go", "urban"] and rng.random() < 0.12:
                accel = rng.uniform(-3.0, -1.0)

            speed = _clamp(speed + accel * 0.8, 0, 130)
            braking = accel < -0.5

            # Power
            if braking:
                power = _clamp(accel * 5.0, -25, -1)
                regen = speed > REGEN_MIN_SPEED
            elif speed < IDLE_SPEED_KMH:
                power = 0.0
                regen = False
            else:
                power = _clamp(speed * 0.15 + accel * 8.0 + aux, 1, 100)
                regen = False

            torque = _clamp(power * 3.5 if abs(power) > 0 else 0, -100, 400)
            grade = rng.normal(0, 1.0)

            rows.append({
                "speed": round(speed, 2),
                "acceleration": round(_clamp(accel, -6, 5), 3),
                "power_required_kw": round(power, 2),
                "torque_required_nm": round(torque, 2),
                "battery_soc": round(soc, 2),
                "battery_temp": round(temp, 2),
                "aux_load_kw": round(aux, 2),
                "grade_angle": round(grade, 2),
                "regen_available": regen,
                "braking": braking,
                "traffic_condition": traffic,
                "current_segment": seg,
                "next_segment": next_seg,
            })

            # SOC evolution
            if braking and regen:
                soc_change = abs(power) * 0.008
            elif power > 0:
                soc_change = -power * 0.004
            else:
                soc_change = -0.02
            soc = _clamp(soc + soc_change, 5, 100)

            # Temperature evolution
            if power > 0:
                temp += rng.uniform(0.01, 0.05)
            else:
                temp -= rng.uniform(0.0, 0.02)
            temp = _clamp(temp, 15, 50)

            aux = _clamp(aux + rng.normal(0, 0.05), 0.3, 4.0)

        return rows

    def run(self, n_rows: int = 50, delay_ms: int = 100) -> None:
        """Run the row-by-row simulation with real-time UI."""
        rng = np.random.default_rng(42)
        stream = self._generate_coherent_stream(n_rows, rng)

        print("\n" + "=" * 115)
        print("  HYBRID EMS REAL-TIME SIMULATOR")
        print("=" * 115)
        print(f"| {'T':>4} | {'Spd':>4} | {'SOC':>5} | {'Pwr':>6} | {'Segment':<10} | {'Next':<10} | {'Mode':<15} | {'Source':<14} | {'Advisory':<20} |")
        print("-" * 115)

        # Suppress logging
        logging.getLogger("src.preprocessing").setLevel(logging.CRITICAL)
        logging.getLogger("src.decision_engine").setLevel(logging.CRITICAL)

        for t, row_dict in enumerate(stream):
            state = VehicleState(**row_dict)
            t_route = t % self.route_manager.total_segments

            # Get predictive context
            lookahead = self.route_manager.get_lookahead(t_route, depth=3)
            dist = self.route_manager.get_distance_remaining(t_route)
            grade = self.route_manager.get_grade_ahead(t_route, depth=3)

            # Decision
            if self.mode == "hybrid" and self.hybrid_engine:
                decision = self.hybrid_engine.decide(
                    state, lookahead=lookahead,
                    distance_remaining=dist, grade_ahead=grade
                )
            else:
                decision = self.rule_engine.decide(state)
                decision.source = "RULE"

            # Advisory summary
            advisory = self.predictive_controller.advise(state, lookahead, dist, grade)
            adv_flags = []
            if advisory.preserve_battery:
                adv_flags.append("PresBat")
            if advisory.prefer_engine:
                adv_flags.append("PrefEng")
            if advisory.prepare_regen:
                adv_flags.append("Regen")
            if advisory.charge_sustain:
                adv_flags.append("ChgSus")
            adv_str = ",".join(adv_flags) if adv_flags else "-"

            source_tag = ("ML" if decision.source == "ML"
                          else ("OVERRIDE" if decision.source == "HYBRID_OVERRIDE"
                                else "RULE"))

            print(f"| {t:>4} | {state.speed:4.0f} | {state.battery_soc:4.1f}% | {state.power_required_kw:5.1f} | {state.current_segment:<10} | {state.next_segment:<10} | {decision.mode:<15} | {source_tag:<14} | {adv_str:<20} |")

            time.sleep(delay_ms / 1000.0)

    def run_drive_cycle(self, cycle_data: np.ndarray, initial_soc: float = 0.70) -> pd.DataFrame:
        """Runs the physics-based simulation on a drive cycle and returns results."""
        return run_physics_simulation(cycle_data, initial_soc)


def run_physics_simulation(cycle_data: np.ndarray, initial_soc: float = 0.70) -> pd.DataFrame:
    """Runs a complete physical HEV powertrain simulation on the provided drive cycle data.

    Executes:
    1. Longitudinal Dynamics
    2. Route segments prediction (RouteLogic)
    3. Traffic state classification & forecast (TrafficPredictor)
    4. Future power demand forecast (DemandForecaster)
    5. SOC Plan selection (AdvancedSOCPlanner)
    6. EMS Mode selection (select_mode)
    7. Powertrain Component updates (EngineState, MotorState, GeneratorState, TransmissionModel)
    8. Battery State update (BatteryState.step)
    """
    import math
    from src.vehicle_model import VehicleState as PhysVehicleState

    transmission = TransmissionModel()
    soc_planner = AdvancedSOCPlanner()
    demand_forecaster = DemandForecaster()
    traffic_predictor = TrafficPredictor()
    route_logic = RouteLogic()

    engine = EngineState(rpm=0.0, torque=0.0, is_running=False)
    motor = MotorState(motor_power=0.0, motor_torque=0.0, operating_mode="IDLE")
    battery = BatteryState(soc=initial_soc, temperature=25.0)
    generator = GeneratorState(input_engine_power=0.0, charging_power=0.0, is_active=False)
    battery_health = BatteryHealthState()

    speeds_kph = cycle_data[:, 1]
    accels_ms2 = np.zeros(len(speeds_kph))
    for t in range(1, len(speeds_kph)):
        accels_ms2[t] = (speeds_kph[t] - speeds_kph[t-1]) / 3.6

    position_km = 0.0
    recent_speeds = []
    recent_accels = []
    records = []

    # Total distance estimation
    total_distance_km = np.sum(speeds_kph / 3.6) * 1.0 / 1000.0

    for t in range(len(cycle_data)):
        speed_kph = speeds_kph[t]
        accel_ms2 = accels_ms2[t]
        speed_ms = speed_kph / 3.6

        # Update position
        position_km += speed_ms * 1.0 / 1000.0

        # VehicleState for longitudinal forces
        v_state = PhysVehicleState(
            speed_kph=speed_kph,
            acceleration=accel_ms2,
            road_grade=0.0
        )
        power_demand_kw = v_state.calculate_wheel_power_demand()
        wheel_torque_demand_nm = v_state.calculate_total_load() * WHEEL_RADIUS

        # RouteLogic
        current_seg = route_logic.current_segment_type(position_km)
        next_seg = route_logic.current_segment_type(position_km + 0.3)
        urban_zone = route_logic.upcoming_urban_zone(position_km, lookahead_km=15.0)

        distance_to_urban = urban_zone.distance_km if urban_zone is not None else 999.0
        urban_duration = urban_zone.estimated_duration_s if urban_zone is not None else 0.0

        route_info = RouteInfo(
            position_km=position_km,
            total_distance_km=total_distance_km,
            next_segment_type=current_seg,
            distance_to_urban_zone_km=distance_to_urban,
            upcoming_urban_duration_s=urban_duration
        )

        # TrafficPredictor
        traffic_prediction = traffic_predictor.predict(speed_kph, time_horizon_s=30.0)

        # DemandForecaster slice
        horizon_slice = cycle_data[t:t+30]
        demand_forecast = demand_forecaster.forecast(horizon_slice, horizon_s=30.0)

        # AdvancedSOCPlanner
        soc_plan = soc_planner.plan(battery.soc, route_info, traffic_prediction, demand_forecast)

        # Traffic classification (for select_mode)
        recent_speeds.append(speed_kph)
        recent_accels.append(accel_ms2)
        if len(recent_speeds) > 30:
            recent_speeds.pop(0)
            recent_accels.pop(0)

        speed_std = np.std(recent_speeds, ddof=0) if len(recent_speeds) > 1 else 0.0
        accel_std = np.std(recent_accels, ddof=0) if len(recent_accels) > 1 else 0.0

        traffic_features = TrafficFeatures(
            speed_kmh=speed_kph,
            speed_std=speed_std,
            accel_std=accel_std
        )
        traffic_state = classify_traffic(traffic_features)

        # Regen Result
        decel_magnitude = abs(accel_ms2) if accel_ms2 < 0 else 0.0
        regen_result = calculate_regen_power(speed_kph, decel_magnitude, battery.soc)

        # Telemetry dict for select_mode
        telemetry = {
            "speed_kmh": speed_kph,
            "soc": battery.soc,
            "power_demand_kw": power_demand_kw,
            "throttle": min(1.0, max(0.0, power_demand_kw / COMBINED_PEAK_POWER)) if power_demand_kw > 0 else 0.0,
            "current_segment": current_seg.lower(),
            "next_segment": next_seg.lower(),
            "position_km": position_km,
            "rpm": engine.rpm,
        }

        # Mode Selection
        ems_mode = select_mode(
            telemetry,
            traffic_state,
            soc_plan,
            battery_health,
            demand_forecast.mean_power_kw,
            regen_result
        )

        # Execute Powertrain & Torque Split
        gear = transmission.gear_select(speed_ms, power_demand_kw)
        gear_ratio = transmission.gear_ratios[gear - 1]

        # Calculate mechanical torque demand at the input shaft
        denom = gear_ratio * transmission.final_drive_ratio * transmission.efficiency
        if denom > 0:
            torque_demand_input = wheel_torque_demand_nm / denom
        else:
            torque_demand_input = 0.0

        # Determine battery power demand depending on EMSMode
        actual_battery_power = 0.0

        if ems_mode == EMSMode.REGEN:
            engine.is_running = False
            engine.torque = 0.0
            engine.rpm = 0.0
            engine.power_kw = 0.0
            engine.efficiency = 0.0
            engine.fuel_rate = 0.0

            actual_regen = motor.regenerate(abs(power_demand_kw))
            battery_power_kw = -actual_regen
            generator.generate_electricity(0.0)

        elif ems_mode == EMSMode.EV:
            engine.is_running = False
            engine.torque = 0.0
            engine.rpm = 0.0
            engine.power_kw = 0.0
            engine.efficiency = 0.0
            engine.fuel_rate = 0.0

            motor.deliver_torque(torque_demand_input, speed_kph, engine_contributing=False, gear=gear, transmission=transmission)
            battery_power_kw = motor.motor_power
            generator.generate_electricity(0.0)

        elif ems_mode == EMSMode.ICE:
            engine.is_running = True
            engine.update_rpm(speed_ms, gear, transmission)
            engine.torque = min(max(0.0, torque_demand_input), MAX_ENGINE_TORQUE_NM)
            engine.calculate_power()
            engine.calculate_efficiency()
            engine.calculate_fuel_consumption()

            motor.deliver_torque(0.0, speed_kph, engine_contributing=True, gear=gear, transmission=transmission)
            generator.generate_electricity(0.0)
            battery_power_kw = 0.0

        elif ems_mode == EMSMode.HYBRID_ASSIST:
            engine.is_running = True
            engine.update_rpm(speed_ms, gear, transmission)
            engine_torque = min(max(0.0, torque_demand_input * 0.60), MAX_ENGINE_TORQUE_NM)
            engine.torque = engine_torque
            engine.calculate_power()
            engine.calculate_efficiency()
            engine.calculate_fuel_consumption()

            motor_torque = torque_demand_input - engine_torque
            motor.deliver_torque(motor_torque, speed_kph, engine_contributing=True, gear=gear, transmission=transmission)
            battery_power_kw = motor.motor_power
            generator.generate_electricity(0.0)

        elif ems_mode == EMSMode.CHARGE_SUSTAIN:
            engine.is_running = True
            engine.update_rpm(speed_ms, gear, transmission)
            engine_traction_torque = min(max(0.0, torque_demand_input), MAX_ENGINE_TORQUE_NM)
            engine.torque = min(engine_traction_torque + 20.0, MAX_ENGINE_TORQUE_NM)
            engine.calculate_power()
            engine.calculate_efficiency()
            engine.calculate_fuel_consumption()

            gen_power_input = max(0.0, (engine.torque - engine_traction_torque) * engine.rpm * 2.0 * math.pi / 60000.0)
            gen_charging = generator.generate_electricity(gen_power_input)

            motor.deliver_torque(0.0, speed_kph, engine_contributing=True, gear=gear, transmission=transmission)
            battery_power_kw = -gen_charging

        else:  # CHARGE_DEPLETING
            engine.is_running = True
            engine.update_rpm(speed_ms, gear, transmission)
            engine_torque = min(max(0.0, torque_demand_input * 0.30), MAX_ENGINE_TORQUE_NM)
            engine.torque = engine_torque
            engine.calculate_power()
            engine.calculate_efficiency()
            engine.calculate_fuel_consumption()

            motor_torque = torque_demand_input - engine_torque
            motor.deliver_torque(motor_torque, speed_kph, engine_contributing=True, gear=gear, transmission=transmission)
            battery_power_kw = motor.motor_power
            generator.generate_electricity(0.0)

        # Step the battery
        _, actual_battery_power = battery.step(battery_power_kw, dt_s=1.0)

        # Temperature & health update
        if actual_battery_power > 0:
            battery.temperature += 0.002 * actual_battery_power
        else:
            battery.temperature += 0.001 * abs(actual_battery_power)
        battery.temperature -= 0.01 * (battery.temperature - 25.0)
        battery.temperature = max(15.0, min(50.0, battery.temperature))

        battery_health.update(battery.soc, battery.temperature, ems_mode == EMSMode.REGEN)

        # Record
        records.append({
            "step": t,
            "speed_kmh": speed_kph,
            "acceleration_ms2": accel_ms2,
            "power_demand_kw": power_demand_kw,
            "actual_battery_power_kw": actual_battery_power,
            "fuel_rate_lh": engine.fuel_rate,
            "engine_efficiency": engine.efficiency,
            "engine_running": engine.is_running,
            "motor_power_kw": motor.motor_power,
            "engine_power_kw": engine.power_kw,
            "rpm": engine.rpm,
            "motor_torque_nm": motor.motor_torque,
            "engine_torque_nm": engine.torque,
            "regen_power_kw": motor.regen_power,
            "position_km": position_km,
            "soc": battery.soc * 100.0,  # [0, 100]
            "soc_target": soc_plan.target_soc_at_urban_entry * 100.0,
            "predicted_speed_kmh": traffic_prediction.predicted_avg_speed_kmh,
            "current_segment": current_seg,
            "ems_mode": ems_mode.value,
            "battery_health_score": battery_health.health_score(),
            "throttle": telemetry["throttle"],
            "temperature_c": battery.temperature,
        })

    return pd.DataFrame(records)


if __name__ == "__main__":
    sim = EMSSimulator(mode="hybrid")
    sim.run(n_rows=20, delay_ms=100)


# ══════════════════════════════════════════════════════════════════
# Phase 2: Standalone scenario library and run_scenario function
# ══════════════════════════════════════════════════════════════════

from src.ems_modes import EMSMode
from src.traffic_predictor import TrafficState, TrafficFeatures, classify_traffic, compute_traffic_features
from src.soc_planner import SOCPlanner, SOCPlan, RouteContext
from src.regen_controller import RegenResult, calculate_regen_power
from src.battery_health import BatteryHealthState
from src.demand_forecaster import estimate_future_demand
from src.decision_engine import select_mode
from src.safety_override import apply_battery_health_constraint
from src.config import (
    REGEN_MIN_DECEL_MS2,
    CONGESTION_SPEED_STD_WINDOW,
)


def _generate_congestion_profile(rng: np.random.Generator) -> list[dict]:
    """Generates a 200-step congestion scenario: alternating 10–60 km/h with high variance."""
    rows: list[dict] = []
    speed = 25.0
    for t in range(200):
        target = rng.uniform(10.0, 60.0)
        speed_error = target - speed
        accel = _clamp(speed_error * 0.3 + rng.normal(0, 1.5), -5.0, 4.0)
        speed = _clamp(speed + accel * 0.8, 0.0, 70.0)
        power = _clamp(speed * 0.12 + accel * 6.0, -20, 80)
        throttle = _clamp(0.3 + accel * 0.08 + rng.normal(0, 0.05), 0.0, 1.0)
        temp = 25.0 + rng.normal(0, 2.0)
        rows.append({
            "speed_kmh": round(speed, 2),
            "acceleration_ms2": round(_clamp(accel, -6, 5), 3),
            "power_demand_kw": round(power, 2),
            "throttle": round(throttle, 3),
            "temperature_c": round(_clamp(temp, 15, 45), 2),
        })
    return rows


def _generate_aggressive_profile(rng: np.random.Generator) -> list[dict]:
    """Generates a 100-step aggressive scenario: frequent hard accel/brake."""
    rows: list[dict] = []
    speed = 20.0
    for t in range(100):
        if rng.random() < 0.4:
            accel = rng.uniform(2.0, 5.0)
            throttle = rng.uniform(0.8, 1.0)
        elif rng.random() < 0.5:
            accel = rng.uniform(-5.0, -1.5)
            throttle = 0.0
        else:
            accel = rng.normal(0, 0.5)
            throttle = _clamp(0.4 + rng.normal(0, 0.1), 0.0, 1.0)
        speed = _clamp(speed + accel * 0.8, 0.0, 140.0)
        power = _clamp(speed * 0.15 + accel * 8.0, -30, 100)
        temp = 28.0 + t * 0.05 + rng.normal(0, 1.0)
        rows.append({
            "speed_kmh": round(speed, 2),
            "acceleration_ms2": round(_clamp(accel, -6, 5), 3),
            "power_demand_kw": round(power, 2),
            "throttle": round(throttle, 3),
            "temperature_c": round(_clamp(temp, 15, 50), 2),
        })
    return rows


def _generate_regen_heavy_profile(rng: np.random.Generator) -> list[dict]:
    """Generates a 150-step regen-heavy scenario: repeated deceleration events >= 1.5 m/s²."""
    rows: list[dict] = []
    speed = 60.0
    for t in range(150):
        if t % 5 == 0:
            accel = rng.uniform(-4.0, -1.5)
        elif t % 5 == 1:
            accel = rng.uniform(-3.0, -1.5)
        else:
            accel = rng.uniform(0.5, 2.5)
        speed = _clamp(speed + accel * 0.8, 0.0, 120.0)
        power = _clamp(speed * 0.12 + accel * 7.0, -30, 80)
        throttle = _clamp(0.3 + max(0, accel) * 0.1, 0.0, 1.0)
        temp = 26.0 + rng.normal(0, 1.5)
        rows.append({
            "speed_kmh": round(speed, 2),
            "acceleration_ms2": round(_clamp(accel, -6, 5), 3),
            "power_demand_kw": round(power, 2),
            "throttle": round(throttle, 3),
            "temperature_c": round(_clamp(temp, 15, 45), 2),
        })
    return rows


def _generate_mixed_route_profile(rng: np.random.Generator) -> list[dict]:
    """Generates a 300-step mixed route: 100 urban + 100 highway + 100 urban."""
    rows: list[dict] = []

    # Urban phase 1 (steps 0–99)
    speed = 10.0
    for t in range(100):
        target = rng.uniform(15, 45)
        accel = _clamp((target - speed) * 0.2 + rng.normal(0, 0.8), -4.0, 3.0)
        speed = _clamp(speed + accel * 0.8, 0.0, 60.0)
        if rng.random() < 0.1:
            accel = rng.uniform(-3.0, -1.0)
            speed = _clamp(speed + accel * 0.8, 0.0, 60.0)
        power = _clamp(speed * 0.12 + accel * 5.0, -15, 50)
        throttle = _clamp(0.3 + accel * 0.06, 0.0, 1.0)
        temp = 24.0 + rng.normal(0, 1.0)
        rows.append({
            "speed_kmh": round(speed, 2),
            "acceleration_ms2": round(_clamp(accel, -6, 5), 3),
            "power_demand_kw": round(power, 2),
            "throttle": round(throttle, 3),
            "temperature_c": round(_clamp(temp, 15, 45), 2),
        })

    # Highway phase (steps 100–199)
    speed = rows[-1]["speed_kmh"]
    for t in range(100):
        target = rng.uniform(85, 120)
        accel = _clamp((target - speed) * 0.15 + rng.normal(0, 0.3), -2.0, 3.0)
        speed = _clamp(speed + accel * 0.8, 0.0, 140.0)
        power = _clamp(speed * 0.18 + accel * 6.0, 5, 80)
        throttle = _clamp(0.5 + accel * 0.05, 0.0, 1.0)
        temp = 27.0 + rng.normal(0, 1.0)
        rows.append({
            "speed_kmh": round(speed, 2),
            "acceleration_ms2": round(_clamp(accel, -6, 5), 3),
            "power_demand_kw": round(power, 2),
            "throttle": round(throttle, 3),
            "temperature_c": round(_clamp(temp, 15, 45), 2),
        })

    # Urban phase 2 (steps 200–299)
    speed = rows[-1]["speed_kmh"]
    for t in range(100):
        target = rng.uniform(15, 45)
        accel = _clamp((target - speed) * 0.2 + rng.normal(0, 0.8), -4.0, 3.0)
        speed = _clamp(speed + accel * 0.8, 0.0, 60.0)
        if rng.random() < 0.12:
            accel = rng.uniform(-3.0, -1.0)
            speed = _clamp(speed + accel * 0.8, 0.0, 60.0)
        power = _clamp(speed * 0.12 + accel * 5.0, -15, 50)
        throttle = _clamp(0.3 + accel * 0.06, 0.0, 1.0)
        temp = 25.0 + rng.normal(0, 1.5)
        rows.append({
            "speed_kmh": round(speed, 2),
            "acceleration_ms2": round(_clamp(accel, -6, 5), 3),
            "power_demand_kw": round(power, 2),
            "throttle": round(throttle, 3),
            "temperature_c": round(_clamp(temp, 15, 45), 2),
        })

    return rows


_SCENARIO_LIBRARY: dict[str, callable] = {
    "congestion": _generate_congestion_profile,
    "aggressive": _generate_aggressive_profile,
    "regen_heavy": _generate_regen_heavy_profile,
    "mixed_route": _generate_mixed_route_profile,
}


def run_scenario(
    scenario_name: str,
    initial_soc: float = 0.80,
) -> pd.DataFrame:
    """Runs a full named scenario and returns per-step results as a DataFrame.

    Uses the Phase 2 decision pipeline: traffic predictor → SOC planner →
    regen controller → demand forecaster → select_mode → safety override.

    Args:
        scenario_name: Key in the scenario library.
        initial_soc: Starting SOC fraction.

    Returns:
        DataFrame with one row per timestep and all per-step record fields.

    Raises:
        KeyError: If scenario_name is not in the scenario library.
    """
    if scenario_name not in _SCENARIO_LIBRARY:
        raise KeyError(
            f"Unknown scenario '{scenario_name}'. "
            f"Available: {list(_SCENARIO_LIBRARY.keys())}"
        )

    rng = np.random.default_rng(42)
    profile = _SCENARIO_LIBRARY[scenario_name](rng)

    soc = initial_soc
    planner = SOCPlanner()
    health = BatteryHealthState()
    records: list[dict] = []

    # Telemetry history for rolling window features
    history_rows: list[dict] = []

    for step_idx, step in enumerate(profile):
        speed = step["speed_kmh"]
        accel = step["acceleration_ms2"]
        power = step["power_demand_kw"]
        throttle = step["throttle"]
        temp = step["temperature_c"]

        # Build telemetry history row
        history_rows.append({
            "speed_kmh": speed,
            "acceleration_ms2": accel,
            "power_demand_kw": power,
        })
        hist_df = pd.DataFrame(history_rows)

        # Traffic classification
        tf = TrafficFeatures(
            speed_kmh=speed,
            speed_std=float(hist_df["speed_kmh"].tail(CONGESTION_SPEED_STD_WINDOW).std(ddof=0))
            if len(hist_df) > 1 else 0.0,
            accel_std=float(hist_df["acceleration_ms2"].tail(CONGESTION_SPEED_STD_WINDOW).std(ddof=0))
            if len(hist_df) > 1 else 0.0,
        )
        traffic_state = classify_traffic(tf)

        # Route context (simplified — infer zone from speed profile)
        if speed > 80:
            zone = "highway"
        elif speed < 30:
            zone = "urban"
        else:
            zone = "mixed"
        n_total = len(profile)
        route_ctx = RouteContext(
            upcoming_zone=zone,
            distance_to_zone_km=max(0.1, (n_total - step_idx) * 0.1),
            total_remaining_km=max(0.1, (n_total - step_idx) * 0.1),
        )

        # SOC planning
        soc_plan = planner.plan(traffic_state, route_ctx, soc)

        # Regen controller
        decel_magnitude = abs(accel) if accel < 0 else 0.0
        regen_result = calculate_regen_power(speed, decel_magnitude, soc)

        # Predictive demand
        predicted_demand = estimate_future_demand(hist_df) if len(hist_df) >= 2 else power

        # Telemetry dict for select_mode
        telemetry = {
            "speed_kmh": speed,
            "acceleration_ms2": accel,
            "power_demand_kw": power,
            "throttle": throttle,
            "soc": soc,
            "stop_duration_s": 0.0,
        }

        # Mode selection
        mode = select_mode(
            telemetry, traffic_state, soc_plan,
            health, predicted_demand, regen_result,
        )

        # Battery health constraint
        mode = apply_battery_health_constraint(mode, health)

        # Update battery health
        health.update(soc, temp, regen_result.is_active)

        # SOC evolution (simplified)
        if mode == EMSMode.REGEN:
            soc_delta = regen_result.regen_power_kw * 0.001
        elif mode in (EMSMode.CHARGE_SUSTAIN, EMSMode.CHARGE_DEPLETING):
            soc_delta = 0.002
        elif mode == EMSMode.EV:
            soc_delta = -max(0.001, power * 0.0003)
        elif mode == EMSMode.ICE:
            soc_delta = 0.0
        else:  # HYBRID_ASSIST
            soc_delta = -max(0.0005, power * 0.00015)
        soc = _clamp(soc + soc_delta, 0.0, 1.0)

        records.append({
            "step": step_idx,
            "speed_kmh": speed,
            "acceleration_ms2": accel,
            "power_demand_kw": power,
            "throttle": throttle,
            "temperature_c": temp,
            "soc": round(soc, 4),
            "ems_mode": mode.value,
            "traffic_state": traffic_state.value,
            "regen_power_kw": round(regen_result.regen_power_kw, 2),
            "battery_health_score": round(health.health_score(), 4),
            "soc_target": round(soc_plan.target_soc, 4),
            "predicted_demand_kw": round(predicted_demand, 2),
        })

    return pd.DataFrame(records)


# ══════════════════════════════════════════════════════════════════
# Phase 3: Drive Cycle Profiles and Extended Scenario Library
# ══════════════════════════════════════════════════════════════════

# Drive cycle format: list of (time_s, speed_kph, road_grade_degrees)

WLTP_URBAN: list[tuple[float, float, float]] = [
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

WLTP_MIXED: list[tuple[float, float, float]] = [
    (0, 0.0, 0.0), (15, 20.0, 0.0), (30, 35.0, 0.0), (50, 50.0, 0.5),
    (70, 60.0, 0.0), (90, 65.0, 0.0), (110, 55.0, -0.5), (130, 40.0, 0.0),
    (145, 25.0, 0.0), (155, 0.0, 0.0), (170, 15.0, 0.0), (185, 30.0, 0.0),
    (200, 45.0, 0.0), (220, 60.0, 1.0), (240, 70.0, 0.5), (260, 76.6, 0.0),
    (280, 72.0, 0.0), (300, 60.0, -0.5), (315, 45.0, 0.0), (330, 30.0, 0.0),
    (345, 20.0, 0.0), (355, 0.0, 0.0), (370, 10.0, 0.0), (385, 25.0, 0.0),
    (400, 45.0, 0.5), (415, 55.0, 0.0), (425, 50.0, 0.0), (433, 35.0, 0.0),
]

BANGALORE_URBAN: list[tuple[float, float, float]] = [
    # Phase 1: Dense traffic near Silk Board junction
    (0, 0.0, 0.0), (20, 0.0, 0.0), (30, 5.0, 0.0), (45, 12.0, 0.0),
    (60, 18.0, 0.0), (75, 8.0, 0.0), (90, 0.0, 0.0), (110, 0.0, 0.0),
    (120, 10.0, 0.0), (135, 15.0, 0.0), (150, 0.0, 0.0), (170, 0.0, 0.0),
    # Phase 2: Moving through outer ring road
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


def _generate_stop_go_urban_profile(rng: np.random.Generator) -> list[dict]:
    """Generates a 250-step aggressive urban profile: 60% time below 20 kph."""
    rows: list[dict] = []
    speed = 0.0
    for t in range(250):
        # Bias toward low speeds with frequent stops
        if rng.random() < 0.20:
            # Full stop event
            accel = _clamp(-speed * 0.5 - rng.uniform(1.0, 3.0), -5.0, 0.0)
        elif speed < 5.0:
            if rng.random() < 0.35:
                accel = rng.uniform(0.5, 2.5)  # Start moving
            else:
                accel = 0.0  # Stay stopped
        elif speed > 20.0 and rng.random() < 0.3:
            accel = rng.uniform(-3.0, -1.0)  # Slow down to maintain low speed
        else:
            target = rng.uniform(8.0, 25.0)
            accel = _clamp((target - speed) * 0.25 + rng.normal(0, 0.8), -4.0, 3.0)

        speed = _clamp(speed + accel * 0.8, 0.0, 40.0)
        power = _clamp(speed * 0.12 + accel * 5.0, -15, 40)
        throttle = _clamp(0.2 + max(0, accel) * 0.08, 0.0, 1.0)
        temp = 26.0 + rng.normal(0, 1.5)
        rows.append({
            "speed_kmh": round(speed, 2),
            "acceleration_ms2": round(_clamp(accel, -6, 5), 3),
            "power_demand_kw": round(power, 2),
            "throttle": round(throttle, 3),
            "temperature_c": round(_clamp(temp, 15, 45), 2),
        })
    return rows


def _generate_highway_cruise_profile(rng: np.random.Generator) -> list[dict]:
    """Generates a 200-step highway cruise: sustained 100–120 kph with mild grade."""
    rows: list[dict] = []
    speed = 30.0  # Start from on-ramp
    for t in range(200):
        if t < 20:
            # Acceleration phase to highway speed
            target = 100.0 + rng.uniform(-5, 5)
            accel = _clamp((target - speed) * 0.15, 0.5, 3.0)
        elif t > 180:
            # Deceleration phase for exit
            target = 40.0
            accel = _clamp((target - speed) * 0.1, -3.0, -0.5)
        else:
            # Cruise phase with slight variations
            target = 110.0 + rng.normal(0, 3)
            accel = _clamp((target - speed) * 0.05 + rng.normal(0, 0.2), -1.5, 1.5)

        speed = _clamp(speed + accel * 0.8, 0.0, 130.0)
        grade = rng.normal(0, 0.8)  # Mild grade variation
        power = _clamp(speed * 0.18 + accel * 7.0 + abs(grade) * 3.0, 5, 90)
        throttle = _clamp(0.5 + accel * 0.05, 0.0, 1.0)
        temp = 27.0 + rng.normal(0, 0.8)
        rows.append({
            "speed_kmh": round(speed, 2),
            "acceleration_ms2": round(_clamp(accel, -6, 5), 3),
            "power_demand_kw": round(power, 2),
            "throttle": round(throttle, 3),
            "temperature_c": round(_clamp(temp, 15, 45), 2),
        })
    return rows


def _generate_urban_to_highway_profile(rng: np.random.Generator) -> list[dict]:
    """Generates a 250-step profile: urban first half, highway second half."""
    rows: list[dict] = []
    speed = 5.0

    # Urban phase (steps 0–124)
    for t in range(125):
        target = rng.uniform(10, 45)
        accel = _clamp((target - speed) * 0.2 + rng.normal(0, 0.8), -4.0, 3.0)
        speed = _clamp(speed + accel * 0.8, 0.0, 55.0)
        if rng.random() < 0.12:
            accel = rng.uniform(-3.0, -1.0)
            speed = _clamp(speed + accel * 0.8, 0.0, 55.0)
        power = _clamp(speed * 0.12 + accel * 5.0, -15, 50)
        throttle = _clamp(0.3 + accel * 0.06, 0.0, 1.0)
        temp = 25.0 + rng.normal(0, 1.0)
        rows.append({
            "speed_kmh": round(speed, 2),
            "acceleration_ms2": round(_clamp(accel, -6, 5), 3),
            "power_demand_kw": round(power, 2),
            "throttle": round(throttle, 3),
            "temperature_c": round(_clamp(temp, 15, 45), 2),
        })

    # Transition + Highway phase (steps 125–249)
    for t in range(125):
        if t < 15:
            target = 60.0 + t * 3.0
        else:
            target = 105.0 + rng.normal(0, 3)
        accel = _clamp((target - speed) * 0.12 + rng.normal(0, 0.3), -2.0, 3.0)
        speed = _clamp(speed + accel * 0.8, 0.0, 130.0)
        power = _clamp(speed * 0.18 + accel * 6.0, 5, 80)
        throttle = _clamp(0.5 + accel * 0.05, 0.0, 1.0)
        temp = 27.0 + rng.normal(0, 0.8)
        rows.append({
            "speed_kmh": round(speed, 2),
            "acceleration_ms2": round(_clamp(accel, -6, 5), 3),
            "power_demand_kw": round(power, 2),
            "throttle": round(throttle, 3),
            "temperature_c": round(_clamp(temp, 15, 45), 2),
        })

    return rows


def _generate_mixed_commute_profile(rng: np.random.Generator) -> list[dict]:
    """Generates a 300-step mixed commute: alternating urban and suburban."""
    rows: list[dict] = []
    speed = 0.0

    phases = [
        (75, 10, 40, "urban"),       # Urban
        (75, 35, 65, "suburban"),     # Suburban
        (75, 10, 45, "urban"),        # Urban
        (75, 30, 60, "suburban"),     # Suburban
    ]

    for n_steps, lo_speed, hi_speed, phase_type in phases:
        for t in range(n_steps):
            target = rng.uniform(lo_speed, hi_speed)
            accel = _clamp((target - speed) * 0.2 + rng.normal(0, 0.8), -4.0, 3.0)
            speed = _clamp(speed + accel * 0.8, 0.0, float(hi_speed + 10))
            if phase_type == "urban" and rng.random() < 0.10:
                accel = rng.uniform(-3.0, -1.0)
                speed = _clamp(speed + accel * 0.8, 0.0, float(hi_speed + 10))
            power = _clamp(speed * 0.12 + accel * 5.0, -15, 60)
            throttle = _clamp(0.3 + accel * 0.06, 0.0, 1.0)
            temp = 25.0 + rng.normal(0, 1.2)
            rows.append({
                "speed_kmh": round(speed, 2),
                "acceleration_ms2": round(_clamp(accel, -6, 5), 3),
                "power_demand_kw": round(power, 2),
                "throttle": round(throttle, 3),
                "temperature_c": round(_clamp(temp, 15, 45), 2),
            })

    return rows


# Register new scenarios (additive — existing entries preserved)
_SCENARIO_LIBRARY["stop_go_urban"] = _generate_stop_go_urban_profile
_SCENARIO_LIBRARY["highway_cruise"] = _generate_highway_cruise_profile
_SCENARIO_LIBRARY["urban_to_highway"] = _generate_urban_to_highway_profile
_SCENARIO_LIBRARY["mixed_commute"] = _generate_mixed_commute_profile
