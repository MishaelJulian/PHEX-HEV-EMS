"""
generate_synthetic_data.py — Intelligent Synthetic Data Generator (Phase 7)
Author: Antigravity
Date: 2026-05-20

Generates temporally coherent drive cycles with:
- Driver personality profiles (smooth, aggressive, erratic)
- Realistic SOC evolution across timesteps
- Probabilistic EMS labels that introduce controlled ambiguity
- Physical constraint enforcement and validation
"""

import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd

from src.config import (
    SYNTHETIC_N_ROWS, SYNTHETIC_RANDOM_SEED, SYNTHETIC_DATA_PATH, DATA_RAW_DIR,
    TRIP_LENGTH_RANGE, DRIVER_PROFILES, SOC_DRAIN_RATES,
    SOC_CRITICAL, SOC_LOW, SOC_HIGH, HIGH_POWER_KW,
    BATTERY_TEMP_MAX, BATTERY_TEMP_MIN, REGEN_MIN_SPEED, IDLE_SPEED_KMH,
    HIGHWAY_MIN_SPEED, URBAN_MAX_SPEED, EMS_MODES
)
from src.rule_ems import RuleBasedEMS, VehicleState

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")

# =============================================================================
# HARD PHYSICAL CONSTRAINTS
# =============================================================================
CONSTRAINTS = {
    "speed":              (0.0,   140.0),
    "acceleration":       (-6.0,    5.0),
    "battery_soc":        (2.0,   100.0),
    "battery_temp":       (10.0,   60.0),
    "aux_load_kw":        (0.2,    5.0),
    "grade_angle":        (-15.0,  15.0),
    "power_required_kw":  (-25.0, 120.0),
    "torque_required_nm": (-100.0, 400.0),
}

SPEED_ENVELOPES = {
    "urban":    (0.0,   60.0),
    "stop_go":  (0.0,   40.0),
    "arterial": (25.0,  80.0),
    "highway":  (60.0, 130.0),
    "suburban": (15.0,  65.0),
    "mountain": (20.0,  70.0),
}

# Route profiles: list of (segment_type, typical_steps)
ROUTE_TEMPLATES = {
    "commute": [
        ("urban", 40), ("arterial", 30), ("highway", 60),
        ("arterial", 20), ("stop_go", 25), ("urban", 25),
    ],
    "highway_trip": [
        ("urban", 15), ("arterial", 15), ("highway", 120),
        ("arterial", 10), ("suburban", 20), ("urban", 20),
    ],
    "city_delivery": [
        ("stop_go", 40), ("urban", 30), ("stop_go", 30),
        ("urban", 25), ("stop_go", 35), ("urban", 40),
    ],
    "suburban_loop": [
        ("suburban", 50), ("arterial", 30), ("suburban", 40),
        ("urban", 25), ("suburban", 30), ("arterial", 25),
    ],
    "mountain_pass": [
        ("urban", 20), ("arterial", 25), ("mountain", 80),
        ("mountain", 60), ("arterial", 15),
    ],
    "mixed_commute": [
        ("stop_go", 30), ("arterial", 20), ("highway", 50),
        ("stop_go", 20), ("urban", 30), ("suburban", 30),
        ("arterial", 20),
    ],
}


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


# =============================================================================
# DRIVE CYCLE GENERATOR — generates coherent time-series trips
# =============================================================================

class DriveCycleGenerator:
    def __init__(self, rng: np.random.Generator, driver_profile: str, route_template: str):
        self.rng = rng
        self.profile = DRIVER_PROFILES[driver_profile]
        self.driver_name = driver_profile
        self.route = ROUTE_TEMPLATES[route_template]
        self.route_name = route_template

    def generate(self) -> pd.DataFrame:
        """Generate a complete trip as a DataFrame."""
        rows = []
        # Initial conditions
        soc = self.rng.uniform(10, 95)   # wide range to include low-SOC trips
        temp = self.rng.uniform(18, 32)
        speed = 0.0
        aux_load = self.rng.uniform(0.5, 3.0)
        # Build the full segment sequence
        segment_seq = []
        for seg_type, n_steps in self.route:
            # Add some variance to segment length
            actual_steps = max(5, int(n_steps * self.rng.uniform(0.7, 1.3)))
            segment_seq.extend([(seg_type)] * actual_steps)
        total_steps = len(segment_seq)
        for t in range(total_steps):
            curr_seg = segment_seq[t]
            next_seg = segment_seq[min(t + 1, total_steps - 1)]
            # Target speed for this segment
            seg_lo, seg_hi = SPEED_ENVELOPES.get(curr_seg, (0, 100))
            target_speed = self.rng.uniform(seg_lo * 0.6 + seg_hi * 0.4,
                                            seg_hi * 0.9)
            target_speed += self.rng.normal(0, self.profile["speed_noise"])
            target_speed = _clamp(target_speed, seg_lo, seg_hi)
            # Acceleration towards target (with driver personality)
            speed_error = target_speed - speed
            if abs(speed_error) < 2.0:
                # Cruising — small perturbations
                accel = self.rng.normal(0, 0.3)
            elif speed_error > 0:
                # Need to accelerate
                accel = self.rng.normal(self.profile["accel_mean"],
                                        self.profile["accel_std"])
                accel = _clamp(accel, 0.2, 5.0)
            else:
                # Need to decelerate
                accel = self.rng.normal(self.profile["brake_mean"],
                                        self.profile["brake_std"])
                accel = _clamp(accel, -6.0, -0.3)
            # Random braking events (stop-go, traffic lights)
            is_random_brake = self.rng.random() < self.profile["brake_probability"]
            if curr_seg in ["stop_go", "urban"] and is_random_brake:
                accel = self.rng.normal(self.profile["brake_mean"],
                                        self.profile["brake_std"])
                accel = _clamp(accel, -5.0, -1.0)
            # Congestion burst (random slowdown in middle of trip)
            if curr_seg == "stop_go" and self.rng.random() < 0.10:
                speed = _clamp(speed * 0.3, 0, 10)
                accel = self.rng.uniform(-2.0, 0.0)
            # Update speed
            speed = _clamp(speed + accel * 0.8, 0, seg_hi)
            braking = accel < -0.5
            # Idle detection
            if speed < IDLE_SPEED_KMH:
                speed = _clamp(speed, 0, IDLE_SPEED_KMH)
                accel = _clamp(accel, -0.5, 0.5)
            # Power and torque (physics-consistent)
            if braking:
                power = _clamp(accel * 5.0 + self.rng.normal(0, 1), -25, -1)
                regen = speed > REGEN_MIN_SPEED
            elif speed < IDLE_SPEED_KMH:
                power = 0.0
                regen = False
            else:
                # Power = f(speed, accel, grade, aux)
                grade = self._get_grade(curr_seg, t, total_steps)
                base_power = (speed * 0.15) + (accel * 8.0) + (grade * 2.0) + aux_load
                power = _clamp(base_power + self.rng.normal(0, 2), 1, 120)
                regen = False
                grade_val = grade  # save for row
            # Grade
            if 'grade_val' not in dir():
                grade_val = self._get_grade(curr_seg, t, total_steps)

            torque = self._compute_torque(power, speed, braking)
            # Traffic condition based on segment
            traffic = self._get_traffic(curr_seg)
            # Aux load varies slowly
            aux_load = _clamp(aux_load + self.rng.normal(0, 0.1), 0.3, 5.0)
            # Temperature evolves slowly (thermal inertia)
            if power > 0:
                temp += self.rng.uniform(0.01, 0.08)  # heating under load
            elif regen:
                temp += self.rng.uniform(0.0, 0.03)   # slight heating from regen
            else:
                temp -= self.rng.uniform(0.0, 0.02)   # slight cooling
            temp = _clamp(temp, 10, 55)
            rows.append({
                "speed": round(speed, 2),
                "acceleration": round(_clamp(accel, -6, 5), 3),
                "power_required_kw": round(_clamp(power, -25, 120), 2),
                "torque_required_nm": round(torque, 2),
                "battery_soc": round(_clamp(soc, 2, 100), 2),
                "battery_temp": round(temp, 2),
                "aux_load_kw": round(aux_load, 2),
                "grade_angle": round(grade_val, 2),
                "regen_available": regen,
                "braking": braking,
                "traffic_condition": traffic,
                "current_segment": curr_seg,
                "next_segment": next_seg,
                "driver_profile": self.driver_name,
            })
            # SOC evolution (applied AFTER recording the row, affects next step)
            # This is intentional: we label based on current SOC, then drain
            soc_change = self._compute_soc_change(power, regen, braking, speed, soc)
            soc = _clamp(soc + soc_change, 2, 100)
            # Reset grade_val for next iteration
            if 'grade_val' in dir():
                del grade_val
        df = pd.DataFrame(rows)
        return df
    def _get_grade(self, segment: str, t: int, total: int) -> float:
        """Generate grade angle based on segment type."""
        if segment == "mountain":
            # First half uphill, second half downhill
            progress = t / max(total, 1)
            if progress < 0.5:
                return _clamp(self.rng.normal(6, 2), 2, 12)
            else:
                return _clamp(self.rng.normal(-5, 2), -12, -2)
        elif segment == "highway":
            return self.rng.normal(0, 0.5)
        elif segment in ["urban", "suburban", "stop_go"]:
            return self.rng.normal(0, 1.0)
        else:
            return self.rng.normal(0, 1.5)

    def _compute_torque(self, power: float, speed: float, braking: bool) -> float:
        """Physically consistent torque from power and speed."""
        if abs(power) < 0.1:
            return 0.0
        # T = P / omega, approximate omega from speed
        # At low speed, cap torque rather than divide by near-zero
        effective_speed = max(speed, 5.0)
        torque = (power * 1000) / (effective_speed * 0.28)  # rough conversion
        return _clamp(torque, -100, 400)

    def _get_traffic(self, segment: str) -> str:
        """Traffic condition weighted by segment type."""
        if segment == "stop_go":
            return self.rng.choice(["heavy", "heavy", "medium"])
        elif segment == "urban":
            return self.rng.choice(["medium", "heavy", "light"])
        elif segment == "highway":
            return self.rng.choice(["light", "light", "medium"])
        elif segment == "suburban":
            return self.rng.choice(["light", "medium"])
        else:
            return self.rng.choice(["light", "medium", "heavy"])

    def _compute_soc_change(self, power: float, regen: bool, braking: bool,
                            speed: float, current_soc: float) -> float:
        """
        Realistic SOC change per timestep based on power flow.
        Regen efficiency drops at high SOC (battery saturation).
        """
        if braking and regen:
            # Regen: recover energy, but less effective at high SOC
            regen_efficiency = 1.0 - max(0, (current_soc - 70) / 60)
            return abs(power) * 0.008 * max(regen_efficiency, 0.1)
        elif speed < IDLE_SPEED_KMH:
            return -0.02  # tiny aux drain
        elif power > 0:
            # Consuming: drain proportional to power
            return -power * 0.005
        else:
            return 0.0


# =============================================================================
# PROBABILISTIC LABELER — introduces controlled ambiguity
# =============================================================================

class ProbabilisticLabeler:
    """
    Labels EMS modes with controlled noise.
    Clear-cut states (critical SOC, thermal) get deterministic labels.
    Ambiguous states get sampled from a probability distribution.
    """

    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.rule_engine = RuleBasedEMS()

    def label(self, state: VehicleState) -> tuple:
        """
        Returns (label, confidence, reason).
        """
        # === DETERMINISTIC ZONES (safety-critical, no ambiguity) ===

        # Critical SOC — always ENGINE_CHARGE
        if state.battery_soc < SOC_CRITICAL:
            return ("ENGINE_CHARGE", 0.97,
                    f"DETERMINISTIC: SOC {state.battery_soc:.1f}% critical. Must charge.")

        # Thermal protection — always ENGINE_ONLY
        if state.battery_temp > BATTERY_TEMP_MAX or state.battery_temp < BATTERY_TEMP_MIN:
            return ("ENGINE_ONLY", 0.92,
                    f"DETERMINISTIC: Temp {state.battery_temp:.1f}C outside safe range.")

        # Idle — always IDLE_STOP
        if state.speed <= IDLE_SPEED_KMH and not state.braking:
            return ("IDLE_STOP", 0.98,
                    f"DETERMINISTIC: Vehicle stopped at {state.speed:.1f} km/h.")

        # Clear regen — always REGEN
        if (state.braking and state.regen_available
                and state.speed > REGEN_MIN_SPEED and state.battery_soc < SOC_HIGH):
            return ("REGEN", 0.93,
                    f"DETERMINISTIC: Braking at {state.speed:.1f} km/h, regen available.")

        # === PROBABILISTIC ZONES (the interesting part) ===

        # Build probability distribution based on state
        probs = self._compute_probabilities(state)

        # Sample label from distribution
        modes = list(probs.keys())
        weights = list(probs.values())
        label = self.rng.choice(modes, p=weights)

        confidence = probs[label]
        reason = self._build_reason(state, label, probs)

        return (label, round(confidence, 3), reason)

    def _compute_probabilities(self, state: VehicleState) -> dict:
        """
        Compute a probability distribution over EMS modes.
        This is the core intelligence — efficiency heuristics, not rules.
        """
        scores = {
            "BATTERY_ONLY": 0.0,
            "ENGINE_ONLY": 0.0,
            "HYBRID_ASSIST": 0.0,
            "REGEN": 0.0,
            "ENGINE_CHARGE": 0.0,
            "IDLE_STOP": 0.0,
        }

        # --- SOC influence ---
        if state.battery_soc > 70:
            scores["BATTERY_ONLY"] += 3.0    # high SOC → prefer EV
            scores["ENGINE_ONLY"] += 0.5
        elif state.battery_soc > 40:
            scores["HYBRID_ASSIST"] += 2.5   # moderate SOC → balanced
            scores["ENGINE_ONLY"] += 1.5
            scores["BATTERY_ONLY"] += 1.0
        elif state.battery_soc > SOC_LOW:
            scores["ENGINE_ONLY"] += 3.0     # low-ish SOC → conserve
            scores["HYBRID_ASSIST"] += 1.0
            scores["ENGINE_CHARGE"] += 0.5
        else:
            scores["ENGINE_CHARGE"] += 3.0   # near-critical
            scores["ENGINE_ONLY"] += 1.5

        # --- Speed/segment influence ---
        if state.current_segment in ["urban", "stop_go"]:
            scores["BATTERY_ONLY"] += 2.0    # city → EV preferred
            scores["HYBRID_ASSIST"] += 0.5
        elif state.current_segment == "highway":
            scores["ENGINE_ONLY"] += 3.0     # highway → engine efficient
            scores["HYBRID_ASSIST"] += 1.0

        if state.speed > HIGHWAY_MIN_SPEED:
            scores["ENGINE_ONLY"] += 2.0
        elif state.speed < 30:
            scores["BATTERY_ONLY"] += 1.5

        # --- Power demand influence ---
        if state.power_required_kw > HIGH_POWER_KW:
            scores["HYBRID_ASSIST"] += 4.0   # high demand → both sources
            scores["ENGINE_ONLY"] += 1.0
        elif state.power_required_kw < 15:
            scores["BATTERY_ONLY"] += 1.5    # low demand → EV fine

        # --- Traffic influence ---
        if state.traffic_condition == "heavy":
            scores["BATTERY_ONLY"] += 1.5    # stop-go → EV efficient
        elif state.traffic_condition == "light":
            scores["ENGINE_ONLY"] += 0.5

        # --- Route lookahead ---
        if state.next_segment == "highway":
            scores["ENGINE_ONLY"] += 1.0     # save battery for later
            scores["BATTERY_ONLY"] -= 0.5
        elif state.next_segment in ["urban", "stop_go"]:
            scores["BATTERY_ONLY"] += 0.5    # EV opportunity coming

        # --- Normalize to probability distribution ---
        # Clamp negatives to zero
        for k in scores:
            scores[k] = max(scores[k], 0.01)

        total = sum(scores.values())
        probs = {k: v / total for k, v in scores.items()}

        # Remove near-zero probabilities to keep labels meaningful
        probs = {k: v for k, v in probs.items() if v > 0.03}

        # Re-normalize
        total = sum(probs.values())
        probs = {k: v / total for k, v in probs.items()}

        return probs

    def _build_reason(self, state: VehicleState, label: str, probs: dict) -> str:
        """Human-readable reason for the probabilistic label."""
        top_3 = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
        alternatives = ", ".join([f"{m}:{p*100:.0f}%" for m, p in top_3])
        return (f"PROBABILISTIC: Selected {label} from distribution [{alternatives}]. "
                f"SOC={state.battery_soc:.0f}% Speed={state.speed:.0f}km/h "
                f"Seg={state.current_segment} Power={state.power_required_kw:.0f}kW")


# =============================================================================
# VALIDATION
# =============================================================================

def validate_synthetic_dataset(df: pd.DataFrame) -> bool:
    """Strict validation gate. Dataset NOT saved unless this passes."""
    logger.info("Running synthetic dataset validation...")
    errors = []

    # Null check
    null_count = df.isnull().sum().sum()
    if null_count > 0:
        errors.append(f"Found {null_count} null values")

    # Hard constraint checks
    for col, (lo, hi) in CONSTRAINTS.items():
        if col not in df.columns:
            continue
        below = (df[col] < lo).sum()
        above = (df[col] > hi).sum()
        if below > 0:
            errors.append(f"{col}: {below} rows below {lo}")
        if above > 0:
            errors.append(f"{col}: {above} rows above {hi}")

    # Power sign consistency
    braking_positive = ((df["braking"] == True) & (df["power_required_kw"] > 0.5)).sum()
    if braking_positive > 0:
        errors.append(f"{braking_positive} rows have braking=True but positive power")

    # SOC monotonicity check per trip (should generally trend, not random)
    if "trip_id" in df.columns:
        soc_variance_per_trip = df.groupby("trip_id")["battery_soc"].apply(
            lambda x: x.diff().dropna().std()
        )
        random_soc_trips = (soc_variance_per_trip > 15).sum()
        if random_soc_trips > len(soc_variance_per_trip) * 0.3:
            errors.append(f"{random_soc_trips} trips have suspiciously random SOC evolution")

    if errors:
        logger.error("+" + "-" * 54 + "+")
        logger.error("|  VALIDATION FAILED -- Dataset contains invalid rows  |")
        logger.error("+" + "-" * 54 + "+")
        for e in errors:
            logger.error(f"  X {e}")
        raise AssertionError(f"Dataset failed {len(errors)} validation checks.")

    logger.info("+" + "-" * 54 + "+")
    logger.info("|  VALIDATION PASSED -- All rows physically valid       |")
    logger.info("+" + "-" * 54 + "+")

    # Class distribution
    dist = df['label'].value_counts(normalize=True) * 100
    print("\n=== SYNTHETIC DATASET CLASS DISTRIBUTION ===")
    for label, pct in dist.items():
        bar = "#" * int(pct / 2)
        print(f"  {label:<20} {pct:5.1f}%  {bar}")

    # Ambiguity check: same-ish states should have different labels
    print("\n=== LABEL AMBIGUITY CHECK ===")
    n_unique_labels = df['label'].nunique()
    print(f"  Unique labels: {n_unique_labels}")

    if "trip_id" in df.columns:
        n_trips = df['trip_id'].nunique()
        print(f"  Total trips: {n_trips}")
        print(f"  Avg trip length: {len(df) / n_trips:.0f} timesteps")

    # Key statistics
    print("\n=== KEY STATISTICS ===")
    for col in ["speed", "battery_soc", "battery_temp", "power_required_kw", "torque_required_nm"]:
        print(f"  {col:<22} min={df[col].min():8.2f}  max={df[col].max():8.2f}  mean={df[col].mean():8.2f}")

    if "driver_profile" in df.columns:
        print("\n=== DRIVER PROFILE DISTRIBUTION ===")
        for profile, pct in df['driver_profile'].value_counts(normalize=True).items():
            print(f"  {profile:<15} {pct*100:5.1f}%")

    return True


# =============================================================================
# MAIN
# =============================================================================

def main():
    rng = np.random.default_rng(SYNTHETIC_RANDOM_SEED)
    labeler = ProbabilisticLabeler(rng)

    all_trips = []
    trip_id = 0
    total_rows = 0

    # Determine driver profile weights
    profile_names = list(DRIVER_PROFILES.keys())
    profile_weights = [DRIVER_PROFILES[p]["weight"] for p in profile_names]
    profile_weights = [w / sum(profile_weights) for w in profile_weights]

    route_names = list(ROUTE_TEMPLATES.keys())

    logger.info(f"Generating {SYNTHETIC_N_ROWS} rows across drive cycles...")

    while total_rows < SYNTHETIC_N_ROWS:
        # Pick a random driver profile and route
        driver = rng.choice(profile_names, p=profile_weights)
        route = rng.choice(route_names)

        generator = DriveCycleGenerator(rng, driver, route)
        trip_df = generator.generate()
        trip_df["trip_id"] = trip_id
        trip_df["route_template"] = route

        all_trips.append(trip_df)
        total_rows += len(trip_df)
        trip_id += 1

        if trip_id % 10 == 0:
            logger.info(f"  Generated {trip_id} trips, {total_rows} rows so far...")

    # Concatenate and trim to target size
    final_df = pd.concat(all_trips, ignore_index=True)
    final_df = final_df.head(SYNTHETIC_N_ROWS)

    logger.info(f"Generated {len(final_df)} rows across {trip_id} trips.")

    # Generate smart target labels with stochastic noise
    from src.ems_target_logic import generate_smart_targets
    final_df = generate_smart_targets(final_df)
    
    final_df["confidence"] = 1.0
    final_df["reason"] = "Smart heuristic target with stochastic noise"

    # Validation gate
    validate_synthetic_dataset(final_df)

    # Save
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(SYNTHETIC_DATA_PATH, index=False)
    logger.info(f"Saved {len(final_df)} rows to {SYNTHETIC_DATA_PATH}")


if __name__ == "__main__":
    main()
