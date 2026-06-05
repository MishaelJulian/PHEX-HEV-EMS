"""
EMS AIML Layer — Configuration Module
======================================
Centralized path configuration for all datasets and output directories.
All paths use raw strings to handle Windows backslashes safely.
"""

from pathlib import Path

# ── Project Paths ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# ── Legacy Dataset Paths (Phase 1) ─────────────────────────────────
# Preserved for backward compatibility with inspect_nasa.py / inspect_telemetry.py
NASA_DATA_DIR = Path(
    r"C:\Users\misha\Downloads\1.Battery_Uniform_Distribution_Charge_Discharge_DataSet_2Post"
    r"\Battery_Uniform_Distribution_Charge_Discharge_DataSet_2Post\data"
)
TELEMETRY_DATA_DIR = Path(r"C:\Users\misha\Downloads\archive")
BEHAVIOR_DATA_DIR = Path(r"C:\Users\misha\Downloads\archive (1)")

# ── EMS Dataset Paths ──────────────────────────────────────────────
SYNTHETIC_DATA_PATH = DATA_RAW_DIR / "synthetic_ems_dataset.csv"
PROCESSED_DATA_PATH = DATA_PROCESSED_DIR / "ems_features_clean.csv"

# ── Model Paths ────────────────────────────────────────────────────
RF_MODEL_PATH    = MODELS_DIR / "ems_rf_model.pkl"
GB_MODEL_PATH    = MODELS_DIR / "ems_gb_model.pkl"
SCALER_PATH      = MODELS_DIR / "ems_scaler.pkl"
ENCODER_PATH     = MODELS_DIR / "ems_label_encoder.pkl"

# ── Battery Thresholds ─────────────────────────────────────────────
SOC_CRITICAL     = 15.0   # % — ENGINE_CHARGE forced below this
SOC_LOW          = 25.0   # % — avoid battery-only below this
SOC_HIGH         = 80.0   # % — prefer battery/regen above this
SOH_EOL          = 0.80   # ratio — end of useful battery life
BATTERY_TEMP_MAX = 40.0   # °C — thermal stress flag
BATTERY_TEMP_MIN = 5.0    # °C — cold performance degradation

# ── Power Thresholds ───────────────────────────────────────────────
HIGH_POWER_KW    = 40.0   # kW — HYBRID_ASSIST preferred above
REGEN_MIN_SPEED  = 10.0   # km/h — regen only above this speed
REGEN_MIN_POWER  = -5.0   # kW — minimum recoverable power

# ── Speed Thresholds ───────────────────────────────────────────────
IDLE_SPEED_KMH   = 2.0    # km/h — below this = effectively stopped
URBAN_MAX_SPEED  = 60.0   # km/h
HIGHWAY_MIN_SPEED= 80.0   # km/h

# ── Bangalore Traffic / Road Adaptation ─────────────────────────────
BANGALORE_CONGESTION_THRESHOLD_KMH = 15.0  # km/h - speeds below this flag heavy congestion
BANGALORE_HEAVY_GRIDLOCK_MULTIPLIER = 1.5   # multiplier for battery preservation urgency

# ── Road Segments ──────────────────────────────────────────────────
SEGMENTS = ["urban", "highway", "arterial", "stop_go"]
TRAFFIC_CONDITIONS = ["light", "medium", "heavy"]
EMS_MODES = [
    "BATTERY_ONLY", "ENGINE_ONLY", "HYBRID_ASSIST",
    "REGEN", "ENGINE_CHARGE", "IDLE_STOP"
]

# ── Synthetic Data Generation ──────────────────────────────────────
SYNTHETIC_N_ROWS      = 20000   # total rows to generate (increased for temporal richness)
SYNTHETIC_RANDOM_SEED = 42
TRIP_LENGTH_RANGE     = (150, 400)  # timesteps per trip

# ── Driver Profiles ────────────────────────────────────────────────
DRIVER_PROFILES = {
    "smooth": {
        "accel_mean": 0.8,  "accel_std": 0.4,
        "brake_mean": -1.0, "brake_std": 0.3,
        "speed_noise": 2.0, "brake_probability": 0.08,
        "weight": 0.35,
    },
    "aggressive": {
        "accel_mean": 2.5,  "accel_std": 1.2,
        "brake_mean": -3.0, "brake_std": 1.0,
        "speed_noise": 5.0, "brake_probability": 0.15,
        "weight": 0.30,
    },
    "erratic": {
        "accel_mean": 1.5,  "accel_std": 2.0,
        "brake_mean": -2.0, "brake_std": 1.5,
        "speed_noise": 8.0, "brake_probability": 0.20,
        "weight": 0.35,
    },
}

# ── SOC Drain Rates (% per timestep) ──────────────────────────────
SOC_DRAIN_RATES = {
    "BATTERY_ONLY":   -0.35,   # drains battery
    "ENGINE_ONLY":     0.0,    # no battery change
    "HYBRID_ASSIST":  -0.15,   # moderate drain
    "REGEN":           0.25,   # recovers energy
    "ENGINE_CHARGE":   0.40,   # actively charges
    "IDLE_STOP":      -0.02,   # tiny aux drain
}

# ── Temporal Feature Engineering ───────────────────────────────────
WINDOW_SIZE_SHORT = 5
WINDOW_SIZE_LONG  = 10

# ── Model Hyperparameters ──────────────────────────────────────────
RF_N_ESTIMATORS   = 200
RF_MAX_DEPTH      = 12
RF_MIN_SAMPLES    = 5
GBM_N_ESTIMATORS  = 200
GBM_MAX_DEPTH     = 6
GBM_LEARNING_RATE = 0.05
RANDOM_STATE      = 42
TEST_SIZE         = 0.2
CV_FOLDS          = 5

# ── Display ────────────────────────────────────────────────────────
SEPARATOR = "=" * 60

# ══════════════════════════════════════════════════════════════════
# Phase 2: Supervisor-Guided EMS Extension Constants
# ══════════════════════════════════════════════════════════════════

# ── Stop-and-Go ───────────────────────────────────────────────────
STOP_AND_GO_SPEED_THRESHOLD_KMH: float = 30.0
STOP_AND_GO_MIN_SOC: float = 0.25          # fraction, 0–1

# ── Low SOC Protection ────────────────────────────────────────────
LOW_SOC_THRESHOLD: float = 0.20
CRITICAL_SOC_THRESHOLD: float = 0.10       # triggers CHARGE_SUSTAIN unconditionally

# ── Traffic Detection ─────────────────────────────────────────────
CONGESTION_SPEED_LOWER_KMH: float = 40.0
CONGESTION_SPEED_UPPER_KMH: float = 60.0
STOP_GO_SPEED_MAX_KMH: float = 15.0
CONGESTION_SPEED_STD_WINDOW: int = 10      # samples for rolling std
CONGESTION_ACCEL_STD_WINDOW: int = 10
CONGESTION_SPEED_STD_THRESHOLD: float = 5.0
CONGESTION_ACCEL_STD_THRESHOLD: float = 1.5

# ── Engine Efficiency ─────────────────────────────────────────────
ENGINE_PEAK_EFFICIENCY_RPM: float = 2500.0
ENGINE_EFFICIENCY_HALF_WIDTH_RPM: float = 500.0  # Gaussian sigma

# ── Highway Cruising ──────────────────────────────────────────────
HIGHWAY_SPEED_THRESHOLD_KMH: float = 90.0

# ── Regenerative Braking ──────────────────────────────────────────
REGEN_MIN_DECEL_MS2: float = 0.5           # m/s² minimum decel to activate regen
REGEN_MAX_POWER_KW: float = 50.0           # hardware ceiling
REGEN_SOC_SATURATION: float = 0.90         # above this, regen tapers to zero
REGEN_SOC_MIN_FOR_FULL: float = 0.30       # below this, full regen always active

# ── Maximum Acceleration ──────────────────────────────────────────
MAX_ACCEL_POWER_DEMAND_THRESHOLD_KW: float = 60.0
MAX_ACCEL_THROTTLE_THRESHOLD: float = 0.85  # 0–1 normalised

# ── Dead-Stop Idle ────────────────────────────────────────────────
IDLE_SPEED_THRESHOLD_KMH: float = 1.0
IDLE_SOC_MIN_FOR_EV: float = 0.30
IDLE_STOP_DURATION_FOR_CHARGE_S: float = 30.0

# ── Engine Charging ───────────────────────────────────────────────
ENGINE_CHARGE_POWER_DEMAND_MAX_KW: float = 15.0  # below this = "low demand"

# ── Battery Health ────────────────────────────────────────────────
BATTERY_TEMP_WARN_C: float = 35.0
BATTERY_TEMP_CRITICAL_C: float = 45.0
BATTERY_REGEN_FREQ_WARN: float = 0.5       # regen fraction of total steps

# ── SOC Planner ───────────────────────────────────────────────────
SOC_URBAN_TARGET: float = 0.70             # preserve battery for urban legs
SOC_HIGHWAY_TARGET: float = 0.40           # allow depletion on highway legs
SOC_DEFAULT_TARGET: float = 0.55
