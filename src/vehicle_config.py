"""
vehicle_config.py — Centralized Vehicle Reference Model Configurations
For Mercedes-Benz A220e Plug-in Hybrid.
All constants annotated and documented.
"""

# ── Vehicle Physical Constants ─────────────────────────────────────
VEHICLE_MASS: float = 1800.0          # kg — Total curb mass + driver
WHEEL_RADIUS: float = 0.31            # m — Wheel radius
AERO_DRAG_COEFF: float = 0.22         # — Aerodynamic drag coefficient (Cd)
FRONTAL_AREA: float = 2.2             # m² — Frontal area (A)
ROLLING_RESISTANCE_COEFF: float = 0.012  # — Rolling resistance coefficient (Crr)
TOP_SPEED: float = 210.0              # km/h — Maximum vehicle speed

# ── Powertrain Constants ───────────────────────────────────────────
BATTERY_CAPACITY: float = 15.0        # kWh — Battery capacity
INITIAL_SOC: float = 0.70             # Fraction (70%) — Starting SOC
MIN_SOC: float = 0.20                 # Fraction (20%) — Low SOC limit
MAX_SOC: float = 1.00                 # Fraction (100%) — Full battery limit
MOTOR_PEAK_POWER: float = 80.0        # kW — Electric motor peak power
ICE_PEAK_POWER: float = 120.0         # kW — ICE engine peak power
COMBINED_PEAK_POWER: float = 180.0    # kW — Combined supervisory power limit (160-180 kW range)

# ── Environment Constants ──────────────────────────────────────────
AIR_DENSITY: float = 1.225            # kg/m³ — Standard atmospheric air density
GRAVITY: float = 9.81                 # m/s² — Acceleration due to gravity
