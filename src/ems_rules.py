"""
ems_rules.py — Phase 3: Named EMS Supervisory Rule Registry
Author: Antigravity
Date: 2026-06-03

Defines the 8 named supervisory rules that govern the EMS decision-making
process.  Each rule has a unique identifier, human-readable description,
condition specification, and action description.

These rules are the codified engineering knowledge that the EMS must
follow.  They are visible, named, and logged when they fire — never
buried in silent if-else chains.

Dependencies:
    - src.battery_model (SOC constants)
    - src.engine_model (engine constants)
    - src.motor_model (motor constants)
"""

from __future__ import annotations

import sys
from pathlib import Path
# Add project root to sys.path to enable running this file standalone
sys.path.append(str(Path(__file__).resolve().parent.parent))

import logging
from typing import List, Dict, Any, Optional

from src.battery_model import SOC_MINIMUM, SOC_TARGET_URBAN
from src.engine_model import MAX_TORQUE_NM as MAX_ENGINE_TORQUE
from src.motor_model import MAX_MOTOR_TORQUE_NM as MAX_MOTOR_TORQUE

logger = logging.getLogger(__name__)

# -- Rule Registry --------------------------------------------------
EMS_RULES: List[Dict[str, str]] = [
    {
        "id": "RULE_01",
        "name": "URBAN_EV_PRIORITY",
        "description": "Urban stop-and-go -> prioritize EV operation",
        "condition": "traffic_state in ['urban', 'stop_go'] and soc > SOC_MINIMUM",
        "action": "Set mode = EV_ONLY",
    },
    {
        "id": "RULE_02",
        "name": "LOW_SOC_CHARGING",
        "description": "SOC < 20% -> activate charging behavior",
        "condition": "soc < SOC_MINIMUM",
        "action": "Set mode = CHARGE_SUSTAIN, engine runs generator",
    },
    {
        "id": "RULE_03",
        "name": "CITY_CONGESTION_EV",
        "description": "Congested city traffic -> maximize electric operation",
        "condition": "traffic_congestion_level > 0.7 and speed_kph < 30",
        "action": "Force EV_ONLY if SOC permits",
    },
    {
        "id": "RULE_04",
        "name": "HIGHWAY_ENGINE_EFFICIENCY",
        "description": "Highway cruising -> keep engine near 2000-3000 RPM",
        "condition": "speed_kph > 70 and segment_type == 'highway'",
        "action": "Target engine RPM = 2500, adjust load split accordingly",
    },
    {
        "id": "RULE_05",
        "name": "REGEN_ON_BRAKING",
        "description": "Braking -> regenerative energy recovery",
        "condition": "wheel_power_demand_kw < 0",
        "action": "Activate motor regeneration, route power to battery",
    },
    {
        "id": "RULE_06",
        "name": "MAX_ACCEL_COMBINED",
        "description": "Maximum acceleration -> ICE + EV simultaneously",
        "condition": (
            f"driver_demand_nm >= 0.85 * "
            f"(MAX_ENGINE_TORQUE + MAX_MOTOR_TORQUE) "
            f"[= 0.85 * ({MAX_ENGINE_TORQUE} + {MAX_MOTOR_TORQUE})]"
        ),
        "action": "Both ICE and Motor at maximum output",
    },
    {
        "id": "RULE_07",
        "name": "URBAN_BATTERY_PRESERVE",
        "description": "Urban route ahead -> preserve battery",
        "condition": (
            f"distance_to_urban_zone_km < 5 and "
            f"current_soc >= SOC_TARGET_URBAN ({SOC_TARGET_URBAN})"
        ),
        "action": "Set SOC planner to PRESERVE mode",
    },
    {
        "id": "RULE_08",
        "name": "ICE_GENERATOR_CHARGING",
        "description": "ICE drives generator -> battery charging",
        "condition": (
            f"engine_is_running and "
            f"soc < SOC_TARGET_URBAN ({SOC_TARGET_URBAN})"
        ),
        "action": "Route excess engine power through generator to battery",
    },
]

# ── Required fields for validation ─────────────────────────────────
_REQUIRED_RULE_FIELDS = {"id", "name", "description", "condition", "action"}


def get_rule_by_id(rule_id: str) -> Optional[Dict[str, str]]:
    """Look up a rule by its unique identifier.

    Args:
        rule_id: The rule ID string (e.g. "RULE_01").

    Returns:
        The rule dict if found, else None.
    """
    for rule in EMS_RULES:
        if rule["id"] == rule_id:
            return rule
    logger.warning("Rule not found: %s", rule_id)
    return None


def get_rule_by_name(name: str) -> Optional[Dict[str, str]]:
    """Look up a rule by its short name.

    Args:
        name: The rule name (e.g. "URBAN_EV_PRIORITY").

    Returns:
        The rule dict if found, else None.
    """
    for rule in EMS_RULES:
        if rule["name"] == name:
            return rule
    logger.warning("Rule not found by name: %s", name)
    return None


def validate_rule_registry() -> bool:
    """Validate that all rules in the registry have required fields.

    Returns:
        True if all rules are valid, False otherwise.
    """
    all_valid = True
    for rule in EMS_RULES:
        missing = _REQUIRED_RULE_FIELDS - set(rule.keys())
        if missing:
            logger.error(
                "Rule %s missing fields: %s",
                rule.get("id", "UNKNOWN"), missing,
            )
            all_valid = False
    return all_valid


def log_rule_fired(rule_id: str, context: Optional[Dict[str, Any]] = None) -> None:
    """Log that a specific rule has fired.

    Args:
        rule_id: The rule ID that was triggered.
        context: Optional dict of context values for the log message.
    """
    rule = get_rule_by_id(rule_id)
    if rule is not None:
        ctx_str = f" | context={context}" if context else ""
        logger.info(
            "EMS RULE FIRED: [%s] %s - %s%s",
            rule["id"], rule["name"], rule["description"], ctx_str,
        )
    else:
        logger.warning("Attempted to log unknown rule: %s", rule_id)


def print_rule_registry() -> None:
    """Print the entire rule registry in a formatted table."""
    print(f"\n{'ID':<10} | {'Name':<28} | {'Description'}")
    print("-" * 80)
    for rule in EMS_RULES:
        print(f"{rule['id']:<10} | {rule['name']:<28} | {rule['description']}")


if __name__ == "__main__":
    print("=" * 80)
    print("  EMS RULES REGISTRY - Phase 3 Supervisory Rules")
    print("=" * 80)

    # Validate
    valid = validate_rule_registry()
    print(f"\nRegistry validation: {'PASS' if valid else 'FAIL'}")
    print(f"Total rules: {len(EMS_RULES)}")

    # Print table
    print_rule_registry()

    # Detailed view
    print("\n\n--- Detailed Rule Specifications ---")
    for rule in EMS_RULES:
        print(f"\n[{rule['id']}] {rule['name']}")
        print(f"  Description : {rule['description']}")
        print(f"  Condition   : {rule['condition']}")
        print(f"  Action      : {rule['action']}")

    # Test lookup
    print("\n\n--- Lookup Tests ---")
    r = get_rule_by_id("RULE_05")
    print(f"By ID 'RULE_05': {r['name'] if r else 'NOT FOUND'}")

    r = get_rule_by_name("MAX_ACCEL_COMBINED")
    print(f"By name 'MAX_ACCEL_COMBINED': {r['id'] if r else 'NOT FOUND'}")

    # Test log
    print("\n--- Rule Fire Logging ---")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log_rule_fired("RULE_01", {"soc": 0.65, "traffic": "urban"})
    log_rule_fired("RULE_05", {"wheel_power_kw": -12.5})

    print("\n" + "=" * 80)
