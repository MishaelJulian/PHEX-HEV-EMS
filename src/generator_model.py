"""
generator_model.py — Phase 3: ICE-Coupled Generator Model
Author: Antigravity
Date: 2026-06-03

Models the on-board generator that is mechanically coupled to the ICE
and charges the battery.  In a series-parallel PHEV, the ICE can either
drive the wheels directly or spin a generator to produce electricity,
which then either charges the battery or directly powers the motor.

Dependencies:
    - None (standalone physics module)

Outputs:
    - Generator activation state
    - Charging power output (kW)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# ── Generator Constants ────────────────────────────────────────────
MAX_GENERATOR_OUTPUT_KW: float = 30.0       # Generator rated output
MIN_ENGINE_POWER_FOR_GENERATION_KW: float = 5.0  # Minimum ICE power before generator engages


@dataclass
class GeneratorState:
    """Represents the operating state of the ICE-coupled generator.

    The generator converts mechanical power from the ICE into electrical
    power for the battery bus.  It only activates when the ICE is
    providing sufficient power above a minimum threshold.

    Attributes:
        input_engine_power: Mechanical power input from ICE in kW.
        generator_efficiency: Conversion efficiency (default 90%).
        charging_power: Electrical output delivered to battery bus in kW.
        is_active: Whether generator is currently coupled to ICE.
    """

    input_engine_power: float = 0.0
    generator_efficiency: float = 0.90
    charging_power: float = 0.0
    is_active: bool = False

    def generate_electricity(self, engine_power_kw: float) -> float:
        """Generate electricity from ICE mechanical power.

        Sets input_engine_power to the provided value.  If the engine is
        too lightly loaded (below MIN_ENGINE_POWER_FOR_GENERATION_KW),
        the generator deactivates and returns 0.0.

        Args:
            engine_power_kw: Mechanical power available from ICE in kW.

        Returns:
            Charging power in kW delivered to the battery bus.
        """
        self.input_engine_power = engine_power_kw

        if engine_power_kw < MIN_ENGINE_POWER_FOR_GENERATION_KW:
            self.is_active = False
            self.charging_power = 0.0
            logger.debug(
                "Generator inactive: engine power %.1f kW < minimum %.1f kW.",
                engine_power_kw, MIN_ENGINE_POWER_FOR_GENERATION_KW,
            )
            return 0.0

        self.is_active = True
        return self.calculate_charging_power()

    def calculate_charging_power(self) -> float:
        """Compute electrical output from input engine power and efficiency.

        Formula:
            charging_power = min(input_engine_power * efficiency, MAX_OUTPUT)

        Updates self.charging_power in place.

        Returns:
            Charging power in kW.
        """
        raw_output = self.input_engine_power * self.generator_efficiency
        self.charging_power = min(raw_output, MAX_GENERATOR_OUTPUT_KW)

        logger.debug(
            "Generator: input=%.1f kW, η=%.2f, raw=%.1f kW, clamped=%.1f kW",
            self.input_engine_power, self.generator_efficiency,
            raw_output, self.charging_power,
        )

        return self.charging_power


if __name__ == "__main__":
    print("=" * 60)
    print("  GENERATOR MODEL — Output vs Engine Power Demo")
    print("=" * 60)

    gen = GeneratorState()
    engine_powers = [0.0, 3.0, 10.0, 20.0, 30.0, 40.0, 50.0]

    print(f"\nGenerator efficiency: {gen.generator_efficiency:.0%}")
    print(f"Max output: {MAX_GENERATOR_OUTPUT_KW} kW")
    print(f"Min engine power: {MIN_ENGINE_POWER_FOR_GENERATION_KW} kW")
    print()
    print(f"{'Engine kW':>10} | {'Active':>6} | {'Charging kW':>11}")
    print("-" * 35)

    for ep in engine_powers:
        output = gen.generate_electricity(ep)
        print(f"{ep:10.1f} | {str(gen.is_active):>6} | {output:11.2f}")

    print()
    print("Note: Output capped at 30 kW regardless of input.")
    print("      Generator inactive below 5 kW engine power.")
    print("=" * 60)
