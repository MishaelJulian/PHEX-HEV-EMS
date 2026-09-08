"""
battery_health.py — Phase 2: Supervisory Battery Health Monitor
Author: Antigravity
Date: 2026-06-01

Lightweight supervisory battery health estimate.  No electrochemistry —
uses simple heuristics (temperature, cycle count, regen frequency) to
produce a health score that informs mode selection.
"""

import logging
from dataclasses import dataclass, field

from src.config import (
    BATTERY_TEMP_WARN_C,
    BATTERY_TEMP_CRITICAL_C,
    BATTERY_REGEN_FREQ_WARN,
)

logger = logging.getLogger(__name__)


@dataclass
class BatteryHealthState:
    soc_history: list[float] = field(default_factory=list)
    temperature_c: float = 25.0
    cycle_count: int = 0
    regen_event_count: int = 0
    total_steps: int = 0
    # Internal tracking for cycle detection
    _last_crossing_direction: int = field(default=0, repr=False)

    def update(
        self,
        soc: float,
        temperature_c: float,
        regen_occurred: bool,
    ) -> None:
        self.soc_history.append(soc)
        self.temperature_c = temperature_c
        self.total_steps += 1
        if regen_occurred:
            self.regen_event_count += 1
        # Cycle detection: SOC crossing 0.5 in opposite direction
        if len(self.soc_history) >= 2:
            prev_soc = self.soc_history[-2]
            if prev_soc < 0.5 <= soc:
                # Rising through 0.5
                if self._last_crossing_direction == -1:
                    self.cycle_count += 1
                    logger.debug("Cycle count incremented to %d (rising)", self.cycle_count)
                self._last_crossing_direction = 1
            elif prev_soc >= 0.5 > soc:
                # Falling through 0.5
                if self._last_crossing_direction == 1:
                    self.cycle_count += 1
                    logger.debug("Cycle count incremented to %d (falling)", self.cycle_count)
                self._last_crossing_direction = -1

    def health_score(self) -> float:
        # Temperature factor
        if self.temperature_c <= BATTERY_TEMP_WARN_C:
            temp_factor = 1.0
        elif self.temperature_c >= BATTERY_TEMP_CRITICAL_C:
            temp_factor = 0.0
        else:
            temp_factor = (
                (BATTERY_TEMP_CRITICAL_C - self.temperature_c)
                / (BATTERY_TEMP_CRITICAL_C - BATTERY_TEMP_WARN_C)
            )
        # Cycle factor
        cycle_penalty = self.cycle_count * 0.001
        cycle_factor = max(0.5, 1.0 - cycle_penalty)
        # Regen frequency factor
        regen_factor = 1.0
        if self.total_steps > 0:
            regen_fraction = self.regen_event_count / self.total_steps
            if regen_fraction > BATTERY_REGEN_FREQ_WARN:
                regen_factor = 0.95  # -0.05 penalty
        # Combined score
        score = temp_factor * cycle_factor * regen_factor
        return max(0.0, min(1.0, score))
