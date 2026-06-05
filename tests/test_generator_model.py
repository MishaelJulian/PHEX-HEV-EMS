"""
test_generator_model.py — Phase 3: Generator Model Unit Tests
"""

import pytest
from src.generator_model import (
    GeneratorState, MAX_GENERATOR_OUTPUT_KW,
    MIN_ENGINE_POWER_FOR_GENERATION_KW,
)


class TestGeneratorModel:
    """Tests for the GeneratorState ICE-coupled generator model."""

    def test_below_minimum_no_output(self):
        """Generator should produce zero output below minimum engine power."""
        gen = GeneratorState()
        output = gen.generate_electricity(3.0)
        assert output == 0.0
        assert gen.is_active is False

    def test_at_minimum_activates(self):
        """Generator should activate at minimum engine power."""
        gen = GeneratorState()
        output = gen.generate_electricity(MIN_ENGINE_POWER_FOR_GENERATION_KW)
        assert output > 0.0
        assert gen.is_active is True

    def test_output_at_rated_power(self):
        """At rated power, output should be input * efficiency, capped at max."""
        gen = GeneratorState(generator_efficiency=0.90)
        output = gen.generate_electricity(30.0)
        expected = min(30.0 * 0.90, MAX_GENERATOR_OUTPUT_KW)
        assert output == pytest.approx(expected, rel=1e-6)

    def test_output_clamped_at_max(self):
        """Output must not exceed MAX_GENERATOR_OUTPUT_KW."""
        gen = GeneratorState(generator_efficiency=0.90)
        output = gen.generate_electricity(50.0)  # 50 * 0.9 = 45, but max = 30
        assert output == pytest.approx(MAX_GENERATOR_OUTPUT_KW, rel=1e-6)

    def test_zero_engine_power(self):
        """Zero engine power should yield zero output and inactive state."""
        gen = GeneratorState()
        output = gen.generate_electricity(0.0)
        assert output == 0.0
        assert gen.is_active is False

    def test_efficiency_applied(self):
        """Output should reflect generator efficiency."""
        gen = GeneratorState(generator_efficiency=0.85)
        output = gen.generate_electricity(20.0)
        expected = 20.0 * 0.85
        assert output == pytest.approx(expected, rel=1e-6)

    def test_state_updated(self):
        """Internal state fields should be updated after generation."""
        gen = GeneratorState()
        gen.generate_electricity(15.0)
        assert gen.input_engine_power == 15.0
        assert gen.charging_power > 0.0
