"""
test_traffic_predictor.py — Phase 2: Tests for traffic_predictor module.
"""

import pytest
import pandas as pd
import numpy as np

from src.traffic_predictor import (
    TrafficState,
    TrafficFeatures,
    classify_traffic,
    compute_traffic_features,
)
from src.config import (
    STOP_GO_SPEED_MAX_KMH,
    CONGESTION_SPEED_LOWER_KMH,
    CONGESTION_SPEED_STD_THRESHOLD,
    CONGESTION_ACCEL_STD_THRESHOLD,
)


class TestClassifyTraffic:
    """Tests for classify_traffic()."""

    def test_low_speed_returns_stop_go(self) -> None:
        """speed=5 → STOP_GO."""
        features = TrafficFeatures(speed_kmh=5.0, speed_std=2.0, accel_std=1.0)
        assert classify_traffic(features) == TrafficState.STOP_GO

    def test_mid_speed_high_std_returns_congested(self) -> None:
        """speed=55, high std → CONGESTED."""
        features = TrafficFeatures(speed_kmh=55.0, speed_std=8.0, accel_std=2.0)
        assert classify_traffic(features) == TrafficState.CONGESTED

    def test_mid_speed_low_std_returns_free_flow(self) -> None:
        """speed=55, low std → FREE_FLOW."""
        features = TrafficFeatures(speed_kmh=55.0, speed_std=2.0, accel_std=0.5)
        assert classify_traffic(features) == TrafficState.FREE_FLOW

    def test_high_speed_returns_free_flow(self) -> None:
        """speed=100 → FREE_FLOW."""
        features = TrafficFeatures(speed_kmh=100.0, speed_std=3.0, accel_std=1.0)
        assert classify_traffic(features) == TrafficState.FREE_FLOW

    def test_boundary_stop_go_speed(self) -> None:
        """Boundary: speed exactly at STOP_GO_SPEED_MAX_KMH → STOP_GO."""
        features = TrafficFeatures(
            speed_kmh=STOP_GO_SPEED_MAX_KMH, speed_std=5.0, accel_std=2.0,
        )
        assert classify_traffic(features) == TrafficState.STOP_GO

    def test_boundary_congestion_lower_with_high_accel_std(self) -> None:
        """Boundary: speed at CONGESTION_SPEED_LOWER_KMH with high accel_std → CONGESTED."""
        features = TrafficFeatures(
            speed_kmh=CONGESTION_SPEED_LOWER_KMH,
            speed_std=1.0,
            accel_std=CONGESTION_ACCEL_STD_THRESHOLD + 0.1,
        )
        assert classify_traffic(features) == TrafficState.CONGESTED

    def test_accepts_traffic_features_dataclass(self) -> None:
        """classify_traffic accepts TrafficFeatures dataclass without error."""
        features = TrafficFeatures(speed_kmh=70.0, speed_std=3.0, accel_std=1.0)
        result = classify_traffic(features)
        assert isinstance(result, TrafficState)


class TestComputeTrafficFeatures:
    """Tests for compute_traffic_features()."""

    def test_correct_std_from_10_row_dataframe(self) -> None:
        """compute_traffic_features: correct std from 10-row DataFrame."""
        rng = np.random.default_rng(42)
        speeds = rng.uniform(30, 70, size=10)
        accels = rng.uniform(-2, 2, size=10)
        df = pd.DataFrame({
            "speed_kmh": speeds,
            "acceleration_ms2": accels,
        })

        features = compute_traffic_features(df)

        assert features.speed_kmh == pytest.approx(speeds[-1])
        expected_speed_std = float(pd.Series(speeds).std(ddof=0))
        assert features.speed_std == pytest.approx(expected_speed_std, rel=1e-3)
        expected_accel_std = float(pd.Series(accels).std(ddof=0))
        assert features.accel_std == pytest.approx(expected_accel_std, rel=1e-3)
