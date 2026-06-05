"""
traffic_predictor.py — Phase 2: Traffic State Classifier
Author: Antigravity
Date: 2026-06-01

Classifies the current traffic state from raw telemetry using rolling
window statistics.  All thresholds are imported from config.py.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import List

import pandas as pd

from src.config import (
    CONGESTION_SPEED_LOWER_KMH,
    CONGESTION_SPEED_UPPER_KMH,
    STOP_GO_SPEED_MAX_KMH,
    CONGESTION_SPEED_STD_WINDOW,
    CONGESTION_ACCEL_STD_WINDOW,
    CONGESTION_SPEED_STD_THRESHOLD,
    CONGESTION_ACCEL_STD_THRESHOLD,
)

logger = logging.getLogger(__name__)


class TrafficState(Enum):
    """Classification of the prevailing traffic condition."""

    FREE_FLOW = "FREE_FLOW"
    CONGESTED = "CONGESTED"
    STOP_GO = "STOP_GO"


@dataclass
class TrafficFeatures:
    """Rolling telemetry statistics consumed by the classifier.

    Attributes:
        speed_kmh: Instantaneous vehicle speed in km/h.
        speed_std: Rolling standard deviation of speed over the last N samples.
        accel_std: Rolling standard deviation of acceleration over the last N samples.
    """

    speed_kmh: float
    speed_std: float
    accel_std: float


def classify_traffic(features: TrafficFeatures) -> TrafficState:
    """Classifies traffic state from instantaneous and rolling telemetry features.

    Logic (all thresholds from config.py):
    1. If speed_kmh <= STOP_GO_SPEED_MAX_KMH → STOP_GO
    2. Elif CONGESTION_SPEED_LOWER_KMH <= speed_kmh <= CONGESTION_SPEED_UPPER_KMH
       AND (speed_std > CONGESTION_SPEED_STD_THRESHOLD
            OR accel_std > CONGESTION_ACCEL_STD_THRESHOLD) → CONGESTED
    3. Else → FREE_FLOW

    Args:
        features: TrafficFeatures with current and rolling statistics.

    Returns:
        TrafficState enum member.
    """
    if features.speed_kmh <= STOP_GO_SPEED_MAX_KMH:
        logger.debug(
            "Traffic classified STOP_GO: speed=%.1f <= %.1f",
            features.speed_kmh,
            STOP_GO_SPEED_MAX_KMH,
        )
        return TrafficState.STOP_GO

    if (
        CONGESTION_SPEED_LOWER_KMH <= features.speed_kmh <= CONGESTION_SPEED_UPPER_KMH
        and (
            features.speed_std > CONGESTION_SPEED_STD_THRESHOLD
            or features.accel_std > CONGESTION_ACCEL_STD_THRESHOLD
        )
    ):
        logger.debug(
            "Traffic classified CONGESTED: speed=%.1f in [%.1f, %.1f], "
            "speed_std=%.2f, accel_std=%.2f",
            features.speed_kmh,
            CONGESTION_SPEED_LOWER_KMH,
            CONGESTION_SPEED_UPPER_KMH,
            features.speed_std,
            features.accel_std,
        )
        return TrafficState.CONGESTED

    logger.debug(
        "Traffic classified FREE_FLOW: speed=%.1f", features.speed_kmh
    )
    return TrafficState.FREE_FLOW


def compute_traffic_features(telemetry_window: pd.DataFrame) -> TrafficFeatures:
    """Computes TrafficFeatures from a rolling telemetry window DataFrame.

    Expected columns in telemetry_window:
        speed_kmh, acceleration_ms2

    Args:
        telemetry_window: DataFrame of the last N timesteps.

    Returns:
        TrafficFeatures populated from the window.
    """
    current_speed = float(telemetry_window["speed_kmh"].iloc[-1])

    speed_series = telemetry_window["speed_kmh"].tail(CONGESTION_SPEED_STD_WINDOW)
    accel_series = telemetry_window["acceleration_ms2"].tail(CONGESTION_ACCEL_STD_WINDOW)

    speed_std = float(speed_series.std(ddof=0)) if len(speed_series) > 1 else 0.0
    accel_std = float(accel_series.std(ddof=0)) if len(accel_series) > 1 else 0.0

    return TrafficFeatures(
        speed_kmh=current_speed,
        speed_std=speed_std,
        accel_std=accel_std,
    )


@dataclass
class TrafficPrediction:
    predicted_avg_speed_kmh: float
    congestion_probability: float  # 0.0–1.0
    traffic_density: str           # 'LOW', 'MEDIUM', 'HIGH', 'CONGESTED'
    speed_trend: str               # 'ACCELERATING', 'STEADY', 'DECELERATING'


class TrafficPredictor:
    def predict(self, current_speed: float, time_horizon_s: float) -> TrafficPrediction:
        """Returns predicted traffic state over time_horizon_s seconds.

        Heuristically maps speed and trends to density and trend predictions.
        """
        if current_speed < 15.0:
            density = 'CONGESTED'
            prob = 0.9
            trend = 'STEADY'
        elif current_speed < 30.0:
            density = 'HIGH'
            prob = 0.7
            trend = 'ACCELERATING'
        elif current_speed < 60.0:
            density = 'MEDIUM'
            prob = 0.3
            trend = 'STEADY'
        else:
            density = 'LOW'
            prob = 0.05
            trend = 'DECELERATING'

        avg_speed = max(5.0, current_speed + (5.0 if trend == 'ACCELERATING' else -5.0 if trend == 'DECELERATING' else 0.0))

        return TrafficPrediction(
            predicted_avg_speed_kmh=float(avg_speed),
            congestion_probability=float(prob),
            traffic_density=density,
            speed_trend=trend,
        )
