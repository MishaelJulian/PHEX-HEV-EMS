"""
feature_windows.py — Phase 1: Rolling Window Feature Builder
Author: Antigravity
Date: 2026-05-22

Constructs lagged/temporal features for time-series demand forecasting
without lookahead data leakage.
"""

import pandas as pd
import logging

logger = logging.getLogger(__name__)

def create_lags(df: pd.DataFrame, columns: list, n_lags: int) -> pd.DataFrame:
    """
    Creates lagged columns for the specified columns in a DataFrame.
    If 'trip_id' exists in the DataFrame, lags are computed within each trip group
    to prevent data leakage across different trips.
    
    Args:
        df: The source DataFrame.
        columns: List of columns to lag.
        n_lags: Number of past time steps to create lags for.
        
    Returns:
        A new DataFrame with original columns and new lagged columns named '<col>_lag_<step>'.
        Lagged values are filled with backward/forward fill or zero to prevent NaNs.
    """
    df_lagged = df.copy()
    
    # If trip_id exists, group by it; otherwise, shift globally.
    if "trip_id" in df.columns:
        grouped = df_lagged.groupby("trip_id")
        for col in columns:
            if col not in df.columns:
                logger.warning(f"Column '{col}' not found in DataFrame; skipping lag creation.")
                continue
            for lag in range(1, n_lags + 1):
                df_lagged[f"{col}_lag_{lag}"] = grouped[col].shift(lag)
    else:
        for col in columns:
            if col not in df.columns:
                logger.warning(f"Column '{col}' not found in DataFrame; skipping lag creation.")
                continue
            for lag in range(1, n_lags + 1):
                df_lagged[f"{col}_lag_{lag}"] = df_lagged[col].shift(lag)
                
    # Fill NaN values introduced by shift with the initial values (bfill) then 0
    # to avoid NaN issues during ML training and inference.
    lagged_cols = [f"{col}_lag_{lag}" for col in columns for lag in range(1, n_lags + 1) if f"{col}_lag_{lag}" in df_lagged.columns]
    if lagged_cols:
        if "trip_id" in df.columns:
            df_lagged[lagged_cols] = df_lagged.groupby("trip_id")[lagged_cols].bfill().fillna(0.0)
        else:
            df_lagged[lagged_cols] = df_lagged[lagged_cols].bfill().fillna(0.0)
            
    return df_lagged
