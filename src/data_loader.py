"""
EMS AIML Layer — Data Loader Module
=====================================
Handles scanning, detecting, and loading all dataset formats:
  - .mat  (NASA battery data via scipy.io.loadmat)
  - .csv  (telemetry + behavior data via pandas)
  - .xlsx (Excel fallback via pandas)
  - .json / .parquet (auto-detected)

IMPORTANT: Always scan folders first before loading to avoid FileNotFoundError.
"""

import os
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── LOGGING SETUP ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# SECTION 1: FOLDER SCANNER
# =============================================================================

def scan_dataset_folder(folder_path: str) -> List[Path]:
    """
    Scan a dataset folder and print every file found.
    
    This should ALWAYS be run before attempting to load any data.
    Avoids FileNotFoundError by confirming what actually exists on disk.
    
    Args:
        folder_path: Absolute path to the dataset folder (use raw string on Windows).
    
    Returns:
        List of Path objects for all files found.
    """
    p = Path(folder_path)
    files_found: List[Path] = []

    if not p.exists():
        logger.error(f"❌ Path does not exist: {p}")
        return files_found

    print(f"\n{'='*70}")
    print(f"📁 Scanning: {p}")
    print(f"{'='*70}")

    for f in sorted(p.rglob("*")):
        if f.is_file():
            size_kb = f.stat().st_size / 1024
            print(f"   {f.name:<50} {size_kb:>10.1f} KB  [{f.suffix}]")
            files_found.append(f)

    print(f"\n   → Total files found: {len(files_found)}")
    return files_found


# =============================================================================
# SECTION 2: FORMAT-AWARE LOADERS
# =============================================================================

def detect_and_load(file_path: Path) -> Any:
    """
    Auto-detect file format and load with the appropriate reader.
    
    Supported formats:
        .csv      → pandas.read_csv()
        .xlsx     → pandas.read_excel()
        .mat      → scipy.io.loadmat()
        .json     → pandas.read_json()
        .parquet  → pandas.read_parquet()
    
    Args:
        file_path: Path to the data file.
    
    Returns:
        Loaded data object (DataFrame or dict for .mat files).
    """
    suffix = file_path.suffix.lower()
    logger.info(f"Loading [{suffix}] → {file_path.name}")

    if suffix == ".csv":
        return pd.read_csv(file_path)
    elif suffix == ".xlsx":
        return pd.read_excel(file_path)
    elif suffix == ".mat":
        from scipy.io import loadmat
        return loadmat(str(file_path), simplify_cells=True)
    elif suffix == ".json":
        return pd.read_json(file_path)
    elif suffix == ".parquet":
        return pd.read_parquet(file_path)
    else:
        logger.warning(f"⚠️ Unsupported format: {suffix} — skipping {file_path.name}")
        return None


def load_nasa_battery_data(data_dir: Path) -> Dict[str, Any]:
    """
    Load all NASA battery .mat files from the Matlab subfolder.
    
    The NASA Randomized Battery Usage Dataset contains charge/discharge
    cycles for batteries RW9, RW10, RW11, RW12 at room temperature.
    
    Each .mat file contains structured arrays with:
      - voltage, current, temperature measurements
      - charge/discharge cycle metadata
      - time series per cycle
    
    Args:
        data_dir: Path to the NASA data directory (contains Matlab/ subfolder).
    
    Returns:
        Dict mapping battery name → loaded .mat data dict.
    """
    matlab_dir = data_dir / "Matlab"
    if not matlab_dir.exists():
        logger.error(f"❌ Matlab subfolder not found at: {matlab_dir}")
        return {}

    battery_data: Dict[str, Any] = {}
    mat_files = sorted(matlab_dir.glob("*.mat"))

    logger.info(f"Found {len(mat_files)} .mat files in {matlab_dir}")

    for mat_file in mat_files:
        name = mat_file.stem  # e.g., "RW9", "RW10"
        battery_data[name] = detect_and_load(mat_file)
        logger.info(f"  ✓ Loaded {name} — top-level keys: {[k for k in battery_data[name].keys() if not k.startswith('__')]}")

    return battery_data


def load_telemetry_data(data_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    Load vehicle telemetry CSV files (train + test motion data).
    
    Args:
        data_dir: Path to the telemetry archive folder.
    
    Returns:
        Dict with 'train' and 'test' DataFrames.
    """
    telemetry: Dict[str, pd.DataFrame] = {}

    for csv_file in sorted(data_dir.glob("*.csv")):
        key = "train" if "train" in csv_file.stem.lower() else "test"
        telemetry[key] = detect_and_load(csv_file)
        logger.info(f"  ✓ Loaded {key}: {telemetry[key].shape}")

    return telemetry


def load_behavior_data(data_dir: Path) -> Optional[pd.DataFrame]:
    """
    Load the Driver Behavior CSV dataset.
    
    Args:
        data_dir: Path to the behavior archive folder.
    
    Returns:
        DataFrame with driver behavior data.
    """
    csv_files = list(data_dir.glob("*.csv"))
    if not csv_files:
        logger.error(f"❌ No CSV files found in {data_dir}")
        return None

    df = detect_and_load(csv_files[0])
    logger.info(f"  ✓ Loaded behavior data: {df.shape}")
    return df


# =============================================================================
# SECTION 3: DATASET INSPECTOR
# =============================================================================

def inspect_dataframe(df: pd.DataFrame, name: str) -> None:
    """
    Full engineering inspection of a DataFrame.
    
    Displays: head, shape, columns, dtypes, nulls, unique categoricals,
    and basic statistics — everything needed for Phase 1 assessment.
    
    Args:
        df:   The DataFrame to inspect.
        name: Human-readable name for display headers.
    """
    print(f"\n{'#'*70}")
    print(f"# DATASET INSPECTION: {name}")
    print(f"{'#'*70}")

    # Shape
    print(f"\n📐 Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # Columns
    print(f"\n📋 Columns ({len(df.columns)}):")
    for i, col in enumerate(df.columns, 1):
        print(f"   {i:>3}. {col}")

    # Data types
    print(f"\n🔢 Data Types:")
    for dtype, count in df.dtypes.value_counts().items():
        print(f"   {str(dtype):<20} → {count} columns")

    # Head
    print(f"\n📊 First 5 rows:")
    print(df.head().to_string())

    # Null values
    null_counts = df.isnull().sum()
    total_nulls = null_counts.sum()
    print(f"\n🕳️  Null Values (total: {total_nulls:,}):")
    if total_nulls > 0:
        for col in null_counts[null_counts > 0].index:
            pct = null_counts[col] / len(df) * 100
            print(f"   {col:<30} → {null_counts[col]:>8,} nulls ({pct:.1f}%)")
    else:
        print("   ✅ No null values found!")

    # Categorical columns
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    if len(cat_cols) > 0:
        print(f"\n🏷️  Categorical Columns:")
        for col in cat_cols:
            unique_vals = df[col].nunique()
            print(f"   {col:<30} → {unique_vals} unique values")
            if unique_vals <= 20:
                print(f"      Values: {df[col].unique().tolist()}")

    # Numeric summary
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(f"\n📈 Numeric Summary:")
        print(df[numeric_cols].describe().round(3).to_string())

    # Sample rows
    print(f"\n🎲 Random Sample (3 rows):")
    print(df.sample(min(3, len(df)), random_state=42).to_string())

    print(f"\n{'─'*70}\n")


def inspect_mat_structure(mat_data: dict, battery_name: str) -> None:
    """
    Inspect the internal structure of a loaded NASA .mat file.
    
    NASA .mat files have deeply nested structures — this function
    recursively explores them to understand what data is available.
    
    Args:
        mat_data:     Dict returned by scipy.io.loadmat().
        battery_name: Name of the battery (e.g., 'RW9').
    """
    print(f"\n{'#'*70}")
    print(f"# NASA BATTERY INSPECTION: {battery_name}")
    print(f"{'#'*70}")

    # Filter out MATLAB metadata keys
    data_keys = [k for k in mat_data.keys() if not k.startswith("__")]
    print(f"\n🔑 Top-level data keys: {data_keys}")

    for key in data_keys:
        obj = mat_data[key]
        print(f"\n   Key: '{key}'")
        print(f"   Type: {type(obj).__name__}")

        if isinstance(obj, np.ndarray):
            print(f"   Shape: {obj.shape}")
            print(f"   Dtype: {obj.dtype}")

            # If it's a structured array, show field names
            if obj.dtype.names:
                print(f"   Fields: {obj.dtype.names}")

            # If it's a dict-like (recarray), try to explore
            if obj.ndim == 0:
                # Scalar structured array — unwrap it
                item = obj.item()
                if isinstance(item, tuple) and obj.dtype.names:
                    for field_name in obj.dtype.names:
                        field_val = obj[field_name]
                        if isinstance(field_val, np.ndarray):
                            print(f"     → {field_name}: ndarray shape={field_val.shape}, dtype={field_val.dtype}")
                        else:
                            print(f"     → {field_name}: {type(field_val).__name__}")
            elif obj.size > 0 and obj.ndim >= 1:
                # Show a small sample
                flat = obj.flatten()
                if flat.size <= 10:
                    print(f"   Values: {flat}")
                else:
                    print(f"   First 5: {flat[:5]}")
                    print(f"   Last 5:  {flat[-5:]}")

        elif isinstance(obj, dict):
            print(f"   Sub-keys: {list(obj.keys())[:20]}")

        elif isinstance(obj, (str, int, float)):
            print(f"   Value: {obj}")

        else:
            print(f"   (Complex type — {type(obj).__name__})")

    print(f"\n{'─'*70}\n")


# =============================================================================
# SECTION 4: RUN SCANNER (execute this first!)
# =============================================================================

if __name__ == "__main__":
    from config import (
        NASA_DATA_DIR,
        TELEMETRY_DATA_DIR,
        BEHAVIOR_DATA_DIR,
    )

    print("\n" + "=" * 70)
    print("  EMS AIML — PHASE 1: DATASET SCAN & INSPECTION")
    print("=" * 70)

    # Step 1: Scan all folders
    print("\n\n" + "▓" * 70)
    print("  STEP 1: SCANNING ALL DATASET FOLDERS")
    print("▓" * 70)

    nasa_files = scan_dataset_folder(str(NASA_DATA_DIR))
    telemetry_files = scan_dataset_folder(str(TELEMETRY_DATA_DIR))
    behavior_files = scan_dataset_folder(str(BEHAVIOR_DATA_DIR))

    # Step 2: Load all datasets
    print("\n\n" + "▓" * 70)
    print("  STEP 2: LOADING ALL DATASETS")
    print("▓" * 70)

    nasa_data = load_nasa_battery_data(NASA_DATA_DIR)
    telemetry_data = load_telemetry_data(TELEMETRY_DATA_DIR)
    behavior_df = load_behavior_data(BEHAVIOR_DATA_DIR)

    # Step 3: Inspect each dataset
    print("\n\n" + "▓" * 70)
    print("  STEP 3: FULL DATASET INSPECTION")
    print("▓" * 70)

    # Inspect telemetry
    for split_name, df in telemetry_data.items():
        inspect_dataframe(df, f"Vehicle Telemetry — {split_name}")

    # Inspect behavior
    if behavior_df is not None:
        inspect_dataframe(behavior_df, "Driver Behavior")

    # Inspect NASA battery
    for batt_name, batt_data in nasa_data.items():
        inspect_mat_structure(batt_data, batt_name)

    print("\n" + "=" * 70)
    print("  ✅ PHASE 1 SCAN COMPLETE")
    print("=" * 70)
