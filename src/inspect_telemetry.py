"""
Phase 1 — Telemetry Dataset Deep Inspector
============================================
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from pathlib import Path

TELEMETRY_DIR = Path(r"C:\Users\misha\Downloads\archive")

print("=" * 70)
print("  VEHICLE TELEMETRY DATASET — INSPECTION")
print("=" * 70)

for csv_file in sorted(TELEMETRY_DIR.glob("*.csv")):
    df = pd.read_csv(csv_file)
    name = csv_file.stem

    print(f"\n{'#'*70}")
    print(f"# {name}")
    print(f"{'#'*70}")

    print(f"\nShape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nData Types:")
    print(df.dtypes.to_string())
    print(f"\nFirst 5 rows:")
    print(df.head().to_string())
    print(f"\nNull values:")
    print(df.isnull().sum().to_string())
    print(f"\nNumeric summary:")
    print(df.describe().round(3).to_string())

    cat_cols = df.select_dtypes(include=["object"]).columns
    if len(cat_cols) > 0:
        print(f"\nCategorical columns:")
        for col in cat_cols:
            print(f"  {col}: {df[col].nunique()} unique -> {df[col].unique().tolist()}")

print(f"\n{'='*70}")
print("  TELEMETRY INSPECTION COMPLETE")
print(f"{'='*70}")
