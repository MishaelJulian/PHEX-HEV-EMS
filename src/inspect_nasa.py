"""
Phase 1 — Deep NASA Battery Structure Inspector
================================================
Explores the nested .mat structure across all 4 batteries.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path
from scipy.io import loadmat
import numpy as np
from collections import Counter

MATLAB_DIR = Path(r"C:\Users\misha\Downloads\1.Battery_Uniform_Distribution_Charge_Discharge_DataSet_2Post\Battery_Uniform_Distribution_Charge_Discharge_DataSet_2Post\data\Matlab")

print("=" * 70)
print("  NASA BATTERY DATASET — DEEP STRUCTURE ANALYSIS")
print("=" * 70)

# Analyze all 4 batteries
for batt in ["RW9", "RW10", "RW11", "RW12"]:
    mat_path = MATLAB_DIR / f"{batt}.mat"
    data = loadmat(str(mat_path), simplify_cells=True)
    steps = data["data"]["step"]

    # Step type distribution
    types = [s["type"] for s in steps]
    type_counts = Counter(types)

    print(f"\n{'─'*70}")
    print(f"  Battery: {batt}")
    print(f"  Procedure: {data['data']['procedure']}")
    print(f"  Total steps: {len(steps):,}")
    print(f"  Step types: {dict(type_counts)}")

    # Sample step details (first 3)
    print(f"\n  Sample steps:")
    for i in [0, 1, 2]:
        s = steps[i]
        print(f"    Step {i}: type={s['type']}, comment=\"{s['comment']}\"")
        print(f"      voltage:  [{s['voltage'].min():.3f}, {s['voltage'].max():.3f}] V  ({len(s['voltage'])} points)")
        print(f"      current:  [{s['current'].min():.3f}, {s['current'].max():.3f}] A")
        print(f"      temp:     [{s['temperature'].min():.1f}, {s['temperature'].max():.1f}] C")
        print(f"      duration: {s['relativeTime'][-1]:.1f} sec")

    # Overall ranges (sample 500 steps for speed)
    # Some steps have scalar (0-dim) arrays — filter them out
    sample_idx = np.linspace(0, len(steps) - 1, min(500, len(steps)), dtype=int)
    valid_v = [steps[i]["voltage"].flatten() for i in sample_idx if isinstance(steps[i]["voltage"], np.ndarray) and steps[i]["voltage"].ndim >= 1 and steps[i]["voltage"].size > 1]
    valid_c = [steps[i]["current"].flatten() for i in sample_idx if isinstance(steps[i]["current"], np.ndarray) and steps[i]["current"].ndim >= 1 and steps[i]["current"].size > 1]
    valid_t = [steps[i]["temperature"].flatten() for i in sample_idx if isinstance(steps[i]["temperature"], np.ndarray) and steps[i]["temperature"].ndim >= 1 and steps[i]["temperature"].size > 1]
    all_v = np.concatenate(valid_v) if valid_v else np.array([0])
    all_c = np.concatenate(valid_c) if valid_c else np.array([0])
    all_t = np.concatenate(valid_t) if valid_t else np.array([0])

    print(f"\n  Overall ranges (sampled {len(sample_idx)} steps):")
    print(f"    Voltage:     [{all_v.min():.3f}, {all_v.max():.3f}] V")
    print(f"    Current:     [{all_c.min():.3f}, {all_c.max():.3f}] A")
    print(f"    Temperature: [{all_t.min():.1f}, {all_t.max():.1f}] C")

print(f"\n{'='*70}")
print("  NASA BATTERY DEEP INSPECTION COMPLETE")
print(f"{'='*70}")
