"""
generate_visualisations.py — Phase 2: Visualization Suite
Author: Antigravity
Date: 2026-06-01

Standalone script that runs all four Phase 2 drive scenarios and produces
six publication-quality figures.  Run via:

    python -m src.generate_visualisations

All figures are saved to outputs/visualizations/.
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless rendering
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.simulator import run_scenario
from src.engine_efficiency import efficiency_score
from src.config import ENGINE_PEAK_EFFICIENCY_RPM, PROJECT_ROOT

logger = logging.getLogger(__name__)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "visualizations"

# Attempt to set the seaborn-v0_8 style; fall back gracefully
try:
    plt.style.use("seaborn-v0_8")
except OSError:
    try:
        plt.style.use("seaborn")
    except OSError:
        logger.warning("seaborn style not available, using default matplotlib style")


# ── Colour palettes ──────────────────────────────────────────────
_TRAFFIC_COLOURS = {
    "FREE_FLOW": "#2ecc71",
    "CONGESTED": "#e67e22",
    "STOP_GO": "#e74c3c",
}

_MODE_COLOURS = {
    "EV": "#27ae60",
    "ICE": "#2980b9",
    "HYBRID_ASSIST": "#8e44ad",
    "REGEN": "#f39c12",
    "CHARGE_SUSTAIN": "#e74c3c",
    "CHARGE_DEPLETING": "#c0392b",
}

_SCENARIO_COLOURS = {
    "congestion": "#e74c3c",
    "aggressive": "#e67e22",
    "regen_heavy": "#2ecc71",
    "mixed_route": "#3498db",
}

SCENARIOS = ["congestion", "aggressive", "regen_heavy", "mixed_route"]


def _ensure_output_dir() -> None:
    """Creates the output directory if it does not exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _run_all_scenarios() -> dict[str, pd.DataFrame]:
    """Runs all four scenarios and returns a dict of DataFrames."""
    results: dict[str, pd.DataFrame] = {}
    for name in SCENARIOS:
        logger.info("Running scenario: %s", name)
        results[name] = run_scenario(name)
    return results


def plot_traffic_state_distribution(results: dict[str, pd.DataFrame]) -> None:
    """Stacked bar chart: fraction of steps per TrafficState, one bar per scenario."""
    fig, ax = plt.subplots(figsize=(10, 6))

    traffic_states = list(_TRAFFIC_COLOURS.keys())
    bar_width = 0.6
    x = np.arange(len(SCENARIOS))

    bottoms = np.zeros(len(SCENARIOS))
    for state in traffic_states:
        fractions = []
        for name in SCENARIOS:
            df = results[name]
            count = (df["traffic_state"] == state).sum()
            fractions.append(count / len(df))
        fractions_arr = np.array(fractions)
        ax.bar(x, fractions_arr, bar_width, bottom=bottoms,
               label=state, color=_TRAFFIC_COLOURS[state])
        bottoms += fractions_arr

    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", " ").title() for s in SCENARIOS])
    ax.set_ylabel("Fraction of Steps")
    ax.set_title("Traffic State Distribution Across Scenarios")
    ax.legend(title="Traffic State")
    plt.tight_layout()
    path = OUTPUT_DIR / "traffic_state_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", path)


def plot_ems_mode_timeline(results: dict[str, pd.DataFrame]) -> None:
    """Horizontal timeline: EMSMode colour-coded per step for mixed_route."""
    df = results["mixed_route"]
    fig, ax = plt.subplots(figsize=(14, 4))

    modes = list(_MODE_COLOURS.keys())
    mode_to_y = {m: i for i, m in enumerate(modes)}

    for step, row in df.iterrows():
        mode = row["ems_mode"]
        if mode in mode_to_y:
            ax.barh(mode_to_y[mode], 1, left=row["step"], height=0.8,
                    color=_MODE_COLOURS[mode], edgecolor="none")

    ax.set_yticks(list(mode_to_y.values()))
    ax.set_yticklabels(modes)
    ax.set_xlabel("Timestep")
    ax.set_title("EMS Mode Timeline — Mixed Route Scenario")

    # Add phase annotations
    ax.axvline(x=100, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.axvline(x=200, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.text(50, len(modes) - 0.3, "Urban 1", ha="center", fontsize=9, color="gray")
    ax.text(150, len(modes) - 0.3, "Highway", ha="center", fontsize=9, color="gray")
    ax.text(250, len(modes) - 0.3, "Urban 2", ha="center", fontsize=9, color="gray")

    plt.tight_layout()
    path = OUTPUT_DIR / "ems_mode_timeline.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", path)


def plot_soc_planner_behavior(results: dict[str, pd.DataFrame]) -> None:
    """Line chart: actual SOC vs soc_target over mixed_route steps."""
    df = results["mixed_route"]
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(df["step"], df["soc"], label="Actual SOC", color="#2980b9", linewidth=1.5)
    ax.plot(df["step"], df["soc_target"], label="SOC Target", color="#e74c3c",
            linewidth=1.5, linestyle="--")

    ax.fill_between(df["step"], df["soc"], df["soc_target"],
                     alpha=0.15, color="#e74c3c")

    ax.axvline(x=100, color="gray", linestyle="--", alpha=0.4)
    ax.axvline(x=200, color="gray", linestyle="--", alpha=0.4)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("SOC (fraction)")
    ax.set_title("SOC Planner Behavior — Mixed Route Scenario")
    ax.legend()
    plt.tight_layout()
    path = OUTPUT_DIR / "soc_planner_behavior.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", path)


def plot_regen_energy_recovery(results: dict[str, pd.DataFrame]) -> None:
    """Area chart: cumulative regen_power_kw over regen_heavy scenario."""
    df = results["regen_heavy"]
    fig, ax = plt.subplots(figsize=(12, 5))

    cumulative = df["regen_power_kw"].cumsum()
    ax.fill_between(df["step"], 0, cumulative, alpha=0.4, color="#f39c12")
    ax.plot(df["step"], cumulative, color="#e67e22", linewidth=1.5,
            label="Cumulative Regen Power (kW·steps)")

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Cumulative Regen Power (kW·steps)")
    ax.set_title("Regenerative Energy Recovery — Regen Heavy Scenario")
    ax.legend()
    plt.tight_layout()
    path = OUTPUT_DIR / "regen_energy_recovery.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", path)


def plot_battery_health_trend(results: dict[str, pd.DataFrame]) -> None:
    """Line chart: battery_health_score over all four scenarios."""
    fig, ax = plt.subplots(figsize=(12, 5))

    for name in SCENARIOS:
        df = results[name]
        ax.plot(df["step"], df["battery_health_score"],
                label=name.replace("_", " ").title(),
                color=_SCENARIO_COLOURS[name], linewidth=1.5)

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Battery Health Score")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Battery Health Trend Across All Scenarios")
    ax.legend()
    plt.tight_layout()
    path = OUTPUT_DIR / "battery_health_trend.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", path)


def plot_engine_efficiency_map() -> None:
    """Line chart: efficiency_score(rpm) for RPM 500–5000, annotated with peak."""
    fig, ax = plt.subplots(figsize=(10, 5))

    rpms = np.linspace(500, 5000, 500)
    scores = [efficiency_score(r) for r in rpms]

    ax.plot(rpms, scores, color="#2980b9", linewidth=2)
    ax.axvline(x=ENGINE_PEAK_EFFICIENCY_RPM, color="#e74c3c",
               linestyle="--", alpha=0.7, label=f"Peak = {ENGINE_PEAK_EFFICIENCY_RPM:.0f} RPM")

    peak_score = efficiency_score(ENGINE_PEAK_EFFICIENCY_RPM)
    ax.annotate(
        f"Peak ({ENGINE_PEAK_EFFICIENCY_RPM:.0f} RPM, {peak_score:.2f})",
        xy=(ENGINE_PEAK_EFFICIENCY_RPM, peak_score),
        xytext=(ENGINE_PEAK_EFFICIENCY_RPM + 600, peak_score - 0.15),
        arrowprops=dict(arrowstyle="->", color="#e74c3c"),
        fontsize=10, color="#e74c3c",
    )

    ax.set_xlabel("Engine RPM")
    ax.set_ylabel("Efficiency Score")
    ax.set_title("ICE Efficiency Map (Gaussian Model)")
    ax.set_ylim(-0.05, 1.1)
    ax.legend()
    plt.tight_layout()
    path = OUTPUT_DIR / "engine_efficiency_map.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", path)


def main() -> None:
    """Entry point: run all scenarios and generate all figures."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    _ensure_output_dir()
    logger.info("Starting Phase 2 visualization suite...")

    results = _run_all_scenarios()

    logger.info("Generating figures...")
    plot_traffic_state_distribution(results)
    plot_ems_mode_timeline(results)
    plot_soc_planner_behavior(results)
    plot_regen_energy_recovery(results)
    plot_battery_health_trend(results)
    plot_engine_efficiency_map()

    logger.info("All 6 figures saved to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
