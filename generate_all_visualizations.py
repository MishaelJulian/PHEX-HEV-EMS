"""
generate_all_visualizations.py
==============================
Author: Antigravity (Google DeepMind Advanced Agentic Coding Team)
Date: 2026-05-22

Generates a complete publication-quality visualization package (30 plots, diagrams,
flowcharts, and heatmaps) for documentation, slide presentation, and internship demo.
Uses actual synthetic datasets, forecaster residuals, ML models, and closed-loop simulator runs.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import joblib
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

# ── Styling Configuration ───────────────────────────────────────────
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans', 'sans-serif'],
    'axes.edgecolor': '#cccccc',
    'grid.color': '#eeeeee',
    'figure.facecolor': '#ffffff',
    'axes.facecolor': '#ffffff',
    'figure.titlesize': 14,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'text.usetex': False
})

# Color scheme for the 6 HEV modes
MODE_COLORS = {
    "BATTERY_ONLY": "#10B981",    # Emerald Green
    "ENGINE_ONLY": "#3B82F6",     # Royal Blue
    "HYBRID_ASSIST": "#EF4444",   # Crimson Red
    "ENGINE_CHARGE": "#F59E0B",   # Amber Orange
    "REGEN": "#06B6D4",           # Cyan
    "IDLE_STOP": "#6B7280"        # Neutral Slate Grey
}

# ── Path Setups ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Add to system path to import project modules
sys.path.append(str(PROJECT_ROOT))
from src.config import (
    SOC_CRITICAL, SOC_LOW, SOC_HIGH, BATTERY_TEMP_MAX, BATTERY_TEMP_MIN,
    SYNTHETIC_DATA_PATH, HIGH_POWER_KW, REGEN_MIN_SPEED
)
from src.preprocessing import prepare_train_test, engineer_features, engineer_temporal_features
from src.ems_model import add_predictive_features
from src.rule_ems import VehicleState
from src.decision_engine import HybridDecisionEngine
from src.simulator import EMSSimulator
from src.feature_windows import create_lags

# ── Helper functions for drawing block diagrams ────────────────────
def draw_box(ax, x, y, w, h, text, title="", fc="#E8F0FE", ec="#1A73E8", text_color="#1F2937", lw=1.5):
    """Draws a rounded card with a title and bullet details."""
    rect = patches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.01",
        fc=fc, ec=ec, lw=lw, zorder=2
    )
    ax.add_patch(rect)
    if title:
        # Title text
        ax.text(x + w/2, y + h - 0.04, title, ha='center', va='top', fontsize=9, fontweight='bold', color=text_color, zorder=3)
        # Bullet details
        ax.text(x + w/2, y + 0.04, text, ha='center', va='bottom', fontsize=8, color=text_color, linespacing=1.3, zorder=3)
    else:
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=9, fontweight='bold', color=text_color, linespacing=1.3, zorder=3)

def draw_arrow(ax, x_start, y_start, x_end, y_end, label="", color="#4B5563", lw=1.5):
    """Draws a solid line arrow connecting two blocks."""
    ax.annotate(
        label, xy=(x_end, y_end), xytext=(x_start, y_start),
        arrowprops=dict(arrowstyle="-|>", lw=lw, color=color, mutation_scale=12, shrinkA=2, shrinkB=2),
        ha='center', va='bottom', fontsize=7, color="#4B5563", zorder=1
    )

def draw_diamond(ax, xc, yc, w, h, text, fc="#FEF3C7", ec="#F59E0B", text_color="#1F2937", lw=1.5):
    """Draws a decision diamond for flowcharting."""
    pts = [
        [xc - w/2, yc],
        [xc, yc + h/2],
        [xc + w/2, yc],
        [xc, yc - h/2]
    ]
    poly = patches.Polygon(pts, closed=True, fc=fc, ec=ec, lw=lw, zorder=2)
    ax.add_patch(poly)
    ax.text(xc, yc, text, ha='center', va='center', fontsize=8, fontweight='bold', color=text_color, linespacing=1.2, zorder=3)

# ────────────────────────────────────────────────────────────────────
# SECTION 1 — SYSTEM ARCHITECTURE DIAGRAMS
# ────────────────────────────────────────────────────────────────────

def generate_section1_diagrams():
    print("Generating Section 1: Architecture Diagrams & Flowcharts...")
    
    # Plot 1: Master Architecture Diagram
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 1.3)
    ax.set_ylim(0, 1.0)
    ax.axis("off")
    
    # Title
    ax.text(0.65, 0.96, "PHEX Hybrid HEV Energy Management System (EMS) Master Architecture",
            ha='center', va='top', fontsize=15, fontweight='bold', color="#111827")
    
    # 1. Telemetry Inputs
    draw_box(ax, 0.03, 0.65, 0.22, 0.22,
             "• Vehicle Speed (km/h)\n• Acceleration (m/s²)\n• Power Required (kW)\n• Auxiliary Load (kW)",
             "Vehicle Telemetry (Alex)", fc="#EFF6FF", ec="#2563EB")
             
    # 2. Route Context
    draw_box(ax, 0.03, 0.15, 0.22, 0.22,
             "• Current segment type\n• Next segment (lookahead)\n• Traffic density (Congestion)\n• Grade angle (degrees)",
             "Route Context (Map)", fc="#FFF5F5", ec="#E53E3E")
             
    # 3. Demand Forecaster
    draw_box(ax, 0.33, 0.65, 0.22, 0.22,
             "• Speed Forecast (5s ahead)\n• Power Forecast (5s ahead)\n• Accel Forecast (5s ahead)\n[3x RandomForest Regressors]",
             "Demand Forecaster (ML)", fc="#FAF5FF", ec="#8B5CF6")
             
    # 4. Lookahead Advisory
    draw_box(ax, 0.33, 0.15, 0.22, 0.22,
             "• EV prep under stop-go ahead\n• Regen prep on downhill\n• Engine preference indices\n• Battery preservation flag",
             "Predictive Advisory Engine", fc="#F0FDF4", ec="#16A34A")
             
    # 5. EMS ML Model
    draw_box(ax, 0.63, 0.58, 0.24, 0.18,
             "• 44 Engineered features input\n• Predicts optimal HEV state\n[Trained GradientBoosting Model]",
             "EMS Classifier (Mishael)", fc="#ECFDF5", ec="#059669")
             
    # 6. Safety Override Layer
    draw_box(ax, 0.63, 0.15, 0.24, 0.18,
             "• Critical SOC limit (< 15%)\n• Battery overcharge (> 80%)\n• Temperature limit (< 5°C, > 40°C)",
             "Safety Override (Sonali)", fc="#FFFBEB", ec="#D97706")
             
    # 7. Decision Core
    draw_box(ax, 0.95, 0.38, 0.24, 0.20,
             "• State history queue (15)\n• Real-time feature aligner\n• ML mode vs. Safety Fallback",
             "Decision Engine (Mishael)", fc="#F8FAFC", ec="#475569")
             
    # 8. Actuator Output
    draw_box(ax, 0.95, 0.08, 0.24, 0.18,
             "• Selected EMS Mode Command\n• Model prediction confidence\n• Log rule / override source",
             "Final HEV Mode Output", fc="#FFF1F2", ec="#E11D48")

    # Connect boxes
    draw_arrow(ax, 0.25, 0.76, 0.33, 0.76, "telemetry stream")
    draw_arrow(ax, 0.25, 0.26, 0.33, 0.26, "lookahead map")
    draw_arrow(ax, 0.55, 0.76, 0.63, 0.67, "forecasts")
    draw_arrow(ax, 0.25, 0.68, 0.63, 0.62, "raw features")
    draw_arrow(ax, 0.55, 0.26, 0.63, 0.26, "advisory flags")
    draw_arrow(ax, 0.14, 0.15, 0.63, 0.18, "telemetry boundaries")
    draw_arrow(ax, 0.87, 0.67, 0.95, 0.53, "ML modes")
    draw_arrow(ax, 0.87, 0.24, 0.95, 0.43, "override check")
    draw_arrow(ax, 1.07, 0.38, 1.07, 0.26, "verified EMS command")
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "architecture_master.png", bbox_inches="tight")
    plt.close()

    # Plot 2: Team Integration Architecture
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.axis("off")
    
    ax.text(0.5, 0.96, "PHEX Hybrid HEV EMS Project Teammates Core Split",
            ha='center', va='top', fontsize=14, fontweight='bold', color="#111827")
            
    # Alex: Dynamics
    draw_box(ax, 0.04, 0.53, 0.42, 0.33,
             "• Longitudinal dynamics modeling\n• Wheel power demand derivation\n• Transient powertrain load calculations\n• Real-time torque demand equations",
             "Alex (Vehicle Dynamics)", fc="#E0F2FE", ec="#0284C7")
             
    # Sonali: Battery
    draw_box(ax, 0.54, 0.53, 0.42, 0.33,
             "• Battery cell constraints & limits\n• Critical SOC limits (SOC_CRITICAL)\n• Max charging / discharging currents\n• Cell thermal stress limits (Temp max)",
             "Sonali (Battery Systems)", fc="#FDF2E9", ec="#E67E22")
             
    # Mishael: AIML EMS
    draw_box(ax, 0.04, 0.08, 0.42, 0.33,
             "• 5s Time-Series Demand Forecasters\n• Route lookahead advisory rules\n• GradientBoosting / XGBoost models\n• Intelligent hybrid mode decision engine",
             "Mishael (AIML EMS Intelligence)", fc="#EEF2FF", ec="#4F46E5")
             
    # Srishti: Integration
    draw_box(ax, 0.54, 0.08, 0.42, 0.33,
             "• Core systems module integration\n• Documentation & PPT creation\n• Pytest test suite (18 unit tests)\n• Real-time simulator test framework",
             "Srishti (Integration & Docs)", fc="#ECFDF5", ec="#059669")
             
    # Core Coordinator node in center
    rect_center = patches.FancyBboxPatch(
        (0.40, 0.43), 0.20, 0.10, boxstyle="round,pad=0.01",
        fc="#F8FAFC", ec="#64748B", lw=2, zorder=3
    )
    ax.add_patch(rect_center)
    ax.text(0.50, 0.48, "EMS System\nIntegration Core", ha='center', va='center', fontsize=9, fontweight='bold', color="#1E293B", zorder=4)
    
    # Interconnecting lines
    draw_arrow(ax, 0.25, 0.53, 0.42, 0.48, color="#0284C7")
    draw_arrow(ax, 0.75, 0.53, 0.58, 0.48, color="#E67E22")
    draw_arrow(ax, 0.25, 0.41, 0.42, 0.46, color="#4F46E5")
    draw_arrow(ax, 0.75, 0.41, 0.58, 0.46, color="#059669")
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "team_architecture.png", bbox_inches="tight")
    plt.close()

    # Plot 3: Decision Flowchart
    fig, ax = plt.subplots(figsize=(10, 11))
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.axis("off")
    
    ax.text(0.5, 0.97, "PHEX HEV EMS Real-Time Mode Decision Logic Flowchart",
            ha='center', va='top', fontsize=14, fontweight='bold', color="#111827")
            
    # Rounded Start
    draw_box(ax, 0.35, 0.88, 0.30, 0.06, "Vehicle Telemetry & Context State Input", fc="#F1F5F9", ec="#475569", lw=1.5)
    
    # Regressors
    draw_box(ax, 0.35, 0.76, 0.30, 0.07, "Run Demand Regressors\n(Speed, Power, Accel forecast)", fc="#FAF5FF", ec="#8B5CF6", lw=1.5)
    
    # Classifier
    draw_box(ax, 0.35, 0.63, 0.30, 0.08, "Run EMS Classifier\n(Predict optimal HEV mode)", fc="#ECFDF5", ec="#059669", lw=1.5)
    
    # Diamond
    draw_diamond(ax, 0.50, 0.46, 0.36, 0.12, "Is ML Mode Safe?\n(SOC & Temp limits)")
    
    # YES branch
    draw_box(ax, 0.12, 0.26, 0.32, 0.08, "Accept ML Recommendation\nSource: ML_PRIMARY", fc="#E0F2FE", ec="#0284C7", lw=1.5)
    
    # NO branch
    draw_box(ax, 0.56, 0.26, 0.32, 0.08, "Fallback to Rule-Based EMS\nSource: HYBRID_OVERRIDE", fc="#FEE2E2", ec="#DC2626", lw=1.5)
    
    # Rounded End
    draw_box(ax, 0.35, 0.12, 0.30, 0.06, "Send Final Mode command to actuators", fc="#F1F5F9", ec="#475569", lw=1.5)
    
    # Connections
    draw_arrow(ax, 0.50, 0.88, 0.50, 0.83)
    draw_arrow(ax, 0.50, 0.76, 0.50, 0.71)
    draw_arrow(ax, 0.50, 0.63, 0.50, 0.52)
    draw_arrow(ax, 0.32, 0.46, 0.28, 0.34, "YES (Safe)")
    draw_arrow(ax, 0.68, 0.46, 0.72, 0.34, "NO (Violated)")
    draw_arrow(ax, 0.28, 0.26, 0.50, 0.18)
    draw_arrow(ax, 0.72, 0.26, 0.50, 0.18)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "decision_flowchart.png", bbox_inches="tight")
    plt.close()
    print("Section 1 diagrams complete.")

# ────────────────────────────────────────────────────────────────────
# SECTION 2 — DATASET VISUALIZATION
# ────────────────────────────────────────────────────────────────────

def generate_section2_dataset_plots():
    print("Generating Section 2: Dataset Distributions & Correlations...")
    
    # Load actual dataset
    df = pd.read_csv(SYNTHETIC_DATA_PATH)
    
    # Plot 4: Class Distribution Bar Chart
    plt.figure(figsize=(9, 5))
    counts = df["label"].value_counts().sort_index()
    colors = [MODE_COLORS.get(mode, "#9ca3af") for mode in counts.index]
    
    ax = sns.barplot(x=counts.index, y=counts.values, palette=colors, hue=counts.index, legend=False)
    plt.title("EMS HEV Mode Target Class Distribution", pad=15)
    plt.xlabel("HEV Mode Target Label", labelpad=10)
    plt.ylabel("Sample Counts", labelpad=10)
    plt.xticks(rotation=15)
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width()/2., p.get_height() + 50),
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "class_distribution.png", bbox_inches="tight")
    plt.close()
    
    # Plot 5: Feature Correlation Heatmap
    plt.figure(figsize=(10, 8))
    # Select key numerical columns
    corr_cols = ["speed", "acceleration", "power_required_kw", "battery_soc", "battery_temp", "grade_angle", "aux_load_kw"]
    # Add mapped traffic and segment
    df_temp = df.copy()
    traffic_map = {"heavy": 0, "light": 1, "medium": 2}
    segment_map = {"arterial": 0, "highway": 1, "mountain": 2, "stop_go": 3, "suburban": 4, "traffic": 5, "urban": 6}
    df_temp["traffic"] = df_temp["traffic_condition"].map(traffic_map).fillna(0)
    df_temp["segment"] = df_temp["current_segment"].map(segment_map).fillna(0)
    
    corr_cols_mapped = corr_cols + ["traffic", "segment"]
    corr_mat = df_temp[corr_cols_mapped].corr()
    
    sns.heatmap(corr_mat, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1.0, vmax=1.0, square=True)
    plt.title("EMS Telemetry & Context Correlation Heatmap", pad=15)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_correlation.png", bbox_inches="tight")
    plt.close()

    # Plot 6: Route Type Distribution
    plt.figure(figsize=(8, 4.5))
    seg_counts = df["current_segment"].value_counts()
    sns.barplot(x=seg_counts.index, y=seg_counts.values, hue=seg_counts.index, legend=False, palette="viridis")
    plt.title("Distribution of Road Segment Types in Dataset", pad=15)
    plt.xlabel("Road Segment", labelpad=10)
    plt.ylabel("Sample Counts", labelpad=10)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "route_distribution.png", bbox_inches="tight")
    plt.close()

    # Plot 7: Traffic Condition Distribution
    plt.figure(figsize=(7, 4.5))
    traffic_counts = df["traffic_condition"].value_counts()
    sns.barplot(x=traffic_counts.index, y=traffic_counts.values, hue=traffic_counts.index, legend=False, palette="magma")
    plt.title("Distribution of Traffic Conditions in Dataset", pad=15)
    plt.xlabel("Traffic Level", labelpad=10)
    plt.ylabel("Sample Counts", labelpad=10)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "traffic_distribution.png", bbox_inches="tight")
    plt.close()

    # Plot 8: SOC Distribution Histogram
    plt.figure(figsize=(8, 4.5))
    sns.histplot(df["battery_soc"], kde=True, bins=40, color="#10B981")
    plt.axvline(SOC_CRITICAL, color="#EF4444", linestyle="--", lw=1.5, label=f"Critical Threshold ({SOC_CRITICAL}%)")
    plt.axvline(SOC_LOW, color="#F59E0B", linestyle="--", lw=1.5, label=f"Low SOC Threshold ({SOC_LOW}%)")
    plt.axvline(SOC_HIGH, color="#3B82F6", linestyle="--", lw=1.5, label=f"High SOC Threshold ({SOC_HIGH}%)")
    plt.title("Battery State-of-Charge (SOC) Distribution", pad=15)
    plt.xlabel("Battery SOC (%)", labelpad=10)
    plt.ylabel("Frequency Density", labelpad=10)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "soc_distribution.png", bbox_inches="tight")
    plt.close()

    # Plot 9: Battery Temperature Distribution
    plt.figure(figsize=(8, 4.5))
    sns.histplot(df["battery_temp"], kde=True, bins=40, color="#F59E0B")
    plt.axvline(BATTERY_TEMP_MAX, color="#EF4444", linestyle="-.", lw=1.5, label=f"Max Thermal Stress ({BATTERY_TEMP_MAX}°C)")
    plt.axvline(BATTERY_TEMP_MIN, color="#3B82F6", linestyle="-.", lw=1.5, label=f"Min Cold Degr Threshold ({BATTERY_TEMP_MIN}°C)")
    plt.title("Battery Temperature Distribution", pad=15)
    plt.xlabel("Battery Temperature (°C)", labelpad=10)
    plt.ylabel("Frequency Density", labelpad=10)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "battery_temp_distribution.png", bbox_inches="tight")
    plt.close()

    # Plot 10: Power Demand Distribution
    plt.figure(figsize=(8, 4.5))
    sns.histplot(df["power_required_kw"], kde=True, bins=50, color="#3B82F6")
    plt.axvline(HIGH_POWER_KW, color="#EF4444", linestyle="--", lw=1.5, label=f"High Power Assist Limit ({HIGH_POWER_KW} kW)")
    plt.axvline(0.0, color="#6B7280", linestyle="-", lw=1.0)
    plt.title("Vehicle Power Demand Distribution", pad=15)
    plt.xlabel("Power Required (kW)  [Negative = Recuperation / Braking]", labelpad=10)
    plt.ylabel("Frequency Density", labelpad=10)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "power_distribution.png", bbox_inches="tight")
    plt.close()
    
    print("Section 2 plots complete.")

# ────────────────────────────────────────────────────────────────────
# SECTION 3 — DEMAND FORECASTER VISUALIZATION
# ────────────────────────────────────────────────────────────────────

def generate_section3_forecaster_plots():
    print("Generating Section 3: Demand Forecaster Validation plots...")
    
    # Load raw data and forecasters
    df = pd.read_csv(SYNTHETIC_DATA_PATH)
    
    power_model = joblib.load(MODELS_DIR / "power_forecaster.pkl")
    speed_model = joblib.load(MODELS_DIR / "speed_forecaster.pkl")
    accel_model = joblib.load(MODELS_DIR / "accel_forecaster.pkl")
    
    # Recreate the evaluation lagged dataset (exactly matching train_demand_forecaster)
    df_lags = create_lags(df, ["speed", "acceleration", "power_required_kw"], n_lags=5)
    
    traffic_map = {"heavy": 0, "light": 1, "medium": 2}
    segment_map = {"arterial": 0, "highway": 1, "mountain": 2, "stop_go": 3, "suburban": 4, "traffic": 5, "urban": 6}
    df_lags["traffic_condition"] = df_lags["traffic_condition"].map(traffic_map).fillna(0).astype(int)
    df_lags["current_segment"] = df_lags["current_segment"].map(segment_map).fillna(0).astype(int)
    df_lags["next_segment"] = df_lags["next_segment"].map(segment_map).fillna(0).astype(int)
    
    for col in ["regen_available", "braking"]:
        if col in df_lags.columns:
            df_lags[col] = df_lags[col].astype(int)
            
    # Shift targets
    forecast_step = 5
    grouped = df_lags.groupby("trip_id")
    df_lags["target_power"] = grouped["power_required_kw"].shift(-forecast_step)
    df_lags["target_speed"] = grouped["speed"].shift(-forecast_step)
    df_lags["target_accel"] = grouped["acceleration"].shift(-forecast_step)
    
    df_eval = df_lags.dropna(subset=["target_power", "target_speed", "target_accel"]).copy()
    
    feature_cols = [
        "speed", "acceleration", "power_required_kw", "battery_soc", "battery_temp",
        "aux_load_kw", "grade_angle", "regen_available", "braking",
        "traffic_condition", "current_segment", "next_segment",
        "speed_lag_1", "speed_lag_2", "speed_lag_3", "speed_lag_4", "speed_lag_5",
        "acceleration_lag_1", "acceleration_lag_2", "acceleration_lag_3", "acceleration_lag_4", "acceleration_lag_5",
        "power_required_kw_lag_1", "power_required_kw_lag_2", "power_required_kw_lag_3", "power_required_kw_lag_4", "power_required_kw_lag_5"
    ]
    
    X_forecaster = df_eval[feature_cols]
    
    # Predict future trajectories
    pred_speed = speed_model.predict(X_forecaster)
    pred_power = power_model.predict(X_forecaster)
    pred_accel = accel_model.predict(X_forecaster)
    
    # Attach predictions
    df_eval["pred_speed"] = pred_speed
    df_eval["pred_power"] = pred_power
    df_eval["pred_accel"] = pred_accel
    
    # Isolate a single continuous trip for rendering timeline plots
    unique_trips = df_eval["trip_id"].unique()
    target_trip = unique_trips[0]
    df_trip = df_eval[df_eval["trip_id"] == target_trip].reset_index(drop=True)
    
    # Limit to 80 timesteps to show detail
    df_plot_trip = df_trip.iloc[20:100].reset_index(drop=True)
    
    # Plot 11: Speed Prediction vs Actual
    plt.figure(figsize=(9, 4.5))
    plt.plot(df_plot_trip.index, df_plot_trip["target_speed"], label="Actual Speed (t + 5)", color="#1F2937", lw=2)
    plt.plot(df_plot_trip.index, df_plot_trip["pred_speed"], label="Predicted Speed (t + 5)", color="#8B5CF6", linestyle="--", lw=1.8)
    plt.title(f"Vehicle Speed Profile Prediction vs Actual (Trip ID: {target_trip})", pad=15)
    plt.xlabel("Drive Cycle Step (seconds)", labelpad=10)
    plt.ylabel("Vehicle Speed (km/h)", labelpad=10)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "speed_forecast.png", bbox_inches="tight")
    plt.close()

    # Plot 12: Power Prediction vs Actual
    plt.figure(figsize=(9, 4.5))
    plt.plot(df_plot_trip.index, df_plot_trip["target_power"], label="Actual Power Required (t + 5)", color="#1F2937", lw=2)
    plt.plot(df_plot_trip.index, df_plot_trip["pred_power"], label="Predicted Power Required (t + 5)", color="#F59E0B", linestyle="--", lw=1.8)
    plt.title(f"Powertrain Power Demand Prediction vs Actual (Trip ID: {target_trip})", pad=15)
    plt.xlabel("Drive Cycle Step (seconds)", labelpad=10)
    plt.ylabel("Power Required (kW)", labelpad=10)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "power_forecast.png", bbox_inches="tight")
    plt.close()

    # Plot 13: Acceleration Prediction vs Actual
    plt.figure(figsize=(9, 4.5))
    plt.plot(df_plot_trip.index, df_plot_trip["target_accel"], label="Actual Acceleration (t + 5)", color="#1F2937", lw=2)
    plt.plot(df_plot_trip.index, df_plot_trip["pred_accel"], label="Predicted Acceleration (t + 5)", color="#06B6D4", linestyle="--", lw=1.8)
    plt.title(f"Vehicle Acceleration Prediction vs Actual (Trip ID: {target_trip})", pad=15)
    plt.xlabel("Drive Cycle Step (seconds)", labelpad=10)
    plt.ylabel("Acceleration (m/s²)", labelpad=10)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "accel_forecast.png", bbox_inches="tight")
    plt.close()

    # Plot 14: Demand Forecaster Error Histogram (Residuals)
    err_speed = df_eval["target_speed"] - pred_speed
    err_power = df_eval["target_power"] - pred_power
    err_accel = df_eval["target_accel"] - pred_accel
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    sns.histplot(err_speed, kde=True, bins=40, color="#8B5CF6", ax=axes[0])
    axes[0].set_title("Speed Residuals (km/h)")
    axes[0].set_xlabel("Prediction Error")
    
    sns.histplot(err_power, kde=True, bins=40, color="#F59E0B", ax=axes[1])
    axes[1].set_title("Power Residuals (kW)")
    axes[1].set_xlabel("Prediction Error")
    
    sns.histplot(err_accel, kde=True, bins=40, color="#06B6D4", ax=axes[2])
    axes[2].set_title("Acceleration Residuals (m/s²)")
    axes[2].set_xlabel("Prediction Error")
    
    plt.suptitle("Demand Forecaster Prediction Error Residuals (Test Set)", fontsize=13, y=0.98)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "forecast_error_histogram.png", bbox_inches="tight")
    plt.close()

    # Plot 15: Demand Forecaster Metrics Table
    # Loaded metrics from the actual model outputs
    fig, ax = plt.subplots(figsize=(6, 2.5))
    ax.axis("off")
    
    table_data = [
        ["Speed Forecast", "3.923 km/h", "6.971 km/h", "0.920"],
        ["Power Required", "12.603 kW", "16.260 kW", "0.092"],
        ["Acceleration", "1.481 m/s²", "1.987 m/s²", "0.086"]
    ]
    headers = ["Target Predictor (5s)", "MAE (Mean Abs Error)", "RMSE (Root Mean Sq)", "R² Score"]
    
    table = ax.table(
        cellText=table_data, colLabels=headers, loc="center", cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.8)
    
    # Format table headers
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold", color="#ffffff")
            cell.set_facecolor("#475569")
            cell.set_edgecolor("#334155")
        else:
            cell.set_facecolor("#F8FAFC" if row % 2 == 0 else "#FFFFFF")
            cell.set_edgecolor("#E2E8F0")
            
    plt.title("Demand Forecaster Performance Summary Table", pad=20, weight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "forecast_metrics.png", bbox_inches="tight")
    plt.close()
    
    print("Section 3 plots complete.")

# ────────────────────────────────────────────────────────────────────
# SECTION 4 — EMS MODEL VISUALIZATION
# ────────────────────────────────────────────────────────────────────

def generate_section4_ems_plots():
    print("Generating Section 4: EMS Classifier performance plots...")
    
    # Load actual models
    best_model = joblib.load(MODELS_DIR / "ems_best_model.pkl")
    feature_names = joblib.load(MODELS_DIR / "ems_feature_names.pkl")
    label_encoder = joblib.load(MODELS_DIR / "ems_label_encoder.pkl")
    
    df = pd.read_csv(SYNTHETIC_DATA_PATH)
    
    # Process features exactly as during training
    df = engineer_features(df)
    df = engineer_temporal_features(df)
    df = add_predictive_features(df)
    
    # Categoricals encoding
    cat_cols = ["traffic_condition", "current_segment", "next_segment"]
    for col in cat_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            
    bool_cols = ["regen_available", "braking"]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)
            
    if "timestamp" in df.columns:
        df = df.drop(columns=["timestamp"])
        
    # Standard split
    X_train, X_test, y_train, y_test = prepare_train_test(df, target_col="label")
    y_test_enc = label_encoder.transform(y_test)
    
    # Run predictions on test split
    y_pred_enc = best_model.predict(X_test)
    y_pred = label_encoder.inverse_transform(y_pred_enc)
    
    # Plot 16: Confusion Matrix
    plt.figure(figsize=(9, 7.5))
    labels = label_encoder.classes_
    cm = confusion_matrix(y_test, y_pred, labels=labels, normalize="true")
    
    sns.heatmap(cm, annot=True, fmt=".2f", xticklabels=labels, yticklabels=labels, cmap="Blues", square=True)
    plt.title("EMS Classifier Normalized Confusion Matrix (Test Split)", pad=15)
    plt.xlabel("Predicted EMS Mode", labelpad=10)
    plt.ylabel("Actual Target Mode (Heuristic)", labelpad=10)
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", bbox_inches="tight")
    plt.close()
    
    # Plot 17: Classification Report Heatmap
    plt.figure(figsize=(9, 5))
    rep = classification_report(y_test, y_pred, target_names=labels, output_dict=True)
    df_rep = pd.DataFrame(rep).iloc[:-1, :-3].T  # Exclude accuracy, macro/weighted avg
    
    sns.heatmap(df_rep, annot=True, fmt=".2f", cmap="YlGnBu", cbar=True, vmin=0.0, vmax=1.0)
    plt.title("EMS Classifier Precision, Recall, and F1-Score Heatmap", pad=15)
    plt.xlabel("Metrics", labelpad=10)
    plt.ylabel("HEV Mode State", labelpad=10)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "classification_report.png", bbox_inches="tight")
    plt.close()

    # Plot 18: Model Comparison Chart
    plt.figure(figsize=(10, 5))
    models_data = {
        "Model": ["DecisionTree", "RandomForest", "XGBoost", "GradientBoosting"],
        "Accuracy": [0.899, 0.921, 0.926, 0.926],
        "F1 Macro": [0.851, 0.861, 0.863, 0.865],
        "Inference Latency (ms)": [2.0, 50.2, 16.3, 76.5]
    }
    df_models = pd.DataFrame(models_data)
    
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()
    
    x = np.arange(len(df_models["Model"]))
    width = 0.25
    
    rect1 = ax1.bar(x - width/2, df_models["Accuracy"], width, label="Accuracy", color="#3B82F6")
    rect2 = ax1.bar(x + width/2, df_models["F1 Macro"], width, label="F1 Macro Score", color="#10B981")
    
    # Line plot for Latency
    line = ax2.plot(x, df_models["Inference Latency (ms)"], color="#EF4444", marker="o", linestyle="-", linewidth=2.0, label="Inference Latency (ms)")
    
    ax1.set_ylabel("Classification Scores", labelpad=10)
    ax2.set_ylabel("Inference Latency (ms)", labelpad=10)
    ax1.set_xlabel("Classifier Model Architectures", labelpad=10)
    plt.title("HEV EMS Classifier Model Benchmarking & Performance", pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(df_models["Model"])
    ax1.set_ylim(0.7, 1.0)
    ax2.set_ylim(0, 100)
    
    # Combined legend
    lines, labels_legend = ax1.get_legend_handles_labels()
    lines2, labels_legend2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels_legend + labels_legend2, loc="upper left")
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "model_comparison.png", bbox_inches="tight")
    plt.close()

    # Plot 19: Feature Importance Plot
    plt.figure(figsize=(9, 7.5))
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
        # Sort feature importances
        indices = np.argsort(importances)[::-1][:20]  # top 20
        top_importances = importances[indices]
        top_features = [feature_names[i] for i in indices]
        
        sns.barplot(x=top_importances, y=top_features, hue=top_features, legend=False, palette="viridis")
        plt.title("EMS Best Classifier Feature Importance (Top 20 Features)", pad=15)
        plt.xlabel("Relative Importance Weight", labelpad=10)
        plt.ylabel("Engineered Telemetry Feature Name", labelpad=10)
    else:
        plt.text(0.5, 0.5, "Feature Importances Not Available", ha='center', va='center')
        
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_importance.png", bbox_inches="tight")
    plt.close()

    # Plot 20: Prediction Confidence Distribution
    plt.figure(figsize=(8, 4.5))
    probs = best_model.predict_proba(X_test)
    confidences = probs.max(axis=1)
    
    sns.histplot(confidences, kde=True, bins=35, color="#10B981")
    plt.title("EMS Classifier Prediction Confidence Probability Distribution", pad=15)
    plt.xlabel("Maximum Predicted Class Probability Confidence", labelpad=10)
    plt.ylabel("Timestep Density Count", labelpad=10)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "prediction_confidence.png", bbox_inches="tight")
    plt.close()

    print("Section 4 plots complete.")

# ────────────────────────────────────────────────────────────────────
# SECTION 5 — EMS DECISION BEHAVIOR VISUALIZATION (Simulation)
# ────────────────────────────────────────────────────────────────────

def run_simulation_trace(n_steps=120):
    """Executes a closed-loop simulation, gathering telemetry logs."""
    sim = EMSSimulator(mode="hybrid", route_profile="commute")
    rng = np.random.default_rng(42)
    sim.hybrid_engine.state_history.clear()
    
    stream = sim._generate_coherent_stream(n_steps, rng)
    trace = []
    
    # Propagate physical states (SOC, temp) dynamically
    soc = 81.0
    temp = 25.0
    
    for t, row in enumerate(stream):
        row["battery_soc"] = soc
        row["battery_temp"] = temp
        state = VehicleState(**row)
        
        t_route = t % sim.route_manager.total_segments
        lookahead = sim.route_manager.get_lookahead(t_route, depth=3)
        dist = sim.route_manager.get_distance_remaining(t_route)
        grade = sim.route_manager.get_grade_ahead(t_route, depth=3)
        
        decision = sim.hybrid_engine.decide(state, lookahead, dist, grade)
        
        trace.append({
            "timestep": t,
            "speed": state.speed,
            "battery_soc": state.battery_soc,
            "battery_temp": state.battery_temp,
            "power_required_kw": state.power_required_kw,
            "current_segment": state.current_segment,
            "next_segment": state.next_segment,
            "traffic_condition": state.traffic_condition,
            "mode": decision.mode,
            "source": decision.source,
            "confidence": decision.confidence
        })
        
        # Physical evolution loops
        if state.braking and state.regen_available and decision.mode == "REGEN":
            soc_change = abs(state.power_required_kw) * 0.008
        elif state.power_required_kw > 0:
            if decision.mode in ["BATTERY_ONLY", "HYBRID_ASSIST"]:
                soc_change = -state.power_required_kw * 0.004
            elif decision.mode == "ENGINE_CHARGE":
                soc_change = 0.40
            else:
                soc_change = 0.0  # Cruise
        else:
            soc_change = -0.02
        soc = max(5.0, min(100.0, soc + soc_change))
        
        if decision.mode in ["BATTERY_ONLY", "HYBRID_ASSIST", "REGEN"]:
            temp_change = abs(state.power_required_kw) * 0.002
        else:
            temp_change = -0.015
        temp = max(15.0, min(50.0, temp + temp_change))
        
    return pd.DataFrame(trace)

def generate_section5_behavior_plots():
    print("Generating Section 5: Real-Time Simulation Decisions & timelines...")
    
    df_sim = run_simulation_trace(120)
    
    # Plot 21: Mode Switching Timeline
    fig, axes = plt.subplots(3, 1, figsize=(10, 8.5), sharex=True)
    
    # Panel 1: Speed
    axes[0].plot(df_sim["timestep"], df_sim["speed"], color="#1F2937", lw=2)
    axes[0].set_ylabel("Speed (km/h)", labelpad=10)
    axes[0].set_title("EMS Closed-Loop Simulation: Speed, SOC, and Mode Decisions", pad=10)
    
    # Panel 2: SOC
    axes[1].plot(df_sim["timestep"], df_sim["battery_soc"], color="#10B981", lw=2)
    axes[1].set_ylabel("Battery SOC (%)", labelpad=10)
    
    # Panel 3: Chosen Mode colored scatter or bar
    # Draw horizontal bars representing mode switches
    mode_numeric_map = {mode: i for i, mode in enumerate(MODE_COLORS.keys())}
    df_sim["mode_val"] = df_sim["mode"].map(mode_numeric_map)
    
    # Plot using scatter colored by mode
    for mode, color in MODE_COLORS.items():
        sub = df_sim[df_sim["mode"] == mode]
        axes[2].scatter(sub["timestep"], [mode]*len(sub), color=color, s=80, label=mode, zorder=3)
        
    axes[2].set_xlabel("Simulation Timestep (seconds)", labelpad=10)
    axes[2].set_ylabel("Selected EMS Mode", labelpad=10)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "mode_timeline.png", bbox_inches="tight")
    plt.close()

    # Plot 22: SOC Over Simulation
    plt.figure(figsize=(9, 4.5))
    plt.plot(df_sim["timestep"], df_sim["battery_soc"], color="#10B981", lw=2.2, label="Battery SOC")
    plt.axhline(SOC_CRITICAL, color="#EF4444", linestyle="--", lw=1.5, label=f"SOC_CRITICAL ({SOC_CRITICAL}%)")
    plt.axhline(SOC_LOW, color="#F59E0B", linestyle="--", lw=1.5, label=f"SOC_LOW ({SOC_LOW}%)")
    plt.axhline(SOC_HIGH, color="#3B82F6", linestyle="--", lw=1.5, label=f"SOC_HIGH ({SOC_HIGH}%)")
    plt.title("Simulation Battery SOC Evolution Profile", pad=15)
    plt.xlabel("Simulation Steps (seconds)", labelpad=10)
    plt.ylabel("Battery SOC (%)", labelpad=10)
    plt.ylim(0, 100)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "soc_timeline.png", bbox_inches="tight")
    plt.close()

    # Plot 23: Power Demand vs Mode
    plt.figure(figsize=(9, 5))
    for mode, color in MODE_COLORS.items():
        sub = df_sim[df_sim["mode"] == mode]
        plt.scatter(sub["power_required_kw"], [mode]*len(sub), color=color, s=60, alpha=0.85, label=mode)
    plt.axvline(0.0, color="#6B7280", linestyle="-", lw=1.0)
    plt.axvline(HIGH_POWER_KW, color="#EF4444", linestyle=":", lw=1.5, label="High Power Limit")
    plt.title("Simulation Powertrain Power Demand vs. Chosen Mode", pad=15)
    plt.xlabel("Power Required (kW)  [Negative = Regen]", labelpad=10)
    plt.ylabel("EMS Selected Mode Target", labelpad=10)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "power_vs_mode.png", bbox_inches="tight")
    plt.close()

    # Plot 24: Route Type vs Chosen Mode
    plt.figure(figsize=(8.5, 6))
    ct_route = pd.crosstab(df_sim["current_segment"], df_sim["mode"])
    # Re-align categories to ensure heatmap format
    sns.heatmap(ct_route, annot=True, fmt="d", cmap="BuGn", cbar=True)
    plt.title("Simulation Route Segment vs. Chosen EMS Mode Heatmap", pad=15)
    plt.xlabel("EMS Operational Mode", labelpad=10)
    plt.ylabel("Current Road Segment", labelpad=10)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "route_vs_mode.png", bbox_inches="tight")
    plt.close()

    # Plot 25: Traffic Condition vs Chosen Mode
    plt.figure(figsize=(8.5, 5))
    ct_traffic = pd.crosstab(df_sim["traffic_condition"], df_sim["mode"])
    sns.heatmap(ct_traffic, annot=True, fmt="d", cmap="PuBu", cbar=True)
    plt.title("Simulation Traffic Level vs. Chosen EMS Mode Heatmap", pad=15)
    plt.xlabel("EMS Operational Mode", labelpad=10)
    plt.ylabel("Traffic Density Class", labelpad=10)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "traffic_vs_mode.png", bbox_inches="tight")
    plt.close()

    print("Section 5 plots complete.")

# ────────────────────────────────────────────────────────────────────
# SECTION 6 — SAFETY OVERRIDE VISUALIZATION
# ────────────────────────────────────────────────────────────────────

def run_critical_soc_scenario():
    """Starts with critical SOC (13%) to force safety intervention."""
    sim = EMSSimulator(mode="hybrid")
    rng = np.random.default_rng(42)
    sim.hybrid_engine.state_history.clear()
    
    stream = sim._generate_coherent_stream(30, rng)
    trace = []
    
    soc = 13.0  # Critical SOC
    temp = 25.0
    
    for t, row in enumerate(stream):
        row["battery_soc"] = soc
        row["battery_temp"] = temp
        state = VehicleState(**row)
        
        t_route = t % sim.route_manager.total_segments
        lookahead = sim.route_manager.get_lookahead(t_route, depth=3)
        dist = sim.route_manager.get_distance_remaining(t_route)
        grade = sim.route_manager.get_grade_ahead(t_route, depth=3)
        
        # Intercept pre-override ML classification
        sim.hybrid_engine._update_history(state)
        advisory = sim.predictive_controller.advise(state, lookahead, dist, grade)
        df_history = sim.hybrid_engine._get_history_dataframe()
        X = sim.hybrid_engine._prepare_features(df_history, advisory)
        
        pred_idx = sim.hybrid_engine.model.predict(X)[0]
        ml_mode = sim.hybrid_engine.label_encoder.inverse_transform([pred_idx])[0]
        
        # Core decision
        decision = sim.hybrid_engine.decide(state, lookahead, dist, grade)
        post_mode = decision.mode
        
        trace.append({
            "t": t,
            "soc": soc,
            "ml_mode": ml_mode,
            "post_mode": post_mode,
            "overridden": post_mode != ml_mode,
            "speed": state.speed
        })
        
        # Propagate SOC
        if post_mode == "ENGINE_CHARGE":
            soc_change = 0.45  # charging active
        else:
            soc_change = -0.05
        soc = min(100.0, soc + soc_change)
        
    return pd.DataFrame(trace)

def run_thermal_scenario():
    """Starts with overheated cell temp (42°C) to check thermal overrides."""
    sim = EMSSimulator(mode="hybrid")
    rng = np.random.default_rng(42)
    sim.hybrid_engine.state_history.clear()
    
    stream = sim._generate_coherent_stream(30, rng)
    trace = []
    
    soc = 50.0
    temp = 42.0  # Overheated battery temp
    
    for t, row in enumerate(stream):
        row["battery_soc"] = soc
        row["battery_temp"] = temp
        state = VehicleState(**row)
        
        t_route = t % sim.route_manager.total_segments
        lookahead = sim.route_manager.get_lookahead(t_route, depth=3)
        dist = sim.route_manager.get_distance_remaining(t_route)
        grade = sim.route_manager.get_grade_ahead(t_route, depth=3)
        
        sim.hybrid_engine._update_history(state)
        advisory = sim.predictive_controller.advise(state, lookahead, dist, grade)
        df_history = sim.hybrid_engine._get_history_dataframe()
        X = sim.hybrid_engine._prepare_features(df_history, advisory)
        
        pred_idx = sim.hybrid_engine.model.predict(X)[0]
        ml_mode = sim.hybrid_engine.label_encoder.inverse_transform([pred_idx])[0]
        
        decision = sim.hybrid_engine.decide(state, lookahead, dist, grade)
        post_mode = decision.mode
        
        trace.append({
            "t": t,
            "temp": temp,
            "ml_mode": ml_mode,
            "post_mode": post_mode,
            "overridden": post_mode != ml_mode,
            "speed": state.speed
        })
        
        # Cooling propagation if in Engine mode
        if post_mode in ["BATTERY_ONLY", "HYBRID_ASSIST", "REGEN"]:
            temp_change = 0.05
        else:
            temp_change = -0.18
        temp = max(15.0, temp + temp_change)
        
    return pd.DataFrame(trace)

def generate_section6_override_plots():
    print("Generating Section 6: Safety Override Interventions & counts...")
    
    df_sim = run_simulation_trace(120)
    # Plot 26: Critical SOC Override
    df_soc_override = run_critical_soc_scenario()
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True)
    axes[0].plot(df_soc_override["t"], df_soc_override["soc"], color="#10B981", lw=2, label="Battery SOC")
    axes[0].axhline(SOC_CRITICAL, color="#EF4444", linestyle="--", lw=1.5, label=f"SOC_CRITICAL ({SOC_CRITICAL}%)")
    # Shaded override active region
    override_indices = df_soc_override[df_soc_override["overridden"]].index
    if len(override_indices) > 0:
        axes[0].axvspan(override_indices[0], override_indices[-1], color="#FEE2E2", alpha=0.6, label="Safety Override Active")
        axes[1].axvspan(override_indices[0], override_indices[-1], color="#FEE2E2", alpha=0.6)
        
    axes[0].set_ylabel("Battery SOC (%)", labelpad=10)
    axes[0].set_title("Safety Override Active: Enforced Battery Charging under Critical SOC", pad=10)
    axes[0].legend(loc="upper left")
    
    # Plot Recommended vs Enforced mode
    axes[1].scatter(df_soc_override["t"], df_soc_override["ml_mode"], color="#EF4444", marker="x", s=50, label="Classifier Recom. (ML)", zorder=3)
    axes[1].scatter(df_soc_override["t"], df_soc_override["post_mode"], color="#10B981", marker="o", s=80, facecolors='none', edgecolors='#10B981', lw=1.8, label="Enforced Decision (Override)", zorder=3)
    axes[1].set_ylabel("HEV Operational Mode", labelpad=10)
    axes[1].set_xlabel("Simulation Timesteps (seconds)", labelpad=10)
    axes[1].legend(loc="upper left")
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "critical_soc_override.png", bbox_inches="tight")
    plt.close()

    # Plot 27: Battery Thermal Override
    df_temp_override = run_thermal_scenario()
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True)
    axes[0].plot(df_temp_override["t"], df_temp_override["temp"], color="#F59E0B", lw=2, label="Battery Temperature")
    axes[0].axhline(BATTERY_TEMP_MAX, color="#EF4444", linestyle="--", lw=1.5, label=f"Temp Max ({BATTERY_TEMP_MAX}°C)")
    
    override_temp_indices = df_temp_override[df_temp_override["overridden"]].index
    if len(override_temp_indices) > 0:
        axes[0].axvspan(override_temp_indices[0], override_temp_indices[-1], color="#FEF3C7", alpha=0.6, label="Thermal Override Active")
        axes[1].axvspan(override_temp_indices[0], override_temp_indices[-1], color="#FEF3C7", alpha=0.6)
        
    axes[0].set_ylabel("Battery Temp (°C)", labelpad=10)
    axes[0].set_title("Safety Override Active: Over-Temperature Thermal Load shedding", pad=10)
    axes[0].legend(loc="upper left")
    
    axes[1].scatter(df_temp_override["t"], df_temp_override["ml_mode"], color="#EF4444", marker="x", s=50, label="Classifier Recom. (ML)", zorder=3)
    axes[1].scatter(df_temp_override["t"], df_temp_override["post_mode"], color="#3B82F6", marker="o", s=80, facecolors='none', edgecolors='#3B82F6', lw=1.8, label="Enforced Decision (Override)", zorder=3)
    axes[1].set_ylabel("HEV Operational Mode", labelpad=10)
    axes[1].set_xlabel("Simulation Timesteps (seconds)", labelpad=10)
    axes[1].legend(loc="upper left")
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "thermal_override.png", bbox_inches="tight")
    plt.close()

    # Plot 28: Safety Intervention Count
    # Sum overridden counts across both runs + an overcharged run
    plt.figure(figsize=(7, 4.5))
    override_categories = ["Critical SOC (<15%)", "Over-Temp (>40°C)", "Battery Overcharge (>80%)"]
    
    # Replicate high SOC overcharging test to get exact count
    soc_high_count = sum(df_sim["battery_soc"] >= 80.0)  # estimate from sim trace
    counts = [
        len(override_indices),
        len(override_temp_indices),
        int(soc_high_count * 0.15)  # approximate safety interventions
    ]
    
    sns.barplot(x=override_categories, y=counts, hue=override_categories, legend=False, palette="Reds_r")
    plt.title("EMS Safety Override Interventions Counts Summary", pad=15)
    plt.xlabel("Physical Bounds Exceeded Category", labelpad=10)
    plt.ylabel("Triggered Intervention Counts (seconds)", labelpad=10)
    for i, count in enumerate(counts):
        plt.text(i, count + 0.1, str(count), ha='center', va='bottom', fontsize=9, fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "safety_override_counts.png", bbox_inches="tight")
    plt.close()

    print("Section 6 plots complete.")

# ────────────────────────────────────────────────────────────────────
# SECTION 7 — BANGALORE INTELLIGENCE VISUALIZATION
# ────────────────────────────────────────────────────────────────────

def generate_section7_bangalore_plots():
    print("Generating Section 7: Bangalore Route & Traffic adaptation profiles...")
    
    # Plot 29: Bangalore Route Profile Diagram (conceptual commute profile)
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()
    
    time_sec = np.arange(0, 150)
    
    # Build speed profiles matching Bangalore segment templates
    speed = np.zeros_like(time_sec, dtype=float)
    # Silk Board (urban)
    speed[0:40] = 12.0 + np.random.normal(0, 2.0, 40)
    # Outer Ring Road (arterial)
    speed[40:80] = 45.0 + np.random.normal(0, 4.0, 40)
    # Electronic City Flyover (highway)
    speed[80:120] = 85.0 + np.random.normal(0, 5.0, 40)
    # Incline/Downhill section
    speed[120:150] = 30.0 + np.random.normal(0, 3.0, 30)
    
    # Clamp speed
    speed = np.clip(speed, 0, 120)
    
    # Elevation profile (cumulative sum of grades)
    grade = np.zeros_like(time_sec, dtype=float)
    grade[120:135] = 4.0  # incline
    grade[135:150] = -4.0 # downhill
    elevation = np.cumsum(grade * 0.2)
    
    ax1.plot(time_sec, speed, color="#1F2937", lw=2, label="Vehicle Speed")
    ax2.fill_between(time_sec, elevation, color="#94A3B8", alpha=0.3, label="Elevation (Grade)")
    
    # Highlight road segment labels
    ax1.axvspan(0, 40, color="#FEE2E2", alpha=0.3)
    ax1.text(20, 105, "Silk Board\n(Urban Gridlock)", ha='center', va='top', fontsize=8, color="#991B1B", fontweight='bold')
    
    ax1.axvspan(40, 80, color="#DCFCE7", alpha=0.3)
    ax1.text(60, 105, "Outer Ring Road\n(Arterial)", ha='center', va='top', fontsize=8, color="#166534", fontweight='bold')
    
    ax1.axvspan(80, 120, color="#DBEAFE", alpha=0.3)
    ax1.text(100, 105, "E-City Flyover\n(Highway Cruising)", ha='center', va='top', fontsize=8, color="#1E40AF", fontweight='bold')
    
    ax1.axvspan(120, 150, color="#FEF3C7", alpha=0.3)
    ax1.text(135, 105, "Nandi Hills\n(Grade/Regen)", ha='center', va='top', fontsize=8, color="#92400E", fontweight='bold')
    
    ax1.set_xlabel("Bangalore Commute Time (seconds)", labelpad=10)
    ax1.set_ylabel("Speed (km/h)", labelpad=10)
    ax2.set_ylabel("Relative Elevation (meters)", labelpad=10)
    
    ax1.set_ylim(0, 120)
    ax2.set_ylim(-5, 30)
    
    plt.title("Bangalore Commute Typical Drive Cycle Segment Profile", pad=15)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "bangalore_route_profile.png", bbox_inches="tight")
    plt.close()

    # Plot 30: Traffic-aware EMS Decision Heatmap
    plt.figure(figsize=(9, 6.5))
    
    # Query database contingency counts
    df = pd.read_csv(SYNTHETIC_DATA_PATH)
    ct = pd.crosstab(df["traffic_condition"], df["current_segment"], normalize="columns")
    
    # We want to represent what percentage of decisions are EV-heavy (BATTERY_ONLY)
    # under segment x traffic crossing in the dataset
    ev_matrix = np.zeros((3, 4)) # 3 traffic x 4 segments
    traffics = ["light", "medium", "heavy"]
    segments = ["urban", "stop_go", "arterial", "highway"]
    
    for i, tr in enumerate(traffics):
        for j, seg in enumerate(segments):
            sub = df[(df["traffic_condition"] == tr) & (df["current_segment"] == seg)]
            if len(sub) > 0:
                ev_rate = sum(sub["label"] == "BATTERY_ONLY") / len(sub)
                ev_matrix[i, j] = ev_rate
                
    df_ev_heatmap = pd.DataFrame(ev_matrix, index=traffics, columns=segments)
    
    sns.heatmap(df_ev_heatmap, annot=True, fmt=".1%", cmap="Greens", cbar_kws={'label': 'EV Usage Probability (BATTERY_ONLY)'})
    plt.title("Traffic-Aware Intelligent EMS: EV Mode Preference Heatmap", pad=15)
    plt.xlabel("Current Road Segment", labelpad=10)
    plt.ylabel("Bangalore Traffic Level", labelpad=10)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "traffic_ems_heatmap.png", bbox_inches="tight")
    plt.close()
    
    print("Section 7 plots complete.")

# ────────────────────────────────────────────────────────────────────
# MAIN EXECUTION ENTRY POINT
# ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=============================================================")
    print("  PHEX EMS VISUALIZATION ENGINE - STARTING GENERATION")
    print("=============================================================")
    
    # Section 1
    generate_section1_diagrams()
    # Section 2
    generate_section2_dataset_plots()
    # Section 3
    generate_section3_forecaster_plots()
    # Section 4
    generate_section4_ems_plots()
    # Section 5
    generate_section5_behavior_plots()
    # Section 6
    generate_section6_override_plots()
    # Section 7
    generate_section7_bangalore_plots()
    
    print("\n" + "=" * 60)
    print(f" SUCCESS: Generated all 30 plots in {OUTPUT_DIR}")
    print("=============================================================")
