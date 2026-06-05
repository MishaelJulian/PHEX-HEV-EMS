import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

# Configure paths
BASE_DIR = Path(r"c:\Users\misha\OneDrive\Desktop\PHEX")
DATA_DIR = BASE_DIR / "data" / "processed"
FIGURES_DIR = BASE_DIR / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def plot_behavior_eda(df):
    logger.info("Generating Driver Behavior plots...")
    
    # 1. Correlation Heatmap
    plt.figure(figsize=(10, 8))
    # Select original unscaled numerical columns for readability
    cols_to_plot = ["efficiency_score", "speed_kmph", "throttle", "brake_pressure", 
                    "accel_x", "accel_y", "steering_angle"]
    corr = df[cols_to_plot].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
    plt.title("Correlation Heatmap: Driver Behavior Features vs Efficiency Target")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "behavior_correlation_heatmap.png", dpi=300)
    plt.close()

    # 2. Efficiency Distribution by Behavior Class
    plt.figure(figsize=(8, 6))
    sns.kdeplot(data=df, x="efficiency_score", hue="behavior_label", fill=True, alpha=0.5)
    plt.title("Distribution of Efficiency Score by Driving Behavior")
    plt.xlabel("Efficiency Score (0-100)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "behavior_efficiency_distribution.png", dpi=300)
    plt.close()

def plot_telemetry_eda(df):
    logger.info("Generating Telemetry IMU plots...")
    
    # 1. Jerk Magnitude by Class
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x="Class", y="jerk_magnitude")
    plt.title("Jerk Magnitude across Driving Classes")
    plt.xlabel("Driving Class")
    plt.ylabel("Jerk Magnitude (rate of acceleration change)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "telemetry_jerk_boxplot.png", dpi=300)
    plt.close()
    
    # 2. Acceleration Magnitude by Class
    plt.figure(figsize=(8, 6))
    sns.violinplot(data=df, x="Class", y="acc_magnitude")
    plt.title("Total Acceleration Magnitude by Driving Class")
    plt.xlabel("Driving Class")
    plt.ylabel("Acceleration Magnitude (g)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "telemetry_accel_violin.png", dpi=300)
    plt.close()

def plot_nasa_eda(df):
    logger.info("Generating NASA Battery plots...")
    
    # Filter for discharge steps to plot capacity proxy over cycles
    discharge_df = df[df["step_type"] == "D"].copy()
    
    # 1. Capacity Degradation over Cycles
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=discharge_df, x="cycle_number", y="capacity_proxy", hue="battery", alpha=0.6, s=15)
    
    # Add trendlines
    for batt in discharge_df["battery"].unique():
        batt_data = discharge_df[discharge_df["battery"] == batt].dropna(subset=["cycle_number", "capacity_proxy"])
        sns.regplot(data=batt_data, x="cycle_number", y="capacity_proxy", 
                    scatter=False, label=f"{batt} Trend", line_kws={"linewidth": 2})
                    
    plt.title("Battery Capacity Degradation over Cycles (Proxy)")
    plt.xlabel("Cycle Number")
    plt.ylabel("Capacity Proxy (|Current| * duration)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "nasa_capacity_degradation.png", dpi=300)
    plt.close()

    # 2. Voltage Sag vs Discharge Current
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=discharge_df, x="current_mean", y="voltage_delta", hue="battery", alpha=0.5)
    plt.title("Voltage Drop during Discharge Step vs Mean Current")
    plt.xlabel("Mean Current (A)")
    plt.ylabel("Voltage Delta (Drop in V)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "nasa_voltage_sag.png", dpi=300)
    plt.close()

# =============================================================================
# EMS SPECIFIC EDA PLOTS
# =============================================================================
import time
from src.config import REPORTS_DIR

def plot_class_distribution(df: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 6))
    counts = df['label'].value_counts()
    percentages = df['label'].value_counts(normalize=True) * 100
    
    sns.barplot(y=counts.index, x=counts.values, palette="husl")
    plt.title("EMS Mode Decision Distribution")
    plt.xlabel("Count")
    plt.ylabel("EMS Mode")
    
    for i, (count, pct) in enumerate(zip(counts.values, percentages.values)):
        plt.text(count + max(counts.values)*0.01, i, f"{count} ({pct:.1f}%)", va='center')
        
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "class_distribution.png", dpi=300)
    plt.close()

def plot_soc_vs_decision(df: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df, x="label", y="battery_soc", palette="husl")
    plt.title("Battery SOC Distribution per EMS Decision Mode")
    plt.xlabel("EMS Mode")
    plt.ylabel("Battery SOC (%)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "soc_vs_decision.png", dpi=300)
    plt.close()

def plot_speed_vs_decision(df: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df, x="label", y="speed", palette="husl")
    plt.title("Vehicle Speed Distribution per EMS Decision Mode")
    plt.xlabel("EMS Mode")
    plt.ylabel("Speed (km/h)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "speed_vs_decision.png", dpi=300)
    plt.close()

def plot_power_vs_decision(df: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="label", y="power_required_kw", palette="husl")
    plt.title("Power Demand Distribution per EMS Decision Mode")
    plt.xlabel("EMS Mode")
    plt.ylabel("Power Required (kW)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "power_vs_decision.png", dpi=300)
    plt.close()

def plot_correlation_matrix(df: pd.DataFrame) -> None:
    plt.figure(figsize=(14, 10))
    numeric_df = df.select_dtypes(include=['float64', 'int64', 'int32'])
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=False, cmap="coolwarm", center=0)
    plt.title("EMS Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "correlation_matrix.png", dpi=300)
    plt.close()

def plot_feature_histograms(df: pd.DataFrame) -> None:
    numeric_cols = df.select_dtypes(include=['float64']).columns
    n_cols = 3
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten()
    
    for i, col in enumerate(numeric_cols):
        sns.histplot(data=df, x=col, hue="label", kde=True, ax=axes[i], element="step", stat="density", common_norm=False)
        axes[i].set_title(col)
        
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
        
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "feature_histograms.png", dpi=300)
    plt.close()

def plot_segment_traffic_distribution(df: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 6))
    cross_tab = pd.crosstab(df['current_segment'], df['label'])
    cross_tab.plot(kind='bar', stacked=True, colormap='tab10', figsize=(10, 6))
    plt.title("EMS Mode Distribution per Road Segment")
    plt.xlabel("Road Segment")
    plt.ylabel("Count")
    plt.legend(title="EMS Mode", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "segment_decision_dist.png", dpi=300)
    plt.close()

def run_full_eda(df: pd.DataFrame) -> None:
    logger.info("Running full EMS EDA pipeline...")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    plots = [
        ("Class Distribution", plot_class_distribution),
        ("SOC vs Decision", plot_soc_vs_decision),
        ("Speed vs Decision", plot_speed_vs_decision),
        ("Power vs Decision", plot_power_vs_decision),
        ("Correlation Matrix", plot_correlation_matrix),
        ("Feature Histograms", plot_feature_histograms),
        ("Segment Distribution", plot_segment_traffic_distribution)
    ]
    
    for name, func in plots:
        start_time = time.time()
        logger.info(f"  -> Generating {name}...")
        func(df)
        logger.info(f"     Done in {time.time() - start_time:.2f}s")
        
    logger.info(f"EMS EDA complete. Plots saved to {REPORTS_DIR}")

def main():
    # Load processed datasets
    behavior_path = DATA_DIR / "behavior_processed.csv"
    telemetry_path = DATA_DIR / "telemetry_train_processed.csv"
    nasa_path = DATA_DIR / "nasa_battery_processed.csv"

    if behavior_path.exists():
        df_behavior = pd.read_csv(behavior_path)
        plot_behavior_eda(df_behavior)
    
    if telemetry_path.exists():
        df_telemetry = pd.read_csv(telemetry_path)
        plot_telemetry_eda(df_telemetry)
        
    if nasa_path.exists():
        df_nasa = pd.read_csv(nasa_path)
        plot_nasa_eda(df_nasa)
        
    logger.info(f"Legacy EDA complete. Plots saved to {FIGURES_DIR}")
    
    # Load EMS Synthetic Dataset
    from src.config import SYNTHETIC_DATA_PATH
    if SYNTHETIC_DATA_PATH.exists():
        logger.info("Found synthetic EMS dataset. Running EMS EDA...")
        df_ems = pd.read_csv(SYNTHETIC_DATA_PATH)
        run_full_eda(df_ems)

if __name__ == "__main__":
    main()
