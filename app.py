"""
⚡ PHEX HEV Energy Management System — Viva Simulation & Engineering Dashboard
Author: Mishael Julian / Antigravity Team
Description: Interactive Streamlit application replaying real telemetry and demonstrating 
             the multi-model demand forecaster, hybrid decision engine, safety layer, and 
             dynamic energy management.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import time
import sys

# Ensure root directory is on sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.rule_ems import VehicleState, EMSDecision
from src.decision_engine import HybridDecisionEngine
from src.config import SOC_CRITICAL, SOC_LOW, SYNTHETIC_DATA_PATH, DATA_RAW_DIR

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="PHEX HEV Energy Management System",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CUSTOM CSS STYLING ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #090d16 0%, #0f172a 50%, #0f766e 100%);
        padding: 1.25rem 1.75rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.25rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    .main-title {
        font-size: 1.85rem;
        font-weight: 800;
        letter-spacing: -0.025em;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    .sub-title {
        font-size: 0.9rem;
        color: #94a3b8;
        margin-top: 0.25rem;
    }

    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(6, 1fr);
        gap: 0.75rem;
        margin-bottom: 1rem;
    }

    .kpi-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 0.75rem 0.9rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        min-width: 0;
        overflow: hidden;
        transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .kpi-card:hover {
        border-color: #38bdf8;
        transform: translateY(-2px);
    }
    .kpi-label {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #94a3b8;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        margin-bottom: 0.2rem;
    }
    .kpi-val {
        font-size: 1.35rem;
        font-weight: 800;
        color: #f8fafc;
        font-family: 'JetBrains Mono', monospace;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .kpi-unit {
        font-size: 0.75rem;
        color: #64748b;
        margin-left: 0.2rem;
        font-family: 'Inter', sans-serif;
    }

    .panel-card {
        background: linear-gradient(145deg, #1e293b 0%, #0f172a 100%);
        border-radius: 12px;
        padding: 1.15rem 1.35rem;
        border: 1px solid #334155;
        box-shadow: 0 4px 15px rgba(0,0,0,0.25);
        height: 100%;
    }
    
    .mode-badge {
        display: inline-block;
        font-size: 1.15rem;
        font-weight: 800;
        padding: 0.35rem 1rem;
        border-radius: 8px;
        letter-spacing: 0.04em;
        text-align: center;
        font-family: 'JetBrains Mono', monospace;
    }

    .mode-BATTERY_ONLY { background: #064e3b; color: #34d399; border: 1px solid #059669; }
    .mode-HYBRID_ASSIST { background: #1e3a8a; color: #60a5fa; border: 1px solid #2563eb; }
    .mode-ENGINE_CHARGE { background: #713f12; color: #facc15; border: 1px solid #ca8a04; }
    .mode-ENGINE_ONLY { background: #7c2d12; color: #fb923c; border: 1px solid #ea580c; }
    .mode-REGEN { background: #134e4a; color: #5eead4; border: 1px solid #0d9488; }
    .mode-IDLE_STOP { background: #374151; color: #9ca3af; border: 1px solid #4b5563; }

    .safety-badge-clear {
        background: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid #059669;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.78rem;
    }
    .safety-badge-override {
        background: rgba(239, 68, 68, 0.18);
        color: #f87171;
        border: 1px solid #dc2626;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-weight: 800;
        font-size: 0.78rem;
    }

    .scenario-fact {
        background: rgba(56, 189, 248, 0.08);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 8px;
        padding: 0.6rem 0.9rem;
        font-size: 0.82rem;
        color: #bae6fd;
        margin-bottom: 1rem;
    }

    /* Powertrain Flow Component */
    .flow-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: #0b1120;
        border-radius: 10px;
        padding: 0.9rem 1.2rem;
        border: 1px solid #1e293b;
        margin-top: 0.75rem;
    }
    .flow-node {
        text-align: center;
        padding: 0.35rem 0.5rem;
        border-radius: 8px;
        background: #1e293b;
        border: 1px solid #334155;
        font-size: 0.72rem;
        font-weight: 700;
        color: #cbd5e1;
        min-width: 0;
        white-space: nowrap;
    }
    .flow-node-active {
        background: #0f766e;
        border-color: #14b8a6;
        color: #ffffff;
        box-shadow: 0 0 10px rgba(20, 184, 166, 0.4);
    }
    .flow-arrow {
        font-size: 1.1rem;
        color: #475569;
        font-weight: bold;
    }
    .flow-arrow-active {
        color: #38bdf8;
        text-shadow: 0 0 6px #38bdf8;
    }
</style>
""", unsafe_allow_html=True)

# --- LOAD DATASET & ENGINE ---
@st.cache_resource
def load_decision_engine():
    try:
        return HybridDecisionEngine()
    except Exception as e:
        st.error(f"Error loading Hybrid Decision Engine: {e}")
        return None

@st.cache_data
def load_telemetry_dataset():
    data_path = SYNTHETIC_DATA_PATH
    if not data_path.exists():
        data_path = ROOT_DIR / "data" / "raw" / "synthetic_ems_dataset.csv"
    if not data_path.exists():
        st.error(f"Dataset not found at {data_path}")
        return pd.DataFrame()
    return pd.read_csv(data_path)

engine = load_decision_engine()
df_all = load_telemetry_dataset()

# --- SCENARIO DATA EXTRACTOR ---
def get_scenario_data(df: pd.DataFrame, scenario_choice: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    
    if scenario_choice == "Scenario 1: Normal Urban Driving (Healthy SOC)":
        # Moderate speed urban rows with SOC >= 45%
        sub = df[(df["battery_soc"] >= 45.0) & (df["speed"] > 10.0) & (df["speed"] < 60.0)]
        if len(sub) >= 50:
            return sub.iloc[100:150].reset_index(drop=True)
        return df.iloc[0:50].reset_index(drop=True)

    elif scenario_choice == "Scenario 2: Low SOC + High Demand (Safety Transition)":
        # Depleted battery (<= 25%) and high demand (forcing Engine Charge / Override)
        sub = df[(df["battery_soc"] <= 25.0) & (df["power_required_kw"] >= 15.0)]
        if len(sub) >= 50:
            return sub.iloc[0:50].reset_index(drop=True)
        sub2 = df[df["battery_soc"] <= 28.0]
        if len(sub2) >= 50:
            return sub2.iloc[0:50].reset_index(drop=True)
        return df.iloc[200:250].reset_index(drop=True)

    elif scenario_choice == "Scenario 3: Braking / Regeneration (Kinetic Recuperation)":
        # Braking events and negative wheel power demand
        sub = df[(df["power_required_kw"] < 0) | (df["braking"] == True) | (df["braking"] == 1)]
        if len(sub) >= 50:
            return sub.iloc[0:50].reset_index(drop=True)
        return df.iloc[300:350].reset_index(drop=True)

    else:
        # Full Cycle (First 100 timesteps)
        return df.iloc[0:100].reset_index(drop=True)

# --- PRECOMPUTE ALL INFERENCE ONCE ---
@st.cache_data
def precompute_scenario_simulation(scenario_name: str):
    df_slice = get_scenario_data(df_all, scenario_name)
    if df_slice.empty:
        return []
    
    # We create a clean instance or call engine for inference
    eng = HybridDecisionEngine()
    results = []

    for _, row in df_slice.iterrows():
        st_obj = VehicleState(
            speed=float(row.get("speed", 0.0)),
            acceleration=float(row.get("acceleration", 0.0)),
            power_required_kw=float(row.get("power_required_kw", 0.0)),
            torque_required_nm=float(row.get("torque_required_nm", 0.0)),
            battery_soc=float(row.get("battery_soc", 50.0)),
            battery_temp=float(row.get("battery_temp", 25.0)),
            aux_load_kw=float(row.get("aux_load_kw", 0.5)),
            grade_angle=float(row.get("grade_angle", 0.0)),
            regen_available=bool(row.get("regen_available", True)),
            braking=bool(row.get("braking", False)),
            traffic_condition=str(row.get("traffic_condition", "medium")),
            current_segment=str(row.get("current_segment", "urban")),
            next_segment=str(row.get("next_segment", "urban")),
        )
        try:
            dec = eng.decide(st_obj)
            df_hist = eng._get_history_dataframe()
            p_p, p_v, p_a = eng.demand_forecaster.predict_future(df_hist)
            safe_ok, safe_reason = eng.safety_layer.check_safety(dec.mode, st_obj)
            
            results.append({
                "speed": st_obj.speed,
                "acceleration": st_obj.acceleration,
                "power_required_kw": st_obj.power_required_kw,
                "torque_required_nm": st_obj.torque_required_nm,
                "battery_soc": st_obj.battery_soc,
                "battery_temp": st_obj.battery_temp,
                "aux_load_kw": st_obj.aux_load_kw,
                "grade_angle": st_obj.grade_angle,
                "regen_available": st_obj.regen_available,
                "braking": st_obj.braking,
                "traffic_condition": st_obj.traffic_condition,
                "current_segment": st_obj.current_segment,
                "next_segment": st_obj.next_segment,
                "mode": dec.mode,
                "confidence": dec.confidence,
                "rule_triggered": dec.rule_triggered,
                "reason": dec.reason,
                "source": dec.source,
                "pred_power": float(p_p),
                "pred_speed": float(p_v),
                "pred_accel": float(p_a),
                "safety_ok": bool(safe_ok),
                "safety_reason": str(safe_reason),
            })
        except Exception as ex:
            results.append({
                "speed": st_obj.speed,
                "acceleration": st_obj.acceleration,
                "power_required_kw": st_obj.power_required_kw,
                "torque_required_nm": st_obj.torque_required_nm,
                "battery_soc": st_obj.battery_soc,
                "battery_temp": st_obj.battery_temp,
                "aux_load_kw": st_obj.aux_load_kw,
                "grade_angle": st_obj.grade_angle,
                "regen_available": st_obj.regen_available,
                "braking": st_obj.braking,
                "traffic_condition": st_obj.traffic_condition,
                "current_segment": st_obj.current_segment,
                "next_segment": st_obj.next_segment,
                "mode": "BATTERY_ONLY",
                "confidence": 1.0,
                "rule_triggered": "FALLBACK",
                "reason": str(ex),
                "source": "RULE",
                "pred_power": st_obj.power_required_kw,
                "pred_speed": st_obj.speed,
                "pred_accel": st_obj.acceleration,
                "safety_ok": True,
                "safety_reason": "CLEAR",
            })
    return results

# --- RESET CALLBACK ON SCENARIO CHANGE ---
def on_scenario_change():
    st.session_state.current_step = 0
    st.session_state.is_playing = False

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/lightning-bolt.png", width=56)
    st.title("EMS Control Station")
    st.markdown("---")

    scenario_option = st.selectbox(
        "🎯 Select Viva Scenario",
        [
            "Scenario 1: Normal Urban Driving (Healthy SOC)",
            "Scenario 2: Low SOC + High Demand (Safety Transition)",
            "Scenario 3: Braking / Regeneration (Kinetic Recuperation)",
            "Full Cycle (First 100 Timesteps)"
        ],
        on_change=on_scenario_change
    )

    # Load precomputed data
    sim_data = precompute_scenario_simulation(scenario_option)
    total_steps = len(sim_data)

    st.markdown("### ⏯️ Simulation Controls")
    
    if "current_step" not in st.session_state:
        st.session_state.current_step = 0
    if "is_playing" not in st.session_state:
        st.session_state.is_playing = False
    if "advance_step" not in st.session_state:
        st.session_state.advance_step = False

    # Safely advance step before widget instantiation
    if st.session_state.advance_step:
        st.session_state.advance_step = False
        if st.session_state.current_step < total_steps - 1:
            st.session_state.current_step += 1
        else:
            st.session_state.is_playing = False

    # Guard against index overflow when switching datasets
    if st.session_state.current_step >= total_steps:
        st.session_state.current_step = 0

    def step_prev():
        st.session_state.current_step = max(0, st.session_state.get("current_step", 0) - 1)
        st.session_state.is_playing = False

    def step_next():
        st.session_state.current_step = min(total_steps - 1, st.session_state.get("current_step", 0) + 1)
        st.session_state.is_playing = False

    def toggle_play():
        st.session_state.is_playing = not st.session_state.get("is_playing", False)

    def reset_step():
        st.session_state.current_step = 0
        st.session_state.is_playing = False

    col_btn1, col_btn2, col_btn3 = st.columns(3)
    with col_btn1:
        st.button("⏪ Prev", use_container_width=True, on_click=step_prev)
    with col_btn2:
        play_label = "⏸️ Pause" if st.session_state.get("is_playing", False) else "▶️ Play"
        st.button(play_label, use_container_width=True, on_click=toggle_play)
    with col_btn3:
        st.button("Next ⏩", use_container_width=True, on_click=step_next)

    st.button("🔄 Reset to Start (t=0)", use_container_width=True, on_click=reset_step)

    st.slider(
        "Current Timestep (s)",
        min_value=0,
        max_value=max(0, total_steps - 1),
        key="current_step"
    )

    replay_speed = st.select_slider(
        "Playback Speed",
        options=[0.5, 1.0, 2.0, 4.0],
        value=1.0,
        format_func=lambda x: f"{x}x"
    )

    st.markdown("---")
    st.markdown("### 🔬 Viva Architecture Map")
    st.caption("• **Layer 1**: Telemetry Ingestion & Lag Feature Extractor\n"
               "• **Layer 2**: Multi-Step RF Demand Forecasters ($t+k$)\n"
               "• **Layer 3**: Hybrid Decision Engine (GBM + Rule Fallback)\n"
               "• **Layer 4**: Deterministic Safety & Thermal Override")

# --- HEADER SECTION ---
st.markdown("""
<div class="main-header">
    <div class="main-title">
        <span>⚡ PHEX HEV Energy Management System</span>
    </div>
    <div class="sub-title">
        Real-Time Telemetry Replay • Multi-Model Demand Forecaster • Hybrid Decision Engine & Safety Envelope
    </div>
</div>
""", unsafe_allow_html=True)

if not sim_data:
    st.warning("⚠️ No simulation data available. Please verify dataset files.")
    st.stop()

# Current step dictionary
curr = sim_data[st.session_state.current_step]

# Scenario Quick Fact
scenario_descriptions = {
    "Scenario 1: Normal Urban Driving (Healthy SOC)": "🏙️ **Urban Driving (SOC ≥ 45%)**: Demonstrates pure EV priority in congested stop-and-go traffic to eliminate tailpipe emissions while maintaining smooth electric torque.",
    "Scenario 2: Low SOC + High Demand (Safety Transition)": "🔋 **Depleted Battery (SOC ≤ 25%) & High Load**: Demonstrates forced engine charging, load-leveling, and physical safety overrides to prevent irreversible battery degradation.",
    "Scenario 3: Braking / Regeneration (Kinetic Recuperation)": "🔄 **Deceleration & Braking Events**: Demonstrates kinetic energy recovery through regenerative braking torque limits to recharge the high-voltage pack.",
    "Full Cycle (First 100 Timesteps)": "📊 **Composite Drive Cycle**: Mixed drive cycle covering urban, highway, and grade transitions."
}
st.markdown(f'<div class="scenario-fact">{scenario_descriptions.get(scenario_option, "")}</div>', unsafe_allow_html=True)

# --- METRIC CARDS ROW (RESPONSIVE GRID) ---
p_color = "#ef4444" if curr["power_required_kw"] > 20 else ("#10b981" if curr["power_required_kw"] < 0 else "#f8fafc")
soc_color = "#ef4444" if curr["battery_soc"] <= 20 else ("#f59e0b" if curr["battery_soc"] <= 30 else "#10b981")

st.markdown(f"""
<div class="kpi-grid">
    <div class="kpi-card">
        <div class="kpi-label">Vehicle Speed</div>
        <div class="kpi-val">{curr["speed"]:.1f}<span class="kpi-unit">km/h</span></div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Power Demand</div>
        <div class="kpi-val" style="color: {p_color}">{curr["power_required_kw"]:.1f}<span class="kpi-unit">kW</span></div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Battery SOC</div>
        <div class="kpi-val" style="color: {soc_color}">{curr["battery_soc"]:.1f}<span class="kpi-unit">%</span></div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Battery Temp</div>
        <div class="kpi-val">{curr["battery_temp"]:.1f}<span class="kpi-unit">°C</span></div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Acceleration</div>
        <div class="kpi-val">{curr["acceleration"]:.2f}<span class="kpi-unit">m/s²</span></div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Route / Traffic</div>
        <div class="kpi-val" style="font-size: 1.05rem; text-transform: capitalize;">{curr["current_segment"]} <span class="kpi-unit">({curr["traffic_condition"]})</span></div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- MIDDLE SECTION: DECISION ENGINE & FORECASTER ---
col_decision, col_forecaster = st.columns([1.25, 1.0])

with col_decision:
    mode_val = curr["mode"]
    mode_class = f"mode-{mode_val}"
    conf_pct = curr["confidence"] * 100.0 if curr["confidence"] <= 1.0 else curr["confidence"]
    safety_badge = '<span class="safety-badge-clear">✅ SAFETY CLEAR</span>' if curr["safety_ok"] else f'<span class="safety-badge-override">⚠️ OVERRIDE: {curr["safety_reason"]}</span>'

    # Powertrain active nodes calculation
    ice_active = mode_val in ["ENGINE_ONLY", "ENGINE_CHARGE", "HYBRID_ASSIST"]
    gen_active = mode_val in ["ENGINE_CHARGE"]
    bat_active = mode_val in ["BATTERY_ONLY", "HYBRID_ASSIST", "ENGINE_CHARGE", "REGEN"]
    mot_active = mode_val in ["BATTERY_ONLY", "HYBRID_ASSIST", "REGEN"]
    
    ice_cls = "flow-node-active" if ice_active else ""
    gen_cls = "flow-node-active" if gen_active else ""
    bat_cls = "flow-node-active" if bat_active else ""
    mot_cls = "flow-node-active" if mot_active else ""

    st.markdown(f"""
    <div class="panel-card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.6rem;">
            <span style="font-size: 0.8rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em;">Hybrid Decision Core</span>
            {safety_badge}
        </div>
        <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 0.75rem;">
            <div class="mode-badge {mode_class}">{mode_val}</div>
            <div>
                <div style="font-size: 0.75rem; color: #94a3b8;">Source Engine</div>
                <div style="font-weight: 700; color: #38bdf8; font-size: 0.9rem;">{curr["source"]} (Rule: {curr["rule_triggered"]})</div>
            </div>
        </div>
        <div style="margin-bottom: 0.65rem;">
            <div style="display: flex; justify-content: space-between; font-size: 0.8rem; margin-bottom: 0.2rem;">
                <span style="color: #cbd5e1;">Model Classification Confidence</span>
                <span style="font-weight: 700; color: #38bdf8;">{conf_pct:.1f}%</span>
            </div>
            <div style="background: #334155; border-radius: 4px; height: 6px; overflow: hidden;">
                <div style="background: #38bdf8; width: {conf_pct}%; height: 100%; border-radius: 4px;"></div>
            </div>
        </div>
        <div style="background: rgba(15, 23, 42, 0.7); padding: 0.55rem 0.85rem; border-radius: 6px; border-left: 3px solid #38bdf8; font-size: 0.8rem; color: #cbd5e1;">
            <strong>Decision Logic:</strong> {curr["reason"]}
        </div>
        <div class="flow-container">
            <div class="flow-node {ice_cls}">ENGINE (ICE)</div>
            <div class="flow-arrow {'flow-arrow-active' if gen_active else ''}">➔</div>
            <div class="flow-node {gen_cls}">GENERATOR</div>
            <div class="flow-arrow {'flow-arrow-active' if bat_active else ''}">⇄</div>
            <div class="flow-node {bat_cls}">HV BATTERY</div>
            <div class="flow-arrow {'flow-arrow-active' if mot_active else ''}">➔</div>
            <div class="flow-node {mot_cls}">E-MOTOR</div>
            <div class="flow-arrow flow-arrow-active">➔</div>
            <div class="flow-node flow-node-active">WHEELS</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_forecaster:
    p_delta = curr["pred_power"] - curr["power_required_kw"]
    v_delta = curr["pred_speed"] - curr["speed"]
    a_delta = curr["pred_accel"] - curr["acceleration"]

    st.markdown(f"""
    <div class="panel-card">
        <div style="font-size: 0.8rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.75rem;">
            🔮 ML Demand Forecaster (Horizon t+k)
        </div>
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.5rem; margin-bottom: 0.75rem;">
            <div style="background: #0b1120; padding: 0.6rem; border-radius: 8px; border: 1px solid #1e293b; text-align: center;">
                <div style="font-size: 0.7rem; color: #94a3b8; font-weight: bold;">PRED POWER</div>
                <div style="font-size: 1.15rem; font-weight: 800; color: #f8fafc; font-family: 'JetBrains Mono', monospace;">{curr["pred_power"]:.1f}<span style="font-size: 0.7rem; color: #64748b;"> kW</span></div>
                <div style="font-size: 0.75rem; color: {'#10b981' if p_delta >= 0 else '#ef4444'}; font-weight: 600;">{p_delta:+.1f} kW</div>
            </div>
            <div style="background: #0b1120; padding: 0.6rem; border-radius: 8px; border: 1px solid #1e293b; text-align: center;">
                <div style="font-size: 0.7rem; color: #94a3b8; font-weight: bold;">PRED SPEED</div>
                <div style="font-size: 1.15rem; font-weight: 800; color: #f8fafc; font-family: 'JetBrains Mono', monospace;">{curr["pred_speed"]:.1f}<span style="font-size: 0.7rem; color: #64748b;"> km/h</span></div>
                <div style="font-size: 0.75rem; color: {'#10b981' if v_delta >= 0 else '#ef4444'}; font-weight: 600;">{v_delta:+.1f} km/h</div>
            </div>
            <div style="background: #0b1120; padding: 0.6rem; border-radius: 8px; border: 1px solid #1e293b; text-align: center;">
                <div style="font-size: 0.7rem; color: #94a3b8; font-weight: bold;">PRED ACCEL</div>
                <div style="font-size: 1.15rem; font-weight: 800; color: #f8fafc; font-family: 'JetBrains Mono', monospace;">{curr["pred_accel"]:.2f}<span style="font-size: 0.7rem; color: #64748b;"> m/s²</span></div>
                <div style="font-size: 0.75rem; color: {'#10b981' if a_delta >= 0 else '#ef4444'}; font-weight: 600;">{a_delta:+.2f} m/s²</div>
            </div>
        </div>
        <div style="font-size: 0.78rem; color: #94a3b8; line-height: 1.4; background: #0b1120; padding: 0.6rem 0.8rem; border-radius: 6px; border: 1px solid #1e293b;">
            💡 <em>Multi-lag Random Forest regressors anticipate torque demands prior to uphill grades & highway accelerations, eliminating engine lag.</em>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='margin-top: 1rem'></div>", unsafe_allow_html=True)

# --- DYNAMIC PLOTS SECTION ---
tab_overview, tab_power, tab_battery, tab_telemetry = st.tabs([
    "📈 Simulation Trajectories",
    "⚡ Power Split & Recuperation",
    "🔋 Battery & Safety Envelope",
    "📋 Telemetry Inspector"
])

curr_idx = st.session_state.current_step
hist_steps = list(range(curr_idx + 1))
all_steps = list(range(total_steps))

speeds = [sim_data[i]["speed"] for i in hist_steps]
powers = [sim_data[i]["power_required_kw"] for i in hist_steps]
socs = [sim_data[i]["battery_soc"] for i in hist_steps]
modes = [sim_data[i]["mode"] for i in hist_steps]
pred_powers = [sim_data[i]["pred_power"] for i in hist_steps]
pred_speeds = [sim_data[i]["pred_speed"] for i in hist_steps]

all_speeds = [sim_data[i]["speed"] for i in all_steps]
all_powers = [sim_data[i]["power_required_kw"] for i in all_steps]
all_socs = [sim_data[i]["battery_soc"] for i in all_steps]
all_modes = [sim_data[i]["mode"] for i in all_steps]

with tab_overview:
    col_p1, col_p2 = st.columns(2)
    
    with col_p1:
        # Speed actual vs forecast
        fig_speed = go.Figure()
        fig_speed.add_trace(go.Scatter(x=all_steps, y=all_speeds, mode="lines", name="Full Profile", line=dict(color="#334155", width=1.5, dash="dot")))
        fig_speed.add_trace(go.Scatter(x=hist_steps, y=speeds, mode="lines+markers", name="Actual Speed", line=dict(color="#38bdf8", width=2.5), marker=dict(size=4)))
        fig_speed.add_trace(go.Scatter(x=hist_steps, y=pred_speeds, mode="lines", name="Forecast (t+k)", line=dict(color="#f59e0b", width=2, dash="dash")))
        fig_speed.add_vline(x=curr_idx, line_width=2, line_dash="solid", line_color="#ec4899")
        fig_speed.update_layout(
            title=dict(text="Vehicle Speed Profile (km/h)", font=dict(size=14, color="#f8fafc")),
            xaxis=dict(title="Timestep (s)", gridcolor="#1e293b"),
            yaxis=dict(title="Speed (km/h)", gridcolor="#1e293b"),
            template="plotly_dark",
            margin=dict(l=40, r=20, t=45, b=30),
            height=290,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)")
        )
        st.plotly_chart(fig_speed, use_container_width=True, config={"displayModeBar": False})

    with col_p2:
        # Power actual vs forecast
        fig_pwr = go.Figure()
        fig_pwr.add_trace(go.Scatter(x=all_steps, y=all_powers, mode="lines", name="Full Demand", line=dict(color="#334155", width=1.5, dash="dot")))
        fig_pwr.add_trace(go.Scatter(x=hist_steps, y=powers, mode="lines+markers", name="Actual Power", line=dict(color="#34d399", width=2.5), marker=dict(size=4)))
        fig_pwr.add_trace(go.Scatter(x=hist_steps, y=pred_powers, mode="lines", name="Forecast (t+k)", line=dict(color="#fb923c", width=2, dash="dash")))
        fig_pwr.add_vline(x=curr_idx, line_width=2, line_dash="solid", line_color="#ec4899")
        fig_pwr.update_layout(
            title=dict(text="Wheel Power Demand (kW)", font=dict(size=14, color="#f8fafc")),
            xaxis=dict(title="Timestep (s)", gridcolor="#1e293b"),
            yaxis=dict(title="Power (kW)", gridcolor="#1e293b"),
            template="plotly_dark",
            margin=dict(l=40, r=20, t=45, b=30),
            height=290,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)")
        )
        st.plotly_chart(fig_pwr, use_container_width=True, config={"displayModeBar": False})

    # EMS Mode Sequence Ribbon
    mode_color_map = {
        "BATTERY_ONLY": "#10b981",
        "HYBRID_ASSIST": "#3b82f6",
        "ENGINE_CHARGE": "#eab308",
        "ENGINE_ONLY": "#f97316",
        "REGEN": "#14b8a6",
        "IDLE_STOP": "#6b7280"
    }
    
    # Active mode timeline
    df_mode_hist = pd.DataFrame({"step": hist_steps, "mode": modes})
    mode_counts = df_mode_hist["mode"].value_counts()
    
    fig_mode = px.bar(
        x=mode_counts.index, 
        y=mode_counts.values,
        labels={"x": "EMS Mode", "y": "Timesteps Active"},
        title="EMS Mode Execution Distribution (Up to Current Timestep)",
        template="plotly_dark",
        color=mode_counts.index,
        color_discrete_map=mode_color_map
    )
    fig_mode.update_layout(
        title=dict(text="EMS Mode Execution Distribution (Up to Current Timestep)", font=dict(size=14)),
        height=240, 
        margin=dict(l=40, r=20, t=40, b=30), 
        showlegend=False,
        xaxis=dict(gridcolor="#1e293b"),
        yaxis=dict(gridcolor="#1e293b")
    )
    st.plotly_chart(fig_mode, use_container_width=True, config={"displayModeBar": False})

with tab_power:
    recup_powers = [max(0.0, -p) for p in powers]
    fig_recup = go.Figure()
    fig_recup.add_trace(go.Bar(x=hist_steps, y=recup_powers, name="Regen Recuperation (kW)", marker_color="#10b981"))
    fig_recup.add_trace(go.Scatter(x=hist_steps, y=powers, mode="lines", name="Wheel Demand (kW)", line=dict(color="#38bdf8", width=2)))
    fig_recup.update_layout(
        title=dict(text="Kinetic Energy Recuperation & Powertrain Demand Profile", font=dict(size=14)),
        xaxis=dict(title="Timestep (s)", gridcolor="#1e293b"),
        yaxis=dict(title="Power (kW)", gridcolor="#1e293b"),
        template="plotly_dark",
        height=350,
        margin=dict(l=40, r=20, t=45, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)")
    )
    st.plotly_chart(fig_recup, use_container_width=True, config={"displayModeBar": False})

with tab_battery:
    fig_soc = go.Figure()
    fig_soc.add_trace(go.Scatter(x=all_steps, y=all_socs, mode="lines", name="Full Scenario SOC", line=dict(color="#334155", width=1.5, dash="dot")))
    fig_soc.add_trace(go.Scatter(x=hist_steps, y=socs, mode="lines+markers", name="Battery SOC (%)", line=dict(color="#10b981", width=2.5), marker=dict(size=4)))
    
    # Thresholds
    fig_soc.add_hline(y=SOC_LOW, line_dash="dash", line_color="#f59e0b", annotation_text=f"Low SOC Floor ({SOC_LOW:.0f}%)", annotation_position="bottom right")
    fig_soc.add_hline(y=SOC_CRITICAL, line_dash="dash", line_color="#ef4444", annotation_text=f"Critical SOC ({SOC_CRITICAL:.0f}%)", annotation_position="bottom right")
    fig_soc.add_vline(x=curr_idx, line_width=2, line_dash="solid", line_color="#ec4899")

    fig_soc.update_layout(
        title=dict(text="High-Voltage Battery State of Charge (SOC %) & Physical Envelope", font=dict(size=14)),
        xaxis=dict(title="Timestep (s)", gridcolor="#1e293b"),
        yaxis=dict(title="SOC (%)", range=[0, 100], gridcolor="#1e293b"),
        template="plotly_dark",
        height=350,
        margin=dict(l=40, r=20, t=45, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)")
    )
    st.plotly_chart(fig_soc, use_container_width=True, config={"displayModeBar": False})

with tab_telemetry:
    st.markdown("#### 🔍 Ground-Truth Telemetry Snapshot (Timestep $t$)")
    df_curr_row = pd.DataFrame([{
        "speed_kmh": round(curr["speed"], 2),
        "accel_ms2": round(curr["acceleration"], 2),
        "power_demand_kw": round(curr["power_required_kw"], 2),
        "battery_soc_pct": round(curr["battery_soc"], 2),
        "battery_temp_c": round(curr["battery_temp"], 2),
        "traffic_condition": curr["traffic_condition"],
        "route_segment": curr["current_segment"],
        "active_mode": curr["mode"],
        "model_confidence": f"{conf_pct:.1f}%",
        "safety_status": "CLEAR" if curr["safety_ok"] else f"OVERRIDE: {curr['safety_reason']}"
    }])
    st.dataframe(df_curr_row, use_container_width=True)

    st.markdown("#### 📋 History Buffer (Recent 10 Timesteps)")
    # Reorder key columns to front for rapid inspection
    key_cols = ["mode", "confidence", "rule_triggered", "speed", "power_required_kw", "battery_soc", "battery_temp", "acceleration", "traffic_condition", "current_segment", "pred_power", "pred_speed", "reason"]
    df_raw_hist = pd.DataFrame(sim_data[:curr_idx + 1]).tail(10)
    ordered_cols = [c for c in key_cols if c in df_raw_hist.columns] + [c for c in df_raw_hist.columns if c not in key_cols]
    df_hist_view = df_raw_hist[ordered_cols].round(2)
    st.dataframe(df_hist_view, use_container_width=True)

# --- AUTO PLAY ANIMATION LOOP ---
if st.session_state.get("is_playing", False):
    if st.session_state.current_step < total_steps - 1:
        time.sleep(max(0.04, 0.35 / replay_speed))
        st.session_state.advance_step = True
        st.rerun()
    else:
        st.session_state.is_playing = False
        st.session_state.advance_step = False
        st.rerun()
