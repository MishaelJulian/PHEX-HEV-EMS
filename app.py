"""
⚡ PHEX HEV Energy Management System — Engineering Intelligence Suite
Author: Mishael Julian / Antigravity Team
Description: Interactive, publication-grade engineering dashboard featuring real telemetry replay,
             multi-step demand forecasting, hybrid decision engine, battery degradation analytics,
             powertrain efficiency maps, and comprehensive ML diagnostics.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import json
import time
import sys

# Ensure root directory is on sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.rule_ems import VehicleState, EMSDecision
from src.decision_engine import HybridDecisionEngine
from src.config import SOC_CRITICAL, SOC_LOW, SYNTHETIC_DATA_PATH

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="PHEX HEV Energy Management System",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- MODERN STYLING (GLASSMORPHISM, INTER TYPOGRAPHY, CLEAN MOTION) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Sleek container headers */
    .hero-header {
        background: radial-gradient(circle at 10% 20%, rgba(15, 118, 110, 0.25) 0%, rgba(15, 23, 42, 0.95) 90%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 1.4rem 2rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(8px);
    }
    
    .hero-title {
        font-size: 1.85rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        color: #f8fafc;
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin: 0;
    }
    
    .hero-sub {
        font-size: 0.88rem;
        color: #94a3b8;
        margin-top: 0.35rem;
        font-weight: 400;
    }

    /* KPI Grid Cards */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(6, 1fr);
        gap: 0.75rem;
        margin-bottom: 1rem;
    }

    .kpi-card {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 10px;
        padding: 0.75rem 0.9rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        transition: all 0.2s ease-in-out;
    }
    .kpi-card:hover {
        border-color: #06b6d4;
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(6, 182, 212, 0.15);
    }
    
    .kpi-label {
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #9ca3af;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        margin-bottom: 0.2rem;
    }
    
    .kpi-val {
        font-size: 1.35rem;
        font-weight: 800;
        color: #f9fafb;
        font-family: 'JetBrains Mono', monospace;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    
    .kpi-unit {
        font-size: 0.75rem;
        color: #6b7280;
        margin-left: 0.2rem;
        font-family: 'Inter', sans-serif;
    }

    /* Surface Panel */
    .glass-panel {
        background: linear-gradient(145deg, #111827 0%, #0b1120 100%);
        border-radius: 12px;
        padding: 1.15rem 1.35rem;
        border: 1px solid #1f2937;
        box-shadow: 0 4px 20px rgba(0,0,0,0.25);
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
        background: rgba(6, 182, 212, 0.08);
        border: 1px solid rgba(6, 182, 212, 0.25);
        border-radius: 8px;
        padding: 0.65rem 1rem;
        font-size: 0.83rem;
        color: #a5f3fc;
        margin-bottom: 1rem;
    }

    /* Powertrain Flow Diagram */
    .flow-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: #030712;
        border-radius: 10px;
        padding: 0.85rem 1.15rem;
        border: 1px solid #1f2937;
        margin-top: 0.75rem;
    }
    .flow-node {
        text-align: center;
        padding: 0.35rem 0.5rem;
        border-radius: 8px;
        background: #111827;
        border: 1px solid #374151;
        font-size: 0.72rem;
        font-weight: 700;
        color: #9ca3af;
        min-width: 0;
        white-space: nowrap;
    }
    .flow-node-active {
        background: #0f766e;
        border-color: #14b8a6;
        color: #ffffff;
        box-shadow: 0 0 12px rgba(20, 184, 166, 0.5);
    }
    .flow-arrow {
        font-size: 1.05rem;
        color: #374151;
        font-weight: bold;
    }
    .flow-arrow-active {
        color: #38bdf8;
        text-shadow: 0 0 8px #38bdf8;
    }

    /* Gallery Card Styling */
    .gallery-card {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 4px 14px rgba(0,0,0,0.2);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .gallery-card:hover {
        border-color: #06b6d4;
        transform: translateY(-2px);
    }
    .gallery-title {
        font-size: 0.95rem;
        font-weight: 700;
        color: #f3f4f6;
        margin-bottom: 0.35rem;
    }
    .gallery-desc {
        font-size: 0.8rem;
        color: #9ca3af;
        margin-bottom: 0.75rem;
        line-height: 1.4;
    }
</style>
""", unsafe_allow_html=True)

# --- LOAD ASSETS & DATASETS ---
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

@st.cache_data
def load_kpi_summaries():
    summaries = {}
    for cycle in ["bangalore_urban", "wltp_mixed", "wltp_urban"]:
        json_path = ROOT_DIR / "outputs" / f"kpi_report_{cycle}.json"
        if json_path.exists():
            with open(json_path, "r") as f:
                summaries[cycle] = json.load(f)
    return summaries

engine = load_decision_engine()
df_all = load_telemetry_dataset()
kpi_data = load_kpi_summaries()

# --- SCENARIO DATA EXTRACTOR ---
def get_scenario_data(df: pd.DataFrame, scenario_choice: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    
    if scenario_choice == "Scenario 1: Normal Urban Driving (Healthy SOC)":
        sub = df[(df["battery_soc"] >= 45.0) & (df["speed"] > 10.0) & (df["speed"] < 60.0)]
        if len(sub) >= 50:
            return sub.iloc[100:150].reset_index(drop=True)
        return df.iloc[0:50].reset_index(drop=True)

    elif scenario_choice == "Scenario 2: Low SOC + High Demand (Safety Transition)":
        sub = df[(df["battery_soc"] <= 25.0) & (df["power_required_kw"] >= 15.0)]
        if len(sub) >= 50:
            return sub.iloc[0:50].reset_index(drop=True)
        sub2 = df[df["battery_soc"] <= 28.0]
        if len(sub2) >= 50:
            return sub2.iloc[0:50].reset_index(drop=True)
        return df.iloc[200:250].reset_index(drop=True)

    elif scenario_choice == "Scenario 3: Braking / Regeneration (Kinetic Recuperation)":
        sub = df[(df["power_required_kw"] < 0) | (df["braking"] == True) | (df["braking"] == 1)]
        if len(sub) >= 50:
            return sub.iloc[0:50].reset_index(drop=True)
        return df.iloc[300:350].reset_index(drop=True)

    else:
        return df.iloc[0:100].reset_index(drop=True)

# --- PRECOMPUTE INFERENCE ---
@st.cache_data
def precompute_scenario_simulation(scenario_name: str):
    df_slice = get_scenario_data(df_all, scenario_name)
    if df_slice.empty:
        return []
    
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

# --- NAVIGATION & VIEW MODE ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/lightning-bolt.png", width=52)
    st.title("PHEX-HEV Control")
    st.markdown("---")

    nav_view = st.radio(
        "🗂️ Engineering Sections",
        [
            "⚡ Live Telemetry & Simulation Replay",
            "🔮 ML Demand Forecasting & Benchmarks",
            "🔋 Battery Degradation & Thermal Safety",
            "🚗 Powertrain, Route & Traffic Analytics",
            "🏛️ System Architecture & KPI Reports"
        ],
        index=0
    )
    st.markdown("---")

# ==============================================================================
# VIEW 1: LIVE TELEMETRY & SIMULATION REPLAY
# ==============================================================================
if nav_view == "⚡ Live Telemetry & Simulation Replay":
    def on_scenario_change():
        st.session_state.current_step = 0
        st.session_state.is_playing = False

    with st.sidebar:
        scenario_option = st.selectbox(
            "🎯 Viva Scenario",
            [
                "Scenario 1: Normal Urban Driving (Healthy SOC)",
                "Scenario 2: Low SOC + High Demand (Safety Transition)",
                "Scenario 3: Braking / Regeneration (Kinetic Recuperation)",
                "Full Cycle (First 100 Timesteps)"
            ],
            on_change=on_scenario_change
        )

        sim_data = precompute_scenario_simulation(scenario_option)
        total_steps = len(sim_data)

        st.markdown("### ⏯️ Simulation Controls")
        
        if "current_step" not in st.session_state:
            st.session_state.current_step = 0
        if "is_playing" not in st.session_state:
            st.session_state.is_playing = False
        if "advance_step" not in st.session_state:
            st.session_state.advance_step = False

        if st.session_state.advance_step:
            st.session_state.advance_step = False
            if st.session_state.current_step < total_steps - 1:
                st.session_state.current_step += 1
            else:
                st.session_state.is_playing = False

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

    # Hero Header
    st.markdown("""
    <div class="hero-header">
        <div class="hero-title">
            <span>⚡ PHEX HEV Energy Management System</span>
        </div>
        <div class="hero-sub">
            Real-Time Drive Cycle Replay • Multi-Step Demand Forecaster • Hybrid Decision Engine & Thermal Safety Envelope
        </div>
    </div>
    """, unsafe_allow_html=True)

    curr = sim_data[st.session_state.current_step]

    scenario_descriptions = {
        "Scenario 1: Normal Urban Driving (Healthy SOC)": "🏙️ **Urban Driving (SOC ≥ 45%)**: Demonstrates pure EV priority in congested stop-and-go traffic to eliminate tailpipe emissions while maintaining smooth electric torque.",
        "Scenario 2: Low SOC + High Demand (Safety Transition)": "🔋 **Depleted Battery (SOC ≤ 25%) & High Load**: Demonstrates forced engine charging, load-leveling, and physical safety overrides to prevent irreversible battery degradation.",
        "Scenario 3: Braking / Regeneration (Kinetic Recuperation)": "🔄 **Deceleration & Braking Events**: Demonstrates kinetic energy recovery through regenerative braking torque limits to recharge the high-voltage pack.",
        "Full Cycle (First 100 Timesteps)": "📊 **Composite Drive Cycle**: Mixed drive cycle covering urban, highway, and grade transitions."
    }
    st.markdown(f'<div class="scenario-fact">{scenario_descriptions.get(scenario_option, "")}</div>', unsafe_allow_html=True)

    # KPI Metric Cards
    p_color = "#ef4444" if curr["power_required_kw"] > 20 else ("#10b981" if curr["power_required_kw"] < 0 else "#f9fafb")
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

    # Decision Engine + Demand Forecaster Panels
    col_decision, col_forecaster = st.columns([1.25, 1.0])

    with col_decision:
        mode_val = curr["mode"]
        mode_class = f"mode-{mode_val}"
        conf_pct = curr["confidence"] * 100.0 if curr["confidence"] <= 1.0 else curr["confidence"]
        safety_badge = '<span class="safety-badge-clear">✅ SAFETY CLEAR</span>' if curr["safety_ok"] else f'<span class="safety-badge-override">⚠️ OVERRIDE: {curr["safety_reason"]}</span>'

        ice_active = mode_val in ["ENGINE_ONLY", "ENGINE_CHARGE", "HYBRID_ASSIST"]
        gen_active = mode_val in ["ENGINE_CHARGE"]
        bat_active = mode_val in ["BATTERY_ONLY", "HYBRID_ASSIST", "ENGINE_CHARGE", "REGEN"]
        mot_active = mode_val in ["BATTERY_ONLY", "HYBRID_ASSIST", "REGEN"]
        
        ice_cls = "flow-node-active" if ice_active else ""
        gen_cls = "flow-node-active" if gen_active else ""
        bat_cls = "flow-node-active" if bat_active else ""
        mot_cls = "flow-node-active" if mot_active else ""

        st.markdown(f"""
        <div class="glass-panel">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.6rem;">
                <span style="font-size: 0.8rem; font-weight: 700; color: #9ca3af; text-transform: uppercase; letter-spacing: 0.05em;">Hybrid Decision Core</span>
                {safety_badge}
            </div>
            <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 0.75rem;">
                <div class="mode-badge {mode_class}">{mode_val}</div>
                <div>
                    <div style="font-size: 0.75rem; color: #9ca3af;">Source Engine</div>
                    <div style="font-weight: 700; color: #38bdf8; font-size: 0.9rem;">{curr["source"]} (Rule: {curr["rule_triggered"]})</div>
                </div>
            </div>
            <div style="margin-bottom: 0.65rem;">
                <div style="display: flex; justify-content: space-between; font-size: 0.8rem; margin-bottom: 0.2rem;">
                    <span style="color: #cbd5e1;">Model Classification Confidence</span>
                    <span style="font-weight: 700; color: #38bdf8;">{conf_pct:.1f}%</span>
                </div>
                <div style="background: #1f2937; border-radius: 4px; height: 6px; overflow: hidden;">
                    <div style="background: #38bdf8; width: {conf_pct}%; height: 100%; border-radius: 4px;"></div>
                </div>
            </div>
            <div style="background: rgba(3, 7, 18, 0.7); padding: 0.55rem 0.85rem; border-radius: 6px; border-left: 3px solid #38bdf8; font-size: 0.8rem; color: #cbd5e1;">
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
        <div class="glass-panel">
            <div style="font-size: 0.8rem; font-weight: 700; color: #9ca3af; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.75rem;">
                🔮 ML Demand Forecaster (Horizon t+k)
            </div>
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.5rem; margin-bottom: 0.75rem;">
                <div style="background: #030712; padding: 0.6rem; border-radius: 8px; border: 1px solid #1f2937; text-align: center;">
                    <div style="font-size: 0.7rem; color: #9ca3af; font-weight: bold;">PRED POWER</div>
                    <div style="font-size: 1.15rem; font-weight: 800; color: #f9fafb; font-family: 'JetBrains Mono', monospace;">{curr["pred_power"]:.1f}<span style="font-size: 0.7rem; color: #6b7280;"> kW</span></div>
                    <div style="font-size: 0.75rem; color: {'#10b981' if p_delta >= 0 else '#ef4444'}; font-weight: 600;">{p_delta:+.1f} kW</div>
                </div>
                <div style="background: #030712; padding: 0.6rem; border-radius: 8px; border: 1px solid #1f2937; text-align: center;">
                    <div style="font-size: 0.7rem; color: #9ca3af; font-weight: bold;">PRED SPEED</div>
                    <div style="font-size: 1.15rem; font-weight: 800; color: #f9fafb; font-family: 'JetBrains Mono', monospace;">{curr["pred_speed"]:.1f}<span style="font-size: 0.7rem; color: #6b7280;"> km/h</span></div>
                    <div style="font-size: 0.75rem; color: {'#10b981' if v_delta >= 0 else '#ef4444'}; font-weight: 600;">{v_delta:+.1f} km/h</div>
                </div>
                <div style="background: #030712; padding: 0.6rem; border-radius: 8px; border: 1px solid #1f2937; text-align: center;">
                    <div style="font-size: 0.7rem; color: #9ca3af; font-weight: bold;">PRED ACCEL</div>
                    <div style="font-size: 1.15rem; font-weight: 800; color: #f9fafb; font-family: 'JetBrains Mono', monospace;">{curr["pred_accel"]:.2f}<span style="font-size: 0.7rem; color: #6b7280;"> m/s²</span></div>
                    <div style="font-size: 0.75rem; color: {'#10b981' if a_delta >= 0 else '#ef4444'}; font-weight: 600;">{a_delta:+.2f} m/s²</div>
                </div>
            </div>
            <div style="font-size: 0.78rem; color: #9ca3af; line-height: 1.4; background: #030712; padding: 0.6rem 0.8rem; border-radius: 6px; border: 1px solid #1f2937;">
                💡 <em>Multi-lag Random Forest regressors anticipate torque demands prior to uphill grades & highway accelerations, eliminating engine lag.</em>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 1rem'></div>", unsafe_allow_html=True)

    # Dynamic Plots
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

    with tab_overview:
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            fig_speed = go.Figure()
            fig_speed.add_trace(go.Scatter(x=all_steps, y=all_speeds, mode="lines", name="Full Profile", line=dict(color="#374151", width=1.5, dash="dot")))
            fig_speed.add_trace(go.Scatter(x=hist_steps, y=speeds, mode="lines+markers", name="Actual Speed", line=dict(color="#38bdf8", width=2.5), marker=dict(size=4)))
            fig_speed.add_trace(go.Scatter(x=hist_steps, y=pred_speeds, mode="lines", name="Forecast (t+k)", line=dict(color="#f59e0b", width=2, dash="dash")))
            fig_speed.add_vline(x=curr_idx, line_width=2, line_dash="solid", line_color="#ec4899")
            fig_speed.update_layout(
                title=dict(text="Vehicle Speed Profile (km/h)", font=dict(size=14, color="#f9fafb")),
                xaxis=dict(title="Timestep (s)", gridcolor="#1f2937"),
                yaxis=dict(title="Speed (km/h)", gridcolor="#1f2937"),
                template="plotly_dark",
                margin=dict(l=40, r=20, t=45, b=30),
                height=290,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)")
            )
            st.plotly_chart(fig_speed, use_container_width=True, config={"displayModeBar": False})

        with col_p2:
            fig_pwr = go.Figure()
            fig_pwr.add_trace(go.Scatter(x=all_steps, y=all_powers, mode="lines", name="Full Demand", line=dict(color="#374151", width=1.5, dash="dot")))
            fig_pwr.add_trace(go.Scatter(x=hist_steps, y=powers, mode="lines+markers", name="Actual Power", line=dict(color="#34d399", width=2.5), marker=dict(size=4)))
            fig_pwr.add_trace(go.Scatter(x=hist_steps, y=pred_powers, mode="lines", name="Forecast (t+k)", line=dict(color="#fb923c", width=2, dash="dash")))
            fig_pwr.add_vline(x=curr_idx, line_width=2, line_dash="solid", line_color="#ec4899")
            fig_pwr.update_layout(
                title=dict(text="Wheel Power Demand (kW)", font=dict(size=14, color="#f9fafb")),
                xaxis=dict(title="Timestep (s)", gridcolor="#1f2937"),
                yaxis=dict(title="Power (kW)", gridcolor="#1f2937"),
                template="plotly_dark",
                margin=dict(l=40, r=20, t=45, b=30),
                height=290,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)")
            )
            st.plotly_chart(fig_pwr, use_container_width=True, config={"displayModeBar": False})

        mode_color_map = {
            "BATTERY_ONLY": "#10b981",
            "HYBRID_ASSIST": "#3b82f6",
            "ENGINE_CHARGE": "#eab308",
            "ENGINE_ONLY": "#f97316",
            "REGEN": "#14b8a6",
            "IDLE_STOP": "#6b7280"
        }
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
            xaxis=dict(gridcolor="#1f2937"),
            yaxis=dict(gridcolor="#1f2937")
        )
        st.plotly_chart(fig_mode, use_container_width=True, config={"displayModeBar": False})

    with tab_power:
        recup_powers = [max(0.0, -p) for p in powers]
        fig_recup = go.Figure()
        fig_recup.add_trace(go.Bar(x=hist_steps, y=recup_powers, name="Regen Recuperation (kW)", marker_color="#10b981"))
        fig_recup.add_trace(go.Scatter(x=hist_steps, y=powers, mode="lines", name="Wheel Demand (kW)", line=dict(color="#38bdf8", width=2)))
        fig_recup.update_layout(
            title=dict(text="Kinetic Energy Recuperation & Powertrain Demand Profile", font=dict(size=14)),
            xaxis=dict(title="Timestep (s)", gridcolor="#1f2937"),
            yaxis=dict(title="Power (kW)", gridcolor="#1f2937"),
            template="plotly_dark",
            height=350,
            margin=dict(l=40, r=20, t=45, b=30),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)")
        )
        st.plotly_chart(fig_recup, use_container_width=True, config={"displayModeBar": False})

    with tab_battery:
        fig_soc = go.Figure()
        fig_soc.add_trace(go.Scatter(x=all_steps, y=all_socs, mode="lines", name="Full Scenario SOC", line=dict(color="#374151", width=1.5, dash="dot")))
        fig_soc.add_trace(go.Scatter(x=hist_steps, y=socs, mode="lines+markers", name="Battery SOC (%)", line=dict(color="#10b981", width=2.5), marker=dict(size=4)))
        fig_soc.add_hline(y=SOC_LOW, line_dash="dash", line_color="#f59e0b", annotation_text=f"Low SOC Floor ({SOC_LOW:.0f}%)", annotation_position="bottom right")
        fig_soc.add_hline(y=SOC_CRITICAL, line_dash="dash", line_color="#ef4444", annotation_text=f"Critical SOC ({SOC_CRITICAL:.0f}%)", annotation_position="bottom right")
        fig_soc.add_vline(x=curr_idx, line_width=2, line_dash="solid", line_color="#ec4899")
        fig_soc.update_layout(
            title=dict(text="High-Voltage Battery State of Charge (SOC %) & Physical Envelope", font=dict(size=14)),
            xaxis=dict(title="Timestep (s)", gridcolor="#1f2937"),
            yaxis=dict(title="SOC (%)", range=[0, 100], gridcolor="#1f2937"),
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
        key_cols = ["mode", "confidence", "rule_triggered", "speed", "power_required_kw", "battery_soc", "battery_temp", "acceleration", "traffic_condition", "current_segment", "pred_power", "pred_speed", "reason"]
        df_raw_hist = pd.DataFrame(sim_data[:curr_idx + 1]).tail(10)
        ordered_cols = [c for c in key_cols if c in df_raw_hist.columns] + [c for c in df_raw_hist.columns if c not in key_cols]
        df_hist_view = df_raw_hist[ordered_cols].round(2)
        st.dataframe(df_hist_view, use_container_width=True)

    # Play loop
    if st.session_state.get("is_playing", False):
        if st.session_state.current_step < total_steps - 1:
            time.sleep(max(0.04, 0.35 / replay_speed))
            st.session_state.advance_step = True
            st.rerun()
        else:
            st.session_state.is_playing = False
            st.session_state.advance_step = False
            st.rerun()

# ==============================================================================
# VIEW 2: ML DEMAND FORECASTING & BENCHMARKS
# ==============================================================================
elif nav_view == "🔮 ML Demand Forecasting & Benchmarks":
    st.markdown("""
    <div class="hero-header">
        <div class="hero-title">
            <span>🔮 Machine Learning & Demand Forecasting Diagnostics</span>
        </div>
        <div class="hero-sub">
            Multi-Model Evaluation • Temporal Lag Feature Importances • Multi-Step Horizon Diagnostics
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab_fc, tab_clf, tab_features = st.tabs([
        "📈 Demand Forecasters (Horizon t+k)",
        "🎯 EMS Classifier Benchmarks",
        "🌟 Feature Importance & Correlation"
    ])

    with tab_fc:
        st.markdown("### 🔮 Multi-Lag Random Forest Regressors (Power, Speed, Acceleration)")
        c1, c2 = st.columns(2)
        with c1:
            p_img = ROOT_DIR / "outputs" / "visualizations" / "power_forecast.png"
            if p_img.exists():
                st.image(str(p_img), caption="Power Demand Forecasting: Actual vs Predicted (kW)", use_container_width=True)
        with c2:
            s_img = ROOT_DIR / "outputs" / "visualizations" / "speed_forecast.png"
            if s_img.exists():
                st.image(str(s_img), caption="Vehicle Speed Forecaster Tracking (km/h)", use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            a_img = ROOT_DIR / "outputs" / "visualizations" / "accel_forecast.png"
            if a_img.exists():
                st.image(str(a_img), caption="Acceleration Trajectory Lookahead", use_container_width=True)
        with c4:
            err_img = ROOT_DIR / "outputs" / "visualizations" / "forecast_error_histogram.png"
            if err_img.exists():
                st.image(str(err_img), caption="Forecasting Residuals & Error Distribution", use_container_width=True)

    with tab_clf:
        st.markdown("### 🏆 EMS Classifier Performance (GBM vs RF vs XGBoost)")
        c1, c2 = st.columns(2)
        with c1:
            cm_img = ROOT_DIR / "outputs" / "visualizations" / "confusion_matrix.png"
            if cm_img.exists():
                st.image(str(cm_img), caption="Multi-Class Decision Confusion Matrix", use_container_width=True)
        with c2:
            m_comp = ROOT_DIR / "outputs" / "visualizations" / "model_comparison.png"
            if m_comp.exists():
                st.image(str(m_comp), caption="Model Accuracy, Precision & F1 Comparison", use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            cr_img = ROOT_DIR / "outputs" / "visualizations" / "classification_report.png"
            if cr_img.exists():
                st.image(str(cr_img), caption="Detailed Class-Wise Classification Metrics", use_container_width=True)
        with c4:
            conf_img = ROOT_DIR / "outputs" / "visualizations" / "prediction_confidence.png"
            if conf_img.exists():
                st.image(str(conf_img), caption="Inference Prediction Confidence Distribution", use_container_width=True)

    with tab_features:
        st.markdown("### 🌟 Feature Ranking & Correlation Maps")
        c1, c2 = st.columns(2)
        with c1:
            fi_img = ROOT_DIR / "outputs" / "visualizations" / "feature_importance.png"
            if fi_img.exists():
                st.image(str(fi_img), caption="Gini / Permutation Feature Importance Ranking", use_container_width=True)
        with c2:
            fc_img = ROOT_DIR / "outputs" / "visualizations" / "feature_correlation.png"
            if fc_img.exists():
                st.image(str(fc_img), caption="Telemetry & Temporal Lag Cross-Correlation Heatmap", use_container_width=True)

# ==============================================================================
# VIEW 3: BATTERY DEGRADATION & THERMAL SAFETY
# ==============================================================================
elif nav_view == "🔋 Battery Degradation & Thermal Safety":
    st.markdown("""
    <div class="hero-header">
        <div class="hero-title">
            <span>🔋 High-Voltage Battery Degradation & Safety Envelope</span>
        </div>
        <div class="hero-sub">
            NASA Battery Aging & Voltage Sag • Bangalore Ambient Thermal Stress • Physical Hard Limits
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab_nasa, tab_safety, tab_regen = st.tabs([
        "🔬 NASA Battery Degradation Physics",
        "🛡️ Safety Layer & Thermal Overrides",
        "⚡ Kinetic Energy Recovery"
    ])

    with tab_nasa:
        st.markdown("### 🧪 Experimental NASA Capacity Degradation & Voltage Sag")
        c1, c2 = st.columns(2)
        with c1:
            cap_img = ROOT_DIR / "reports" / "figures" / "nasa_capacity_degradation.png"
            if cap_img.exists():
                st.image(str(cap_img), caption="NASA Li-ion Capacity Fade vs Cycle Number", use_container_width=True)
        with c2:
            sag_img = ROOT_DIR / "reports" / "figures" / "nasa_voltage_sag.png"
            if sag_img.exists():
                st.image(str(sag_img), caption="High Discharge Rate Voltage Sag Dynamics", use_container_width=True)

    with tab_safety:
        st.markdown("### 🛡️ Deterministic Safety Envelopes")
        c1, c2 = st.columns(2)
        with c1:
            soc_ov = ROOT_DIR / "outputs" / "visualizations" / "critical_soc_override.png"
            if soc_ov.exists():
                st.image(str(soc_ov), caption="Critical SOC Floor (15%) Mandatory Engine Charge Trigger", use_container_width=True)
        with c2:
            therm_ov = ROOT_DIR / "outputs" / "visualizations" / "thermal_override.png"
            if therm_ov.exists():
                st.image(str(therm_ov), caption="High Temperature (40°C) Power Derating & Thermal Override", use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            sc_img = ROOT_DIR / "outputs" / "visualizations" / "safety_override_counts.png"
            if sc_img.exists():
                st.image(str(sc_img), caption="Safety Override Interventions Count Across Test Fleet", use_container_width=True)
        with c4:
            tb_img = ROOT_DIR / "outputs" / "visualizations" / "battery_temp_distribution.png"
            if tb_img.exists():
                st.image(str(tb_img), caption="Battery Temperature Distribution Under Indian Traffic", use_container_width=True)

    with tab_regen:
        st.markdown("### ⚡ Regenerative Braking & SOC Planning")
        c1, c2 = st.columns(2)
        with c1:
            rg_img = ROOT_DIR / "outputs" / "visualizations" / "regen_energy_recovery.png"
            if rg_img.exists():
                st.image(str(rg_img), caption="Kinetic Energy Recuperation (MJ) Across Urban Routes", use_container_width=True)
        with c2:
            sp_img = ROOT_DIR / "outputs" / "visualizations" / "soc_planner_behavior.png"
            if sp_img.exists():
                st.image(str(sp_img), caption="Predictive Route-Aware SOC Planner Trajectories", use_container_width=True)

# ==============================================================================
# VIEW 4: POWERTRAIN, ROUTE & TRAFFIC ANALYTICS
# ==============================================================================
elif nav_view == "🚗 Powertrain, Route & Traffic Analytics":
    st.markdown("""
    <div class="hero-header">
        <div class="hero-title">
            <span>🚗 Powertrain Efficiency & Traffic Intelligence</span>
        </div>
        <div class="hero-sub">
            Internal Combustion Engine Efficiency (BSFC) • Bangalore Traffic Adaptation • Mode Transitions
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab_ice, tab_traffic, tab_route = st.tabs([
        "⚙️ Engine BSFC & Power Distribution",
        "🚦 Traffic Intelligence & Bangalore Adaptation",
        "🗺️ Route Segment Dynamics"
    ])

    with tab_ice:
        c1, c2 = st.columns(2)
        with c1:
            eng_img = ROOT_DIR / "outputs" / "visualizations" / "engine_efficiency_map.png"
            if eng_img.exists():
                st.image(str(eng_img), caption="Engine Operating Sweet Spot / Brake Specific Fuel Consumption (BSFC) Map", use_container_width=True)
        with c2:
            pwm_img = ROOT_DIR / "outputs" / "visualizations" / "power_vs_mode.png"
            if pwm_img.exists():
                st.image(str(pwm_img), caption="Power Distribution by Active EMS Operating Mode", use_container_width=True)

    with tab_traffic:
        c1, c2 = st.columns(2)
        with c1:
            hm_img = ROOT_DIR / "outputs" / "visualizations" / "traffic_ems_heatmap.png"
            if hm_img.exists():
                st.image(str(hm_img), caption="Traffic Density vs EMS Mode Selection Heatmap", use_container_width=True)
        with c2:
            tvm_img = ROOT_DIR / "outputs" / "visualizations" / "traffic_vs_mode.png"
            if tvm_img.exists():
                st.image(str(tvm_img), caption="Traffic State vs EMS Operating Mode Split", use_container_width=True)

    with tab_route:
        c1, c2 = st.columns(2)
        with c1:
            bng_img = ROOT_DIR / "outputs" / "visualizations" / "bangalore_route_profile.png"
            if bng_img.exists():
                st.image(str(bng_img), caption="Bangalore Heavy Traffic Speed & Elevation Profile", use_container_width=True)
        with c2:
            rvm_img = ROOT_DIR / "outputs" / "visualizations" / "route_vs_mode.png"
            if rvm_img.exists():
                st.image(str(rvm_img), caption="Route Classification (Arterial / Highway / Urban) vs EMS Mode", use_container_width=True)

# ==============================================================================
# VIEW 5: SYSTEM ARCHITECTURE & KPI BENCHMARKS
# ==============================================================================
elif nav_view == "🏛️ System Architecture & KPI Reports":
    st.markdown("""
    <div class="hero-header">
        <div class="hero-title">
            <span>🏛️ System Architecture & Viva KPI Benchmark Reports</span>
        </div>
        <div class="hero-sub">
            Modular Micro-Architecture • Deterministic State Machine Flow • Standardized WLTP vs Urban Benchmarks
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab_arch, tab_kpi, tab_audit = st.tabs([
        "📐 System Architecture & Flowcharts",
        "📊 Drive Cycle KPI Reports",
        "📄 Viva Audit & Compliance"
    ])

    with tab_arch:
        st.markdown("### 📐 Master System Architecture & Team Module Breakdown")
        c1, c2 = st.columns(2)
        with c1:
            arch_img = ROOT_DIR / "outputs" / "visualizations" / "architecture_master.png"
            if arch_img.exists():
                st.image(str(arch_img), caption="Master End-to-End System Pipeline & Signal Flow", use_container_width=True)
        with c2:
            flow_img = ROOT_DIR / "outputs" / "visualizations" / "decision_flowchart.png"
            if flow_img.exists():
                st.image(str(flow_img), caption="Hierarchical Deterministic Rule Fallback Flowchart", use_container_width=True)

        team_img = ROOT_DIR / "outputs" / "visualizations" / "team_architecture.png"
        if team_img.exists():
            st.image(str(team_img), caption="Team Module Responsibilities & Inter-Layer Contract Mapping", use_container_width=True)

    with tab_kpi:
        st.markdown("### 📊 Standardized KPI Benchmark Summary")
        if kpi_data:
            col_k1, col_k2, col_k3 = st.columns(3)
            with col_k1:
                st.markdown("""
                <div class="glass-panel">
                    <h4 style="color: #38bdf8; margin: 0 0 0.5rem 0;">🏙️ Bangalore Urban Cycle</h4>
                    <p style="font-size: 0.85rem; color: #9ca3af;">Extreme stop-and-go congestion & EV priority.</p>
                """, unsafe_allow_html=True)
                st.json(kpi_data.get("bangalore_urban", {}))
                st.markdown("</div>", unsafe_allow_html=True)

            with col_k2:
                st.markdown("""
                <div class="glass-panel">
                    <h4 style="color: #34d399; margin: 0 0 0.5rem 0;">🛣️ WLTP Mixed Cycle</h4>
                    <p style="font-size: 0.85rem; color: #9ca3af;">Combined urban, extra-urban, and highway drive cycle.</p>
                """, unsafe_allow_html=True)
                st.json(kpi_data.get("wltp_mixed", {}))
                st.markdown("</div>", unsafe_allow_html=True)

            with col_k3:
                st.markdown("""
                <div class="glass-panel">
                    <h4 style="color: #facc15; margin: 0 0 0.5rem 0;">🚦 WLTP Urban Cycle</h4>
                    <p style="font-size: 0.85rem; color: #9ca3af;">Standardized European low-speed urban cycle.</p>
                """, unsafe_allow_html=True)
                st.json(kpi_data.get("wltp_urban", {}))
                st.markdown("</div>", unsafe_allow_html=True)

    with tab_audit:
        st.markdown("### 📋 Viva Verification & Architecture Audit")
        audit_file = ROOT_DIR / "outputs" / "audit_report.md"
        if audit_file.exists():
            with open(audit_file, "r", encoding="utf-8") as f:
                audit_md = f.read()
            st.markdown(audit_md)
        else:
            st.info("Audit report document not found.")
