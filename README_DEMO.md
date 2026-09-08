# ⚡ PHEX HEV Energy Management System — Viva Simulation Guide

## 1. How to Launch the Dashboard
Open your terminal in the project directory and run:

```bash
cd PHEX-HEV-EMS-main
streamlit run app.py
```

The interactive engineering dashboard will automatically open in your browser at `http://localhost:8501`.

---

## 2. Real Telemetry Dataset Being Replayed
The simulation replays real, physics-consistent telemetry rows directly from:
- `data/raw/synthetic_ems_dataset.csv` (20,000 recorded timesteps across multiple driver profiles, trip lengths, and route segments).

Three deterministic viva demonstration slices are provided:
1. **Scenario 1: Normal Urban Driving**: 50 real rows of moderate-speed urban/suburban driving with healthy battery SOC ($\ge 45\%$). Demonstrates EV priority in congested conditions and demand tracking.
2. **Scenario 2: Low SOC + High Demand**: 50 real rows under depleted battery conditions ($\text{SOC} \le 25\%$) and high power demand ($\ge 18\,\text{kW}$). Demonstrates safety overrides and engine charge-sustaining transitions.
3. **Scenario 3: Braking / Regeneration**: 50 real rows with negative wheel power demand ($P_{req} < 0$) and braking events. Demonstrates kinetic energy recuperation.

---

## 3. Existing Classes & Functions Reused
The simulation dashboard does **not** duplicate or rewrite any algorithms. It directly imports and orchestrates:

| Pipeline Stage | Existing Module & Class | Purpose |
| :--- | :--- | :--- |
| **Telemetry Input** | [`src/rule_ems.py`](file:///c:/Users/misha/OneDrive/Desktop/PHEX-HEV-EMS-main/PHEX-HEV-EMS-main/src/rule_ems.py) (`VehicleState`) | Dataclass snapshot representing vehicle state |
| **Feature Engineering** | [`src/preprocessing.py`](file:///c:/Users/misha/OneDrive/Desktop/PHEX-HEV-EMS-main/PHEX-HEV-EMS-main/src/preprocessing.py) (`engineer_features`, `engineer_temporal_features`) | Static flags, ratios, rolling statistics |
| **Demand Forecasting** | [`src/demand_forecaster.py`](file:///c:/Users/misha/OneDrive/Desktop/PHEX-HEV-EMS-main/PHEX-HEV-EMS-main/src/demand_forecaster.py) (`DemandForecaster`) | Multi-model temporal lag Random Forest regressors |
| **Route & Traffic Lookahead** | [`src/route_logic.py`](file:///c:/Users/misha/OneDrive/Desktop/PHEX-HEV-EMS-main/PHEX-HEV-EMS-main/src/route_logic.py) (`RouteManager`), [`src/predictive_controller.py`](file:///c:/Users/misha/OneDrive/Desktop/PHEX-HEV-EMS-main/PHEX-HEV-EMS-main/src/predictive_controller.py) | Lookahead distance, grades, predictive advisories |
| **Hybrid Decision Core** | [`src/decision_engine.py`](file:///c:/Users/misha/OneDrive/Desktop/PHEX-HEV-EMS-main/PHEX-HEV-EMS-main/src/decision_engine.py) (`HybridDecisionEngine`) | ML inference (`ems_best_model.pkl`) with confidence |
| **Safety Envelope** | [`src/safety_override.py`](file:///c:/Users/misha/OneDrive/Desktop/PHEX-HEV-EMS-main/PHEX-HEV-EMS-main/src/safety_override.py) (`SafetyOverrideLayer`) | Hard physical bounds on SOC, temperature, current |
| **Deterministic Rule Base** | [`src/rule_ems.py`](file:///c:/Users/misha/OneDrive/Desktop/PHEX-HEV-EMS-main/PHEX-HEV-EMS-main/src/rule_ems.py) (`RuleBasedEMS`) | Hierarchical fallback rule chain |

---

## 4. Dashboard Panels Explained
1. **Top Metric Cards**: Displays the current ground-truth vehicle state (Speed, Acceleration, Power Demand, SOC %, Battery Temp, Traffic Condition, Route Segment).
2. **ML Demand Forecaster Panel**: Displays the model's forward predictions ($P_{req,t+k}$, $v_{t+k}$, $a_{t+k}$) alongside prediction deltas.
3. **Hybrid EMS Decision Panel**: Prominently highlights the selected mode (`BATTERY_ONLY`, `HYBRID_ASSIST`, `ENGINE_CHARGE`, `ENGINE_ONLY`, `REGEN`, `IDLE_STOP`), model confidence percentage, triggered rule, and safety layer status.
4. **Live Dynamic Plots**:
   - **Speed: Actual vs Forecasted** ($km/h$)
   - **Wheel Power Demand: Actual vs Forecasted** ($kW$)
   - **Battery State of Charge** ($SOC\,\%$) with Critical (15%) and Low (25%) threshold lines
   - **EMS Operating Mode Transition Timeline**
   - **Energy Recuperation Power** ($kW$) during braking events

---

## 5. Viva Explanation of the Simulation Architecture
> *"This application is a **software-based EMS proof-of-concept and telemetry replay platform**. Rather than running hardcoded values or mock simulations, it streams real drive-cycle telemetry row-by-row into our trained machine learning pipeline. Each timestep undergoes temporal feature extraction, multi-step demand forecasting via Random Forest regressors, route lookahead advisory querying, and classification by our Hybrid Decision Engine. The decision is validated against physical battery safety constraints before commanding the powertrain."*
