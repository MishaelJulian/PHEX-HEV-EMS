# Phase 2 Extension: Supervisor-Guided Energy Management System (EMS) Walkthrough

I have successfully completed all implementation and verification steps for Phase 2 of the Plug-in Hybrid Electric Vehicle (PHEV) Energy Management System (EMS). The system has transitioned to a canonical, enum-driven `EMSMode` taxonomy, and incorporates advanced supervisory decision logic (traffic prediction, SOC planning, regenerative braking power tapering, and battery health monitoring).

---

## 1. What Was Done

All 5 new modules and extensions to existing modules have been implemented and validated:

1. **Enum Transition & Configuration**:
   - Built a single canonical `EMSMode` enum in [ems_modes.py](file:///c:/Users/misha/OneDrive/Desktop/PHEX-HEV-EMS-main/PHEX-HEV-EMS-main/src/ems_modes.py) to represent vehicle modes: `EV`, `ICE`, `HYBRID_ASSIST`, `REGEN`, `CHARGE_SUSTAIN`, and `CHARGE_DEPLETING`.
   - Updated existing files ([rule_ems.py](file:///c:/Users/misha/OneDrive/Desktop/PHEX-HEV-EMS-main/PHEX-HEV-EMS-main/src/rule_ems.py), [ems_model.py](file:///c:/Users/misha/OneDrive/Desktop/PHEX-HEV-EMS-main/PHEX-HEV-EMS-main/src/ems_model.py), and [safety_override.py](file:///c:/Users/misha/OneDrive/Desktop/PHEX-HEV-EMS-main/PHEX-HEV-EMS-main/src/safety_override.py)) to map clean values, ensuring backwards compatibility for legacy string inputs.
   - Sourced all constants cleanly in [config.py](file:///c:/Users/misha/OneDrive/Desktop/PHEX-HEV-EMS-main/PHEX-HEV-EMS-main/src/config.py) (no magic numbers).

2. **Five New Supervisory Modules**:
   - [traffic_predictor.py](file:///c:/Users/misha/OneDrive/Desktop/PHEX-HEV-EMS-main/PHEX-HEV-EMS-main/src/traffic_predictor.py): Classifies traffic into `STOP_GO`, `CONGESTED`, or `FREE_FLOW` based on speed and standard deviations from a rolling window.
   - [engine_efficiency.py](file:///c:/Users/misha/OneDrive/Desktop/PHEX-HEV-EMS-main/PHEX-HEV-EMS-main/src/engine_efficiency.py): Modeled ICE efficiency as a Gaussian distribution centered at 2500 RPM, determining if a given operating point is efficient.
   - [regen_controller.py](file:///c:/Users/misha/OneDrive/Desktop/PHEX-HEV-EMS-main/PHEX-HEV-EMS-main/src/regen_controller.py): Computes regenerative power dynamically using deceleration magnitude and tapering linearly between 30% and 90% SOC.
   - [soc_planner.py](file:///c:/Users/misha/OneDrive/Desktop/PHEX-HEV-EMS-main/PHEX-HEV-EMS-main/src/soc_planner.py): Strategically plans SOC targets depending on upcoming segments (preserving charge for urban and stop-and-go zones).
   - [battery_health.py](file:///c:/Users/misha/OneDrive/Desktop/PHEX-HEV-EMS-main/PHEX-HEV-EMS-main/src/battery_health.py): Computes battery health degradation based on operating temperature, charging cycles, and excessive regenerative cycles.

3. **Time-Series Demand Forecasting Extension**:
   - [demand_forecaster.py](file:///c:/Users/misha/OneDrive/Desktop/PHEX-HEV-EMS-main/PHEX-HEV-EMS-main/src/demand_forecaster.py): Implemented a trend-projection feature (`estimate_future_demand`) that uses linear regression (`np.polyfit`) over recent telemetry to forecast power requirements.

4. **Integrated Decision Logic**:
   - [decision_engine.py](file:///c:/Users/misha/OneDrive/Desktop/PHEX-HEV-EMS-main/PHEX-HEV-EMS-main/src/decision_engine.py): Implemented the standalone `select_mode` function merging the predictive models, the traffic classification, and the SOC planner targets with a 9-rule fallback logic.

5. **Simulator and Scenarios**:
   - [simulator.py](file:///c:/Users/misha/OneDrive/Desktop/PHEX-HEV-EMS-main/PHEX-HEV-EMS-main/src/simulator.py): Configured 4 new scenario profiles (`congestion`, `aggressive`, `regen_heavy`, `mixed_route`) and added a standalone `run_scenario` loop.

---

## 2. What Was Tested & Validation Results

### 2.1 Automated Tests
A suite of 50 tests was executed successfully using `pytest`. The suite includes tests validating the newly introduced components alongside the original codebase logic.

```bash
python -m pytest
```

Output:
```text
============================= test session starts =============================
platform win32 -- Python 3.11.6, pytest-9.0.3, pluggy-1.6.0
collected 50 items

tests\test_battery_health.py ......                                      [ 12%]
tests\test_decision_engine.py ..                                         [ 16%]
tests\test_demand_forecaster.py .                                        [ 18%]
tests\test_ems_target_logic.py ...                                       [ 24%]
tests\test_engine_efficiency.py ......                                   [ 36%]
tests\test_feature_windows.py ..                                         [ 40%]
tests\test_generate_data.py ....                                         [ 48%]
tests\test_preprocessing.py ..                                           [ 52%]
tests\test_regen_controller.py ......                                    [ 64%]
tests\test_rule_ems.py ...                                               [ 70%]
tests\test_safety_override.py .                                          [ 72%]
tests\test_soc_planner.py ......                                         [ 84%]
tests\test_traffic_predictor.py ........                                 [100%]

============================= 50 passed in 2.65s ==============================
```

---

### 2.2 Model Training Pipeline
The ML training pipeline was successfully re-run on a fresh 20,000-row synthetic dataset incorporating all the new predictive features. The best model selected was the **Gradient Boosting Classifier**, achieving an accuracy of **92.6%** and an F1 Macro score of **86.5%**.

```bash
python run_ems.py --train
```

Performance comparison:
```text
+------------------+----------+----------+----------+-----------+
| Model            | Accuracy | F1 Macro | Train(s) | Pred(ms)  |
+------------------+----------+----------+----------+-----------+
| DecisionTree     | 0.899    | 0.851    | 0.21     | 2.8       |
| RandomForest     | 0.921    | 0.861    | 0.74     | 62.0      |
| GradientBoosting | 0.926    | 0.865    | 200.88   | 75.7      |
| XGBoost          | 0.926    | 0.863    | 1.88     | 25.3      |
+------------------+----------+----------+----------+-----------+
```

---

### 2.3 Scenario Evaluation
The safety overrides and predictive controller were verified under 9 simulated road scenarios. Every scenario correctly prioritized safety or optimized fuel utilization according to the rules:

```bash
python run_ems.py --evaluate
```

Verification log snippet:
- **Critical SOC**: Forced engine charge when SOC dropped to 12%.
- **Thermal Override**: Overruled ML to activate ICE mode when temperature spiked to 45°C.
- **Highway approaching preserve**: ML correctly leveraged the route advisory to pre-emptively preserve battery before merging onto the highway.

---

### 2.4 Real-time Simulation
The simulator operates in both rule-based and ML-hybrid modes correctly.
```bash
python run_ems.py --mode hybrid --rows 20 --delay 0
```

---

## 3. Visualization Gallery

The Phase 2 visualization suite generated 6 high-quality analysis figures illustrating the decision behaviors across the four main driving scenarios.

````carousel
![Traffic State Distribution](traffic_state_distribution.png)
<!-- slide -->
![EMS Mode Timeline](ems_mode_timeline.png)
<!-- slide -->
![SOC Planner Behavior](soc_planner_behavior.png)
<!-- slide -->
![Regen Energy Recovery](regen_energy_recovery.png)
<!-- slide -->
![Battery Health Trend](battery_health_trend.png)
<!-- slide -->
![Engine Efficiency Map](engine_efficiency_map.png)
````

Each plot aligns with key specifications:
- **Traffic State**: Demonstrates how urban scenarios exhibit heavy `CONGESTED` and `STOP_GO` conditions while highway is strictly `FREE_FLOW`.
- **Mode Timeline**: Charts real-time mode transitions (`EV` in urban phases, transitioning to `ICE` on the highway, and back to `EV`).
- **SOC Behavior**: Displays the target SOC profile computed by the `SOCPlanner` (solid blue line) and the actual battery tracking.
- **Regenerative Recovery**: Plots the cumulative energy recovered over time, validating the tapering controls when approaching full charge.
- **Battery Health**: Visualizes degradation patterns, tracking thermal, cycling, and charge-rate stress.
- **Engine Efficiency**: Confirms that ICE mode maps cleanly to the Gaussian curve around the 2500 RPM sweet spot.
