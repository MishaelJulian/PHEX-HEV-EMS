# PHEX Energy Management System - Gap Analysis

Below is the prioritized list of gaps identified in the existing codebase that must be addressed to fulfill the specification.

## Prioritized Gap Registry

| Gap ID | Description | Priority | Complexity | Affected Modules |
| :--- | :--- | :--- | :--- | :--- |
| **GAP-01** | Central configuration file `src/vehicle_config.py` is missing. Constants are hardcoded and mismatched across powertrain files. | **CRITICAL** | **LOW** | `src/vehicle_config.py`, all models |
| **GAP-02** | Transmission model `src/transmission_model.py` is missing, preventing physical mapping of engine/motor speeds and torques to wheels. | **CRITICAL** | **MEDIUM** | `src/transmission_model.py`, engine/motor models |
| **GAP-03** | `evaluation.py` and evaluation output files are missing. Existing evaluation script does not calculate required KPIs (CO2, costs, average efficiency). | **CRITICAL** | **MEDIUM** | `src/evaluation.py` |
| **GAP-04** | Visualization suite `src/visualization.py` is missing. Plots G.01 to G.13 do not exist or are incorrect. | **HIGH** | **HIGH** | `src/visualization.py`, outputs |
| **GAP-05** | Supervisory EMS Rules (C.RULE1 to C.RULE8) are not fully integrated or use wrong thresholds/parameters/logging codes. | **HIGH** | **HIGH** | `src/decision_engine.py`, `src/rule_ems.py` |
| **GAP-06** | Predictive layers (`TrafficPredictor`, `RouteLogic`, `DemandForecaster`, `AdvancedSOCPlanner`) have missing classes or mismatched API signatures. | **HIGH** | **HIGH** | Predictive EMS modules |
| **GAP-07** | Standard drive cycles (WLTP Urban, WLTP Mixed, Bangalore Urban) are not loaded or executed as actual time-series velocity profiles. | **HIGH** | **MEDIUM** | `src/drive_cycles.py`, `src/simulator.py` |
| **GAP-08** | Unit tests for missing modules (vehicle_config, transmission_model, evaluation, visualization) are missing. Test suite has incomplete coverage. | **MEDIUM** | **MEDIUM** | `tests/` |

---

## Detailed Gap Explanations

### GAP-01: Mismatched and Missing Vehicle Constants
- **Problem**: Battery capacity is 13.8 kWh instead of 15.0 kWh. Rolling resistance is 0.011 instead of 0.012. Engine peak power is 110 kW instead of 120 kW. Initial SOC is 80% instead of 70%.
- **Solution**: Create `src/vehicle_config.py` with all annotated constants, and update all models to import from it.

### GAP-02: Missing Transmission Model
- **Problem**: Powertrain models directly map motor torque/speed to wheels without transmission gears, differential ratio, or efficiency losses.
- **Solution**: Implement `TransmissionModel` mapping engine/motor shaft to wheel shaft, including gear selection and gear-dependent efficiency.

### GAP-03: Missing Evaluation Suite and KPIs
- **Problem**: `src/evaluate_ems.py` does not output the required json/txt reports, nor does it compute fuel economy based on physical fuel density, CO2 emissions (2.31 kg CO2/L), or operating cost estimates.
- **Solution**: Implement `src/evaluation.py` computing all 10 KPIs.

### GAP-04: Missing Visualization Suite
- **Problem**: Missing `generate_all_figures()` function yielding the 13 required charts G.01 to G.13.
- **Solution**: Implement `src/visualization.py` using matplotlib/seaborn to render and save all figures at 150+ DPI.

### GAP-05: Non-compliant Decision Engine Rules
- **Problem**: Decision engine does not log reason codes or route excess engine power to charging.
- **Solution**: Revise `select_mode` and `HybridDecisionEngine` in `src/decision_engine.py` to enforce the rules exactly, using the correct configuration thresholds and logging tags.

### GAP-06: API Mismatches in Predictive Layer
- **Problem**: Mismatches in class signatures between current implementations and required types/returns.
- **Solution**: Refactor `TrafficPredictor`, `RouteLogic`, `DemandForecaster`, and `AdvancedSOCPlanner` to match specified method signatures.

### GAP-07: Inactive Drive Cycles
- **Problem**: Drive cycles are hardcoded lists of segment states instead of time-resolved speed/grade profiles.
- **Solution**: Implement `src/drive_cycles.py` returning time-series arrays. Update `src/simulator.py` to interpolate speed/grade from these profiles.
