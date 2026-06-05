# Project PHEX — Systems Control Architecture & Data Flow

This document details the data flow and immutable control hierarchy of the Predictive Hybrid Energy Executive (PHEX) Energy Management System (EMS). 

## Immutable Control Hierarchy
The control logic flows strictly top-down to prevent lower-level component constraints from leaking into strategic planners:

```mermaid
graph TD
    %% Define Nodes
    A["1. Traffic & Route Predictor<br>(RouteLogic & TrafficPredictor)"]
    B["2. Demand Forecaster<br>(DemandForecaster)"]
    C["3. SOC Planner<br>(AdvancedSOCPlanner)"]
    D["4. Decision Engine<br>(select_mode Rule-Based & ML Engine)"]
    E["5. Safety Override Layer<br>(SafetyOverrideLayer)"]
    F["6. Final Mode Selection<br>(EMSMode output)"]
    G["7. Torque Split Controller<br>(TransmissionModel & TorqueSplitController)"]
    H["8. Powertrain Components<br>(Engine, Motor, Battery, Generator)"]

    %% Connect Nodes
    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
    G --> H

    %% Class Styling
    style A fill:#EFF6FF,stroke:#2563EB,stroke-width:2px
    style B fill:#FAF5FF,stroke:#8B5CF6,stroke-width:2px
    style C fill:#F5F3FF,stroke:#7C3AED,stroke-width:2px
    style D fill:#ECFDF5,stroke:#059669,stroke-width:2px
    style E fill:#FFFBEB,stroke:#D97706,stroke-width:2px
    style F fill:#FFF1F2,stroke:#E11D48,stroke-width:2px
    style G fill:#F8FAFC,stroke:#475569,stroke-width:2px
    style H fill:#E0F2FE,stroke:#0284C7,stroke-width:2px
```

## Detailed Data Flow Map

```mermaid
flowchart TD
    subgraph RouteContext ["1. Traffic / Route Prediction Layer"]
        A1["RouteLogic"] -->|"position, distance, segments"| A3["RouteInfo"]
        A2["TrafficPredictor"] -->|"avg speed, congestion prob"| A4["TrafficPrediction"]
    end

    subgraph DemandForecast ["2. Demand Forecasting Layer"]
        B1["DemandForecaster"] -->|"slice remaining cycle"| B2["DemandForecast (power, peak, confidence)"]
    end

    subgraph SOCPlanning ["3. Strategic SOC Planning Layer"]
        C1["AdvancedSOCPlanner"]
        A3 --> C1
        A4 --> C1
        B2 --> C1
        C1 -->|"SOC target, recom mode"| C2["SOCPlan"]
    end

    subgraph DecisionCore ["4-6. Mode Selection Core"]
        D1["select_mode (EMS Rules)"]
        C2 --> D1
        D2["BatteryHealthState"] --> D1
        D3["TrafficFeatures"] --> D1
        D1 -->|"EMSMode enum"| D4["Final Mode Selection"]
    end

    subgraph ActuatorSplit ["7-8. Physical Powertrain Execution"]
        E1["TransmissionModel"]
        E2["TorqueSplitController"]
        D4 --> E2
        E1 -->|"gear ratios & efficiency"| E2
        
        E2 -->|"engine torque"| F1["EngineState (ICE)"]
        E2 -->|"motor torque"| F2["MotorState (PMSM)"]
        F1 -->|"excess power"| F3["GeneratorState"]
        
        F2 -->|"regen power"| F4["BatteryState"]
        F3 -->|"charging power"| F4
    end
```
