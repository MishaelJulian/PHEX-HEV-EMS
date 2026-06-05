"""
architecture_map.py
Displays the architecture map for the Hybrid AI Energy Management System.
"""

def print_architecture():
    print(r"""
================================================================================================
                      ⚡ HYBRID AI ENERGY MANAGEMENT SYSTEM (EMS) ⚡                      
================================================================================================

  [ TEAM INPUTS ]                                         [ SUPERVISORY CONTROLLER ]

  1. Vehicle Dynamics (Alex) ──────┐                 ┌────────────────────────────────────┐
     - speed (km/h)                │                 │ 🧠 HybridDecisionEngine (AI)       │
     - acceleration (m/s²)         │                 │                                    │
     - power_req (kW)              │                 │   1. engineer_features()           │
     - braking (bool)              ├──────►[STATE]──►│      - high_power_flag             │
                                   │                 │      - regen_candidate             │
  2. Powertrain/Battery (Sonali) ──┤                 │      - load_intensity              │
     - battery_soc (%)             │                 │                                    │
     - battery_temp (°C)           │                 │   2. AI Inference (XGB/RF)         │
                                   │                 │      - predicts optimal mode       │
  3. Integration (Srishti) ────────┤                 │                                    │
     - current_segment             │                 │   3. Safety Override Check         │
     - next_segment                │                 │      - block unsafe AI actions     │
     - traffic_condition           │                 │                                    │
                                   │                 └─────────────────┬──────────────────┘
                                   │                                   │
                                   └────────────────────────[FALLBACK / OVERRIDE]
                                                                       │
                                                     ┌─────────────────▼──────────────────┐
                                                     │ ⚙️ RuleBasedEMS (Deterministic)    │
                                                     │                                    │
                                                     │  R1. REGEN_BRAKING                 │
                                                     │  R2. IDLE_STOP                     │
                                                     │  R3. CRITICAL_SOC_CHARGE           │
                                                     │  R4. THERMAL_PROTECT               │
                                                     │  R5. CITY_EV_MODE                  │
                                                     │  R6. HIGHWAY_ENGINE_ONLY           │
                                                     │  R7. HYBRID_ASSIST                 │
                                                     └─────────────────┬──────────────────┘
                                                                       │
                                                              [ FINAL EMS MODE ]
                                                                       │
                                                                       ▼
                                                       [ POWER DISTRIBUTION MODULE ]
                                                     (Controls physical inverters/motors)

================================================================================================
    """)

if __name__ == "__main__":
    print_architecture()
