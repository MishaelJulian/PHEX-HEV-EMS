"""
run_ems.py — Entry Point
Author: Antigravity
Date: 2026-05-19

CLI for generating data, training models, running evaluation, and simulating.
"""

import argparse
import sys
import logging
from src.config import SEPARATOR

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

def main():
    parser = argparse.ArgumentParser(description="PHEX EMS AI Pipeline")
    parser.add_argument("--generate-data", action="store_true", help="Generate synthetic training data")
    parser.add_argument("--train", action="store_true", help="Train ML models on generated data")
    parser.add_argument("--evaluate", action="store_true", help="Run safety override evaluation suite")
    parser.add_argument("--simulate-cycles", action="store_true", help="Execute physics-based simulation on standard drive cycles, write KPIs, and generate all 13 plots")
    parser.add_argument("--mode", choices=["rule", "hybrid"], default=None, help="Run real-time simulation in specified mode")
    parser.add_argument("--rows", type=int, default=50, help="Number of rows for simulation")
    parser.add_argument("--delay", type=int, default=100, help="Delay in ms for simulation UI")

    args = parser.parse_args()

    # If no args provided, print help
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    if args.generate_data:
        logger.info(SEPARATOR)
        logger.info(" PHASE 1: GENERATING SYNTHETIC DATA")
        logger.info(SEPARATOR)
        from src.generate_synthetic_data import main as generate_main
        generate_main()

    if args.train:
        logger.info(SEPARATOR)
        logger.info(" PHASE 2: TRAINING EMS ML MODELS")
        logger.info(SEPARATOR)
        from src.ems_model import main as train_main
        train_main()

    if args.evaluate:
        from src.evaluate_ems import run_evaluation
        run_evaluation()

    if args.simulate_cycles:
        logger.info(SEPARATOR)
        logger.info(" RUNNING PHYSICS SIMULATIONS ON STANDARD DRIVE CYCLES")
        logger.info(SEPARATOR)
        
        import json
        from pathlib import Path
        from src.drive_cycles import get_wltp_urban, get_wltp_mixed, get_bangalore_urban
        from src.simulator import run_physics_simulation
        from src.evaluation import compute_kpis, save_kpi_reports
        from src.visualization import generate_all_figures
        
        # 1. Run all three simulations
        logger.info("Executing WLTP Urban Cycle...")
        df_urban = run_physics_simulation(get_wltp_urban(), initial_soc=0.70)
        kpis_urban = compute_kpis(df_urban, initial_soc=0.70)
        logger.info(f"WLTP Urban Fuel Economy: {kpis_urban['fuel_economy_l_100km']} L/100km")
        
        logger.info("Executing WLTP Mixed Cycle...")
        df_mixed = run_physics_simulation(get_wltp_mixed(), initial_soc=0.70)
        kpis_mixed = compute_kpis(df_mixed, initial_soc=0.70)
        logger.info(f"WLTP Mixed Fuel Economy: {kpis_mixed['fuel_economy_l_100km']} L/100km")
        
        logger.info("Executing Bangalore Urban Cycle...")
        df_bangalore = run_physics_simulation(get_bangalore_urban(), initial_soc=0.70)
        kpis_bangalore = compute_kpis(df_bangalore, initial_soc=0.70)
        logger.info(f"Bangalore Urban Fuel Economy: {kpis_bangalore['fuel_economy_l_100km']} L/100km")
        
        # 2. Write reports
        outputs_dir = Path("outputs")
        outputs_dir.mkdir(parents=True, exist_ok=True)
        
        # Save individual JSON reports
        with open(outputs_dir / "kpi_report_wltp_urban.json", "w") as f:
            json.dump(kpis_urban, f, indent=4)
        with open(outputs_dir / "kpi_report_wltp_mixed.json", "w") as f:
            json.dump(kpis_mixed, f, indent=4)
        with open(outputs_dir / "kpi_report_bangalore_urban.json", "w") as f:
            json.dump(kpis_bangalore, f, indent=4)
            
        def write_txt_summary(kpis, name, filename):
            with open(outputs_dir / filename, "w") as f:
                f.write(f"PHEX HEV EMS Simulation - {name} KPI Summary Report\n")
                f.write("=============================================\n\n")
                f.write(f"Fuel Economy             : {kpis.get('fuel_economy_l_100km', 0.0):.2f} L/100 km\n")
                f.write(f"SOC Deviation (RMS)      : {kpis.get('soc_deviation_pct', 0.0):.2f} %\n")
                f.write(f"Energy Recuperation      : {kpis.get('energy_recuperation_pct', 0.0):.1f} %\n")
                f.write(f"EV Mode Share            : {kpis.get('ev_mode_share_pct', 0.0):.1f} %\n")
                f.write(f"Hybrid Mode Share        : {kpis.get('hybrid_mode_share_pct', 0.0):.1f} %\n")
                f.write(f"ICE Mode Share           : {kpis.get('ice_mode_share_pct', 0.0):.1f} %\n")
                f.write(f"Operating Cost Estimate  : {kpis.get('operating_cost_currency_km', 0.0):.4f} currency/km\n")
                f.write(f"CO2 Estimate             : {kpis.get('co2_g_km', 0.0):.1f} g/km\n")
                f.write(f"Average ICE Efficiency   : {kpis.get('average_ice_efficiency_pct', 0.0):.1f} %\n")
                f.write(f"Peak Battery Power       : {kpis.get('peak_battery_power_kw', 0.0):.2f} kW\n")
                
        write_txt_summary(kpis_urban, "WLTP Urban", "kpi_summary_wltp_urban.txt")
        write_txt_summary(kpis_mixed, "WLTP Mixed", "kpi_summary_wltp_mixed.txt")
        write_txt_summary(kpis_bangalore, "Bangalore Urban", "kpi_summary_bangalore_urban.txt")
        
        # Save the primary reports (defaulting to WLTP Mixed)
        save_kpi_reports(kpis_mixed, output_dir=outputs_dir)
        
        # 3. Generate figures (attach all actual economies to df_mixed.attrs for G.13)
        df_mixed.attrs["economies"] = [
            kpis_urban["fuel_economy_l_100km"],
            kpis_mixed["fuel_economy_l_100km"],
            kpis_bangalore["fuel_economy_l_100km"]
        ]
        
        logger.info("Generating all 13 plots...")
        generate_all_figures(df_mixed, output_dir=outputs_dir / "figures")
        logger.info("Figures successfully saved to outputs/figures/")
        logger.info("Simulation cycles run completed successfully.")

    if args.mode:
        from src.simulator import EMSSimulator
        sim = EMSSimulator(mode=args.mode)
        sim.run(n_rows=args.rows, delay_ms=args.delay)

if __name__ == "__main__":
    main()
