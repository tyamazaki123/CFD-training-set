#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File Name:main.py
Author:Tomoki Yamazaki
Updated by:May 2025
"""
import argparse
import configparser
import os
from solver import SodShockTubeSolver,InitialCondition,BoundaryCondition,RHS


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sod Shock Tube Simulator")
    parser.add_argument("config", help="Path to configuration .ini file")
    args = parser.parse_args()
    config_path = args.config
    # Read configuration file
    config_parser = configparser.ConfigParser()
    if not os.path.isfile(config_path):
        print(f"Error: Config file '{config_path}' not found.", file=sys.stderr)
        sys.exit(1)
    config_parser.read(config_path)
    # Use section "Simulation" if it exists, otherwise use the first section
    if "Simulation" in config_parser:
        config = config_parser["Simulation"]
    else:
        # If no sections defined, ConfigParser puts them in DEFAULT
        config = config_parser.defaults()
    # Prepare output directory and log file
    os.makedirs("output", exist_ok=True)
    log_file_path = os.path.join("output", "simulation.log")
    with open(log_file_path, "w") as log_file:
        # Initialize solver and auxiliary classes
        solver = SodShockTubeSolver(config)
        init_cond = InitialCondition()
        boundary = BoundaryCondition()
        rhs = RHS()
        # Write initial configuration to log
        log_file.write("Configuration Parameters:\n")
        for key in config:
            log_file.write(f"  {key} = {config.get(key)}\n")
        log_file.write("\n")
        # Run the simulation
        solver.run(init_cond, boundary, rhs, log_file)
        print(f"Simulation finished. See '{log_file_path}' for log and 'output/' for results.")
