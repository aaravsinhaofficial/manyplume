#!/bin/bash
# Train walking agents for odor plume tracking
# This script invokes the main training script with walking-specific parameters

# Navigate to the ppo directory and run the walking training script
cd "$(dirname "$0")/../../oneplume/ppo"
./train_agents_walking.sh

# Key walking parameters (set in the main script):
# - walking=True
# - move_capacity=0.05 m/s (5 cm/s)
# - turn_capacity=3.14 rad/s
# - homed_radius=0.022 m (2.2 cm)
# - stray_max=0.05 m
# - No wind advection
