#!/bin/bash
# Train flying agents for odor plume tracking
# This script invokes the main training script with flying-specific parameters

# Navigate to the ppo directory and run the flying training script
cd "$(dirname "$0")/../../oneplume/ppo"
./train_agents.sh

# Key flying parameters (set in the main script):
# - walking=False (default, not explicitly set)
# - move_capacity=2.0 m/s
# - turn_capacity=6.25*pi rad/s
# - homed_radius=0.2 m
# - stray_max=2.0 m
# - Wind advection enabled
