# Locomotor Mode Comparison: Flying vs Walking Agents

This directory contains the core code and configuration for comparing odor plume tracking between **flying** and **walking** agents using Deep Reinforcement Learning.

## Overview

The locomotor mode comparison investigates how different movement modalities (flying vs walking) affect odor plume tracking strategies and performance. This is a key aspect of the research presented in our NeurIPS AI4Science paper.

### Key Differences Between Locomotor Modes

| Feature | Flying Agent | Walking Agent |
|---------|-------------|---------------|
| **Move Capacity** | 2.0 m/s | 0.05 m/s (5 cm/s) |
| **Turn Capacity** | 6.25π rad/s (~19.6 rad/s) | π rad/s (~3.14 rad/s) |
| **Homed Radius** | 0.2 m | 0.022 m (2.2 cm) |
| **Stray Max** | 2.0 m | 0.05 m |
| **Wind Drift** | Yes (advected by wind) | No |
| **Plume Query Region** | Full plume | x ≤ 0.5 m (near-source) |
| **Tick Penalty** | -10/episode_steps | -1.2/episode_steps |

## Directory Structure

```
locomotor_comparison/
├── README.md               # This file
├── flying/
│   └── train_flying.sh     # Training script for flying agents
├── walking/
│   └── train_walking.sh    # Training script for walking agents
└── shared/
    └── compare_modes.py    # Comparison analysis utilities
```

## Training Agents

### Flying Agents

```bash
cd ../oneplume/ppo
./train_agents.sh
```

Key parameters for flying:
- `walking=False` (default)
- Higher movement capacity (2.0 m/s)
- Larger arena and homing radius
- Wind advection enabled

### Walking Agents

```bash
cd ../oneplume/ppo
./train_agents_walking.sh
```

Key parameters for walking:
- `walking=True`
- Lower movement capacity (0.05 m/s)
- Smaller arena and homing radius
- No wind advection (agent controls its movement fully)
- Different reward shaping for metabolic costs

## Core Implementation

The locomotor mode is controlled by the `walking` parameter in `oneplume/plume_env.py`:

```python
# Flying mode (default)
env = PlumeEnvironment(walking=False)

# Walking mode
env = PlumeEnvironment(walking=True)
```

### Relevant Files

- **Environment**: `oneplume/plume_env.py` - Contains the `PlumeEnvironment` class with locomotor-specific parameters
- **Agents**: `oneplume/agents.py` - Contains various agent implementations
- **Training**: `oneplume/ppo/main.py` - Main training script with PPO algorithm
- **Simulation**: `oneplume/sim_cli.py` (flying) and `oneplume/sim_cli_walking.py` (walking) - Generate plume data

## Generating Plume Data

### For Flying-Scale Experiments

```bash
cd oneplume
python sim_cli.py --duration 120 --dataset_name constantx5b5 --wind_magnitude 0.5 --birth_rate 1.0
```

### For Walking-Scale Experiments

```bash
cd oneplume
python sim_cli_walking.py --duration 150 --dataset_name constantx5b5_walk --wind_magnitude 0.002 --dt 0.005
```

## Evaluation

Both flying and walking agents can be evaluated using:

```bash
cd oneplume/ppo
./eval_walking_agents.sh  # For walking agents
# or use evalCli.py for custom evaluation
```

## References

For more details on the methodology and results, see our paper:
- "Using Deep Reinforcement Learning to understand Odor Plume Tracking in Walking and Flying Agents" (NeurIPS AI4Science)
