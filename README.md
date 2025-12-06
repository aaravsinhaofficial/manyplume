# ManyPlume: Deep Reinforcement Learning for Odor Plume Tracking

**Official repository for Aarav Sinha and Satpreet Singh's project "Using Deep Reinforcement Learning to understand Odor Plume Tracking in Walking and Flying Agents", accepted to NeurIPS AI4Science.**

## Overview

This repository contains the code for training and evaluating deep reinforcement learning agents that perform odor plume tracking. A key focus of this work is the **comparison between different locomotor modes** - specifically, how flying and walking agents develop different strategies for tracking odor plumes to their source.

## 🚀 Locomotor Mode Comparison (Flying vs Walking)

One of the central contributions of this research is understanding how different movement modalities affect odor plume tracking. The key differences are:

| Feature | Flying Agent | Walking Agent |
|---------|-------------|---------------|
| **Movement Speed** | 2.0 m/s | 0.05 m/s (5 cm/s) |
| **Turn Rate** | ~19.6 rad/s | ~3.14 rad/s |
| **Success Radius** | 20 cm | 2.2 cm |
| **Maximum Stray Distance** | 2.0 m | 5 cm |
| **Wind Advection** | Affected by wind | Not affected |
| **Arena Scale** | Meters | Centimeters |

For detailed documentation and training scripts, see the [`locomotor_comparison/`](locomotor_comparison/) directory.

### Quick Start: Training Agents

**Flying Agents:**
```bash
cd locomotor_comparison/flying
./train_flying.sh
```

**Walking Agents:**
```bash
cd locomotor_comparison/walking
./train_walking.sh
```

### Key Implementation Details

The locomotor mode is controlled by the `walking` parameter in `oneplume/plume_env.py`:

```python
from plume_env import PlumeEnvironment

# Flying agent (default)
env = PlumeEnvironment(walking=False)

# Walking agent
env = PlumeEnvironment(walking=True)
```

## Repository Structure

```
manyplume/
├── README.md                    # This file
├── locomotor_comparison/        # 🆕 Locomotor mode comparison (flying vs walking)
│   ├── README.md                # Detailed documentation
│   ├── flying/                  # Flying agent training
│   ├── walking/                 # Walking agent training
│   └── shared/                  # Shared comparison utilities
├── oneplume/                    # Single plume tracking experiments
│   ├── plume_env.py             # Core environment (flying/walking modes)
│   ├── agents.py                # Agent implementations
│   ├── sim_cli.py               # Flying-scale simulation generation
│   ├── sim_cli_walking.py       # Walking-scale simulation generation
│   └── ppo/                     # PPO training code
│       ├── main.py              # Main training script
│       ├── train_agents.sh      # Flying agent training
│       └── train_agents_walking.sh  # Walking agent training
└── twoplume/                    # Two-plume tracking experiments
    └── ...
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/aaravsinhaofficial/manyplume.git
cd manyplume
```

2. Install dependencies:
```bash
cd oneplume
pip install -r requirements.txt
```

## Generating Plume Data

### Flying-Scale Plumes
```bash
cd oneplume
python sim_cli.py --duration 120 --dataset_name constantx5b5 --wind_magnitude 0.5 --birth_rate 1.0
```

### Walking-Scale Plumes
```bash
cd oneplume
python sim_cli_walking.py --duration 150 --dataset_name constantx5b5_walk --wind_magnitude 0.002 --dt 0.005
```

## Citation

If you use this code in your research, please cite:
```
@inproceedings{sinha2024plume,
  title={Using Deep Reinforcement Learning to understand Odor Plume Tracking in Walking and Flying Agents},
  author={Sinha, Aarav and Singh, Satpreet},
  booktitle={NeurIPS AI4Science},
  year={2024}
}
```

## License

See individual directories for license information.
