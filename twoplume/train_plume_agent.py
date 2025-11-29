#focus on after finishing up oneplume training animation!

import os
import pandas as pd
from plume_env import PlumeEnvironment
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise

# Configuration
DATA_DIR = "/path/to/plumedata/"
DATASET_NAME = "dual_plume_warmup"
PUFF_FILE = os.path.join(DATA_DIR, f"puff_data_{DATASET_NAME}.pickle")
WIND_FILE = os.path.join(DATA_DIR, f"wind_data_{DATASET_NAME}.pickle")
MODEL_PATH = f"plume_ddpg_{DATASET_NAME}"

# Load data
puff_df = pd.read_pickle(PUFF_FILE)
wind_df = pd.read_pickle(WIND_FILE)

# Create environment
env = PlumeEnvironment(puff_df, wind_df)

# Add action noise for exploration
n_actions = env.action_space.shape[0]
action_noise = NormalActionNoise(
    mean=np.zeros(n_actions),
    sigma=0.1 * np.ones(n_actions)
)

# Initialize DDPG agent
model = DDPG(
    "MlpPolicy",
    env,
    action_noise=action_noise,
    verbose=1,
    device="cpu"  # Use "cuda" for GPU acceleration
)

# Train the agent
print("Starting training...")
model.learn(total_timesteps=100000)

# Save the trained model
model.save(MODEL_PATH)
print(f"Saved trained model to {MODEL_PATH}")