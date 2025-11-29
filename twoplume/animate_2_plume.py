import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings
<<<<<<< train2Plume
from agents import SurgeCastAgent, SurgeRandomAgent
=======
from agents import SurgeRandomAgent
>>>>>>> main
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Circle

warnings.simplefilter(action='ignore', category=FutureWarning)

# Configuration - UPDATED FOR DUAL PLUMES
DATA_DIR = "/Users/aaravsinha/plume/plumedata/"
DATASET_NAME = "dual_plume_warmup"
PUFF_FILE = os.path.join(DATA_DIR, f"puff_data_{DATASET_NAME}.pickle")
WIND_FILE = os.path.join(DATA_DIR, f"wind_data_{DATASET_NAME}.pickle")
OUTPUT_MP4 = f"{DATASET_NAME}_with_surgecast_agent_1.mp4"
WARMUP_TIME = 100.0  # Warm-up period in seconds (must match data generation)
FPS = 30  # Frames per second for animation
SOURCE_LOCATIONS = [(0, -1), (0, 1)]  # Dual plume centers
START_POSITION = (4.0, 0.0)  # Central starting position
ARENA_X_LIM = (-2, 10)  # Expanded X limits
ARENA_Y_LIM = (-3, 3)  # Expanded Y limits

# Wind compass visualization parameters
COMPASS_POS = (8.0, 2.5)  # Moved to avoid plume overlap
COMPASS_SIZE = 0.8  # Radius of the compass circle
ARROW_COLOR = 'darkred'
ARROW_ALPHA = 0.8
COMPASS_BG_COLOR = 'lightgray'
COMPASS_EDGE_COLOR = 'black'

# Wind compass visualization parameters
COMPASS_POS = (-1.0, 1.8)  # Position in data coordinates (top-right)
COMPASS_SIZE = 0.8  # Radius of the compass circle
ARROW_COLOR = 'darkred'
ARROW_ALPHA = 0.8
COMPASS_BG_COLOR = 'lightgray'
COMPASS_EDGE_COLOR = 'black'

# Load data with verification
print("Loading data...")
puff_df = pd.read_pickle(PUFF_FILE)
wind_df = pd.read_pickle(WIND_FILE)

# Diagnostic output
print("\n=== Data Verification ===")
print(f"Wind data shape: {wind_df.shape}")
print(f"Wind time range: {wind_df['time'].min():.2f}s to {wind_df['time'].max():.2f}s")
print(f"Puff data shape: {puff_df.shape}")
print(f"Puff time range: {puff_df['time'].min():.2f}s to {puff_df['time'].max():.2f}s")

# Create time grid using wind data (complete timeline)
times = sorted(wind_df['time'].unique())
times_post_warmup = [t for t in times if t >= WARMUP_TIME]
print(f"\nUsing {len(times_post_warmup)} time points after {WARMUP_TIME}s warm-up")

if len(times_post_warmup) == 0:
    raise ValueError(
        f"No valid timesteps after warmup. Data goes to {max(times):.2f}s. "
        f"Regenerate with longer duration."
    )

# Define environment with heading tracking - UPDATED FOR DUAL PLUMES
class PlumeEnvironment:
    def __init__(self, puff_df, wind_df, max_steps=750, threshold=0.3, 
                 action_type='continuous', dt=0.01, source_locations=SOURCE_LOCATIONS,
                 warmup_time=100.0):  # Warmup must match data generation
        self.puff_df = puff_df
        self.wind_df = wind_df
        self.max_steps = max_steps
        self.threshold = threshold
        self.action_type = action_type
        self.dt = dt
        self.source_locations = source_locations
        self.warmup_time = warmup_time
        
        # Use wind_df for complete time reference
        self.times = sorted(wind_df['time'].unique())
        self.times = [t for t in self.times if t >= warmup_time]
        
        if len(self.times) == 0:
            raise ValueError(f"No times after warmup_time={warmup_time}. "
                           f"Max time in data: {wind_df['time'].max():.2f}")
        
        self.current_time_idx = 0
        self.current_time = self.times[self.current_time_idx]
        self.current_position = np.array([0.0, 0.0])
        self.current_heading = 0  # Track agent's heading
        self.done = False
        self.step_count = 0
        self.action_space = type('', (), {})()
        self.action_space.sample = lambda: np.random.uniform(-1, 1, size=2)

    def reset(self):
        self.current_position = np.array(START_POSITION)
        self.current_heading = 0  # Reset heading
        self.current_time_idx = 0
        self.current_time = self.times[self.current_time_idx]
        self.done = False
        self.step_count = 0
        return self._get_observation()

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, {}
        
        # Continuous movement with heading
        step_size = action[0][0]
        turn_strength = action[0][1]
        
        # Update heading
        self.current_heading += turn_strength * np.pi/2 * self.dt
        self.current_heading %= 2*np.pi
        
        # Apply movement
        self.current_position[0] += step_size * np.cos(self.current_heading) * self.dt
        self.current_position[1] += step_size * np.sin(self.current_heading) * self.dt

        self.step_count += 1
        
        # Check distance to plume sources
        distances = [np.linalg.norm(self.current_position - loc) 
                     for loc in self.source_locations]
        min_distance = min(distances)
        
        # Termination conditions
        if min_distance < self.threshold:  # Reached a plume center
            self.done = True
            reward = 10.0  # Success reward
        elif self.step_count >= self.max_steps or self.current_time_idx >= len(self.times) - 1:
            self.done = True
            reward = -1.0  # Failure penalty
        else:
            # Continuous reward based on proximity
            reward = 1 / (1 + min_distance)
            
            # Move to next time step
            self.current_time_idx += 1
            self.current_time = self.times[self.current_time_idx]
            
        return self._get_observation(), reward, self.done, {}

    def _get_observation(self):
        # Handle potential missing wind data
        wind_mask = wind_df['time'] == self.current_time
        if wind_mask.any():
            wind_row = wind_df[wind_mask].iloc[0]
        else:
            # Find closest time if exact match not found
            idx = wind_df['time'].sub(self.current_time).abs().idxmin()
            wind_row = wind_df.loc[idx]
        
        # Calculate odor using Gaussian model - UPDATED FOR DUAL PLUMES
        odor = 0.0
        current_puffs = puff_df[puff_df['time'] == self.current_time]
        if not current_puffs.empty:
            for idx, puff in current_puffs.iterrows():
                r = puff['radius'] * 10  # Scale radius for better sensitivity
                if r > 0:
                    d = np.sqrt((puff['x'] - self.current_position[0])**2 + 
                                (puff['y'] - self.current_position[1])**2)
                    # Gaussian odor concentration model
                    odor += np.exp(-d**2 / (2 * r**2))
        
        # Debug: print when odor is detected
        if odor > 0.05:
            print(f"Odor detected: {odor:.4f} at position {self.current_position}")

        return {
            'wind_x': wind_row['wind_x'],
            'wind_y': wind_row['wind_y'],
            'odor': odor,
            'heading': self.current_heading  # Include heading in observation
        }

# Initialize environment and agent
print("\nInitializing simulation...")
env = PlumeEnvironment(puff_df, wind_df, warmup_time=WARMUP_TIME, source_locations=SOURCE_LOCATIONS)
<<<<<<< train2Plume
agent = SurgeRandomAgent(env.action_space)  # Using the surge-cast agent
=======
agent = SurgeRandomAgent(env.action_space)  # Using discrete agent
>>>>>>> main
obs = env.reset()
agent_positions = [env.current_position.copy()]
agent_headings = [env.current_heading]
done = False

# Run agent simulation
print("Running agent...")
step_count = 0

while not done and step_count < 5000:
    # Build observation array: [wind_x, wind_y, odor, heading]
    obs_array = np.array([
        obs['wind_x'],
        obs['wind_y'],
        obs['odor'],
        obs['heading']
    ]).reshape(1, -1)
    
    # ---- DEBUG: print the inputs to the agent ----
    print(f"Step {step_count:4d} — OBS:", 
          f"wind_x={obs_array[0,0]:.3f}, wind_y={obs_array[0,1]:.3f},",
          f"odor={obs_array[0,2]:.4f}, heading={obs_array[0,3]:.3f}")
    
    # print agent mode
    print(f"Step {step_count:4d}: MODE = {agent.mode}")

    # Agent chooses an action
    action = agent.act(obs_array, 0, done)
    
    # ---- DEBUG: print the outputs from the agent ----
    print(f"           → ACTION:", 
          f"step_size={action[0,0]:.3f}, turn_strength={action[0,1]:.3f}\n")
    
    # Step the environment
    obs, _, done, _ = env.step(action)
    agent_positions.append(env.current_position.copy())
    agent_headings.append(env.current_heading)
    step_count += 1

agent_positions = np.array(agent_positions)

<<<<<<< train2Plume
# Create animation - UPDATED FOR DUAL PLUMES
=======
# Create animation
>>>>>>> main
print(f"\nCreating animation ({FPS} FPS)...")
fig, ax = plt.subplots(figsize=(12, 8))
metadata = dict(title='Plume & Agent Animation', artist='Matplotlib', comment=f'Warmup: {WARMUP_TIME}s')
writer = FFMpegWriter(fps=FPS, metadata=metadata)

with writer.saving(fig, OUTPUT_MP4, dpi=100):
    for i, t in enumerate(times_post_warmup[:len(agent_positions)]):
        ax.clear()
        ax.set_title(f"Time: {t:.2f}s (After {WARMUP_TIME}s warm-up)")
        ax.set_xlabel("x position")
        ax.set_ylabel("y position")
        ax.grid(True)

        # Plot puffs with increased size for visibility
        current_puffs = puff_df[puff_df["time"] == t]
        if len(current_puffs) > 0:
            # Scale puff sizes for better visibility
            ax.scatter(current_puffs["x"], current_puffs["y"], 
                      s=current_puffs["radius"]*5000, alpha=0.2, color='blue', label='Plume')

        # Plot agent path
        ax.plot(agent_positions[:i+1, 0], agent_positions[:i+1, 1], 'r-', linewidth=2, label='Agent Path')
        
        # Plot current position with heading indicator
        current_pos = agent_positions[i]
        heading = agent_headings[i]
        ax.plot(current_pos[0], current_pos[1], 'ro', markersize=8, label='Current Position')
        
        # Add heading arrow
        arrow_len = 0.5
        dx = arrow_len * np.cos(heading)
        dy = arrow_len * np.sin(heading)
        ax.arrow(current_pos[0], current_pos[1], dx, dy, 
                 head_width=0.1, head_length=0.2, fc='darkred', ec='darkred')
        
        # Plot start position
        ax.plot(agent_positions[0, 0], agent_positions[0, 1], 'go', markersize=8, label='Start')
        
        # Plot plume centers - UPDATED FOR DUAL PLUMES
        for loc in SOURCE_LOCATIONS:
            ax.plot(loc[0], loc[1], 'k*', markersize=15, label='Plume Center')

<<<<<<< train2Plume
        # Set wider arena bounds
        ax.set_xlim(ARENA_X_LIM[0], ARENA_X_LIM[1])
        ax.set_ylim(ARENA_Y_LIM[0], ARENA_Y_LIM[1])
        
        # Visualizing wind direction as a compass
        try:
            # Get wind data for current frame
            wind_mask = wind_df['time'] == t
            if wind_mask.any():
                wind_row = wind_df[wind_mask].iloc[0]
            else:
                idx = wind_df['time'].sub(t).abs().idxmin()
                wind_row = wind_df.loc[idx]
                
=======
        # Set consistent bounds
        ax.set_xlim(-2, 8)
        ax.set_ylim(-3, 3)
        
        #visualizing wind direction as a compass
        try:
            # Get wind data for current frame
            wind_row = wind_df[wind_df['time'] == t].iloc[0]
>>>>>>> main
            wind_x = wind_row['wind_x']
            wind_y = wind_row['wind_y']
            
            # Calculate magnitude and direction
            mag = np.sqrt(wind_x**2 + wind_y**2)
            if mag > 0:
                wind_x_norm = wind_x / mag
                wind_y_norm = wind_y / mag
            else:
                wind_x_norm, wind_y_norm = 0, 0
                
            # Draw compass circle
            compass_circle = Circle(
                COMPASS_POS, 
                COMPASS_SIZE,
                facecolor=COMPASS_BG_COLOR,
                edgecolor=COMPASS_EDGE_COLOR,
                linewidth=2,
                zorder=10
            )
            ax.add_patch(compass_circle)
            
            # Draw compass markers (N, E, S, W)
            ax.text(COMPASS_POS[0], COMPASS_POS[1] + COMPASS_SIZE + 0.1, 'N', 
                   ha='center', va='center', fontsize=8, zorder=11)
            ax.text(COMPASS_POS[0] + COMPASS_SIZE + 0.1, COMPASS_POS[1], 'E', 
                   ha='center', va='center', fontsize=8, zorder=11)
            ax.text(COMPASS_POS[0], COMPASS_POS[1] - COMPASS_SIZE - 0.1, 'S', 
                   ha='center', va='center', fontsize=8, zorder=11)
            ax.text(COMPASS_POS[0] - COMPASS_SIZE - 0.1, COMPASS_POS[1], 'W', 
                   ha='center', va='center', fontsize=8, zorder=11)
            
            # Draw wind direction arrow (scaled to compass size)
            arrow_length = COMPASS_SIZE * 0.9  # Leave some margin
            ax.arrow(
                COMPASS_POS[0], COMPASS_POS[1],
                wind_x_norm * arrow_length, wind_y_norm * arrow_length,
                head_width=0.1, head_length=0.15,
                fc=ARROW_COLOR, ec=ARROW_COLOR,
                alpha=ARROW_ALPHA,
                zorder=12
            )
            
            # Add wind speed text below compass
            ax.text(
                COMPASS_POS[0], 
                COMPASS_POS[1] - COMPASS_SIZE - 0.3,
                f"Wind: {mag:.2f} m/s", 
                fontsize=9,
                ha='center',
                bbox=dict(facecolor='white', alpha=0.7)
            )
<<<<<<< train2Plume
        except Exception as e:
            print(f"Error drawing wind compass: {e}")
=======
        except IndexError:
            pass  # Skip if wind data is missing(stupid imports!!)
>>>>>>> main
        
        # Create legend handles
        handles = [
            plt.Line2D([0], [0], color='red', marker='o', linestyle='-', label='Agent Path'),
            plt.Line2D([0], [0], color='green', marker='o', linestyle='None', label='Start'),
            plt.Line2D([0], [0], color='black', marker='*', linestyle='None', label='Plume Center'),
            plt.Line2D([0], [0], color='blue', marker='o', alpha=0.3, linestyle='None', label='Plume')
        ]
        ax.legend(handles=handles, loc='upper right')

        writer.grab_frame()
        if i % 100 == 0:
            print(f"Frame {i}/{len(times_post_warmup)} ({i/len(times_post_warmup):.1%})")

print(f"\n✅ Successfully saved animation to: {OUTPUT_MP4}")
print(f"Agent ended at position: {agent_positions[-1]}")
print(f"Closest plume center: {min(SOURCE_LOCATIONS, key=lambda loc: np.linalg.norm(agent_positions[-1] - np.array(loc)))}")
<<<<<<< train2Plume
print(f"Total steps: {step_count}")
=======
>>>>>>> main
