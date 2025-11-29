import pickle
import pandas as pd
import sys
sys.path.append('/Users/aaravsinha/manyplume/oneplume')
import log_analysis

# Load episodes from evalCli.py output files
model_dir = "/Users/aaravsinha/manyplume/oneplume/ppo/trained_models/Walking20250807/plume_Walk-v17_20250807_VRNN_constantx5b5noisy3x5b5_stepoob_bx2.72_t100000010000_q0.40.5_dmx0.80.8_dmn0.70.4_h64_wd0.00012_n1_walking_seed30970f1"  # Directory containing .pkl files from evalCli
dataset = 'constantx5b5'  # or 'switch45x5b5', 'noisy3x5b5', etc.

# Load the episode logs
log_fname = f"{model_dir}/{dataset}.pkl"
with open(log_fname, 'rb') as f:
    episode_logs = pickle.load(f)

# Option 1: Select episodes by index
selected_episodes = [0, 5, 10, 15]  # Choose episode indices you want
selected_logs = [episode_logs[i] for i in selected_episodes]

# Option 2: Or use the log_analysis utility to get structured selection
selected_df = log_analysis.get_selected_df(
    model_dir, 
    use_datasets=[dataset], 
    n_episodes_home=30,  # Number of HOME episodes
    n_episodes_other=30,  # Number of OOB episodes
    min_ep_steps=0
)

import numpy as np
import matplotlib.pyplot as plt

# For each selected episode, generate odor timeseries with proper color coding
for idx, log in enumerate(selected_logs):
    # Extract trajectory dataframe with odor information
    traj_df = log_analysis.get_traj_df(
        log, 
        extended_metadata=True,  # This includes odor metrics
        squash_action=True,
        seed='3307e9'  # Optional: specify model seed
    )
    
    # Create time array in seconds (convert to numpy array for indexing)
    time_array = np.array(traj_df.index) * 0.04
    
    # Plot odor concentration time series with green/blue color coding
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    
    # Plot 1: Raw odor concentration with color-coded background
    # Color background based on odor presence (green = in plume, blue = out of plume)
    for i in range(len(traj_df) - 1):
        if traj_df['odor_01'].iloc[i] > 0:  # In plume
            axes[0].axvspan(time_array[i], time_array[i+1], 
                           facecolor='green', alpha=0.2)
        else:  # Out of plume
            axes[0].axvspan(time_array[i], time_array[i+1], 
                           facecolor='blue', alpha=0.1)
    
    axes[0].plot(time_array, traj_df['odor_clip'], 'k-', linewidth=1.5, alpha=0.8)
    axes[0].set_ylabel('Odor Concentration')
    axes[0].set_title(f'Episode {selected_episodes[idx]}: Odor Encounter Time Series (Green=In Plume, Blue=Out of Plume)')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Binary odor encounters with color coding
    # Create a filled area plot for better visualization
    axes[1].fill_between(time_array, 0, traj_df['odor_01'], 
                         where=(traj_df['odor_01'] > 0), 
                         color='green', alpha=0.6, label='In Plume')
    axes[1].fill_between(time_array, 0, 1, 
                         where=(traj_df['odor_01'] == 0), 
                         color='blue', alpha=0.2, label='Out of Plume')
    axes[1].plot(time_array, traj_df['odor_01'], 'k-', linewidth=1, alpha=0.7)
    axes[1].set_ylabel('Odor Detection')
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Time since last encounter with threshold coloring
    axes[2].plot(time_array, traj_df['odor_lastenc'], 'k-', linewidth=1.5)
    # Add horizontal line for typical reacquisition threshold (e.g., 2 seconds)
    axes[2].axhline(y=2.0, color='orange', linestyle='--', alpha=0.5, label='2s threshold')
    axes[2].set_ylabel('Time Since Last Encounter (s)')
    axes[2].set_xlabel('Time (s)')
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'odor_timeseries_ep{selected_episodes[idx]}_colored.png', dpi=300, bbox_inches='tight')
    plt.show()

# Create a combined plot showing odor encounters across episodes
fig, axes = plt.subplots(len(selected_logs), 1, figsize=(14, 3*len(selected_logs)), sharex=True)
if len(selected_logs) == 1:
    axes = [axes]

for idx, log in enumerate(selected_logs):
    traj_df = log_analysis.get_traj_df(log, extended_metadata=True, squash_action=True, seed='3307e9')
    time_array = np.array(traj_df.index) * 0.04
    
    # Create color-coded background
    for i in range(len(traj_df) - 1):
        if traj_df['odor_01'].iloc[i] > 0:  # In plume
            axes[idx].axvspan(time_array[i], time_array[i+1], 
                             facecolor='green', alpha=0.25)
        else:  # Out of plume
            axes[idx].axvspan(time_array[i], time_array[i+1], 
                             facecolor='blue', alpha=0.1)
    
    # Plot odor concentration
    axes[idx].plot(time_array, traj_df['odor_clip'], 'k-', linewidth=1.2)
    axes[idx].set_ylabel(f'Ep {selected_episodes[idx]}\nOdor Conc.')
    axes[idx].grid(True, alpha=0.3)
    
    # Add outcome annotation
    outcome = log['infos'][-1][0]['done']
    axes[idx].text(0.98, 0.95, f'Outcome: {outcome}', 
                   transform=axes[idx].transAxes, 
                   ha='right', va='top', 
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

axes[-1].set_xlabel('Time (s)')
fig.suptitle('Odor Encounters Across Episodes (Green=In Plume, Blue=Out of Plume)', fontsize=14)
plt.tight_layout()
plt.savefig('odor_encounters_all_episodes.png', dpi=300, bbox_inches='tight')
plt.show()

# Calculate odor statistics for selected episodes
odor_stats = []
for idx, log in enumerate(selected_logs):
    traj_df = log_analysis.get_traj_df(log, extended_metadata=True, squash_action=True, seed='3307e9')
    
    # Calculate time in plume vs out of plume
    total_time = len(traj_df) * 0.04  # in seconds
    time_in_plume = (traj_df['odor_01'] > 0).sum() * 0.04
    time_out_plume = total_time - time_in_plume
    
    stats = {
        'episode': selected_episodes[idx],
        'total_odor_encounters': (traj_df['odor_01'] > 0).sum(),
        'time_in_plume_s': time_in_plume,
        'time_out_plume_s': time_out_plume,
        'percent_in_plume': (time_in_plume / total_time) * 100,
        'max_odor_conc': traj_df['odor_clip'].max(),
        'mean_odor_conc': traj_df['odor_clip'].mean(),
        'max_time_without_odor': traj_df['odor_lastenc'].max(),
        'outcome': log['infos'][-1][0]['done']  # HOME, OOB, etc.
    }
    odor_stats.append(stats)

odor_stats_df = pd.DataFrame(odor_stats)
print("\nOdor Encounter Statistics:")
print(odor_stats_df.to_string())

# Create a summary pie chart for time in/out of plume
fig, axes = plt.subplots(1, len(selected_logs), figsize=(4*len(selected_logs), 4))
if len(selected_logs) == 1:
    axes = [axes]

for idx, stats in enumerate(odor_stats):
    sizes = [stats['time_in_plume_s'], stats['time_out_plume_s']]
    colors = ['green', 'blue']
    labels = [f"In Plume\n{stats['percent_in_plume']:.1f}%", 
              f"Out of Plume\n{100-stats['percent_in_plume']:.1f}%"]
    
    axes[idx].pie(sizes, labels=labels, colors=colors, autopct='%1.1f s', 
                  startangle=90, alpha=0.7)
    axes[idx].set_title(f"Episode {stats['episode']}\n{stats['outcome']}")

plt.suptitle('Time Distribution: In Plume vs Out of Plume', fontsize=14)
plt.tight_layout()
plt.savefig('plume_time_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# Color by behavioral regime (keeping original code)
for idx, log in enumerate(selected_logs):
    traj_df = log_analysis.get_traj_df(log, extended_metadata=True, squash_action=True, seed='3307e9')
    time_array = np.array(traj_df.index) * 0.04
    
    # Get regime colors
    regime_colorby = log_analysis.regime_to_colors(traj_df['regime'].to_list())
    
    # Plot odor with regime coloring
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # Plot odor concentration with regime colors as background
    for i in range(len(traj_df) - 1):
        ax.axvspan(time_array[i], time_array[i+1], 
                   facecolor=regime_colorby[i], alpha=0.3)
    
    ax.plot(time_array, traj_df['odor_clip'], 'k-', linewidth=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Odor Concentration')
    ax.set_title(f'Episode {selected_episodes[idx]}: Odor Encounters by Behavioral Regime')
    
    # Add legend for regimes
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='cyan', alpha=0.3, label='WARMUP'),
        Patch(facecolor='green', alpha=0.3, label='TRACK'),
        Patch(facecolor='blue', alpha=0.3, label='RECOVER'),
        Patch(facecolor='magenta', alpha=0.3, label='LOST')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(f'odor_regime_ep{selected_episodes[idx]}.png', dpi=300, bbox_inches='tight')
    plt.show()