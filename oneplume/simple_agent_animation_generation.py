#!/usr/bin/env python3
"""
Simplified agent animation visualization for plume environment
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch
from plume_env import PlumeEnvironment
import pandas as pd
from tqdm import tqdm

# Set global configuration parameters
config = {
    'env': {
        'odor_threshold': 0.1,
        'plume_type': 'constant',
        'arena_size': (10, 5),
        'source_position': (0, 0),
        'wind_angle': 0,
        'wind_speed': 1.0,
        'diffusion_rate': 0.1,
        'puff_release_rate': 5,
        'puff_density': 0.5
    }
}

def run_agent_episode(env, actor_critic, device):
    """Run a single episode and collect trajectory data"""
    obs = env.reset()
    obs = torch.FloatTensor(obs).unsqueeze(0).to(device)
    
    positions = []
    concentrations = []
    rewards = []
    infos = []
    done = False
    
    # Initialize hidden state if recurrent
    if hasattr(actor_critic.base, 'rnn'):
        hx = torch.zeros(1, actor_critic.recurrent_hidden_state_size).to(device)
    else:
        hx = None
        
    masks = torch.ones(1, 1).to(device)

    while not done:
        with torch.no_grad():
            outputs = actor_critic.act(obs, hx, masks, deterministic=True)
            action = outputs[1]
            
            # Update hidden state if recurrent
            if hx is not None:
                hx = outputs[3]
        
        # Step environment
        obs, reward, done, info = env.step(action.cpu().numpy()[0])
        obs = torch.FloatTensor(obs).unsqueeze(0).to(device)
        masks = torch.tensor([[0.0]] if done else [[1.0]]).float().to(device)
        
        # Store trajectory data
        positions.append(info['position'])
        concentrations.append(info['concentration'])
        rewards.append(reward)
        infos.append(info)
    
    return np.array(positions), np.array(concentrations), np.array(rewards), infos

def create_plume_field(env, resolution=100):
    """Create a plume concentration field for visualization"""
    plume_field = np.zeros((resolution, resolution))
    x_min, x_max = -1, env.arena_size[0] + 1
    y_min, y_max = -env.arena_size[1] - 1, env.arena_size[1] + 1
    
    for i in range(resolution):
        for j in range(resolution):
            x = x_min + (x_max - x_min) * i / resolution
            y = y_min + (y_max - y_min) * j / resolution
            plume_field[i, j] = env.get_concentration_at_point((x, y))
    
    return plume_field

def create_animation(positions, concentrations, rewards, infos, out_path, env):
    """Create animation of agent trajectory"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot plume concentration field
    plume_img = ax1.imshow(create_plume_field(env), 
                          cmap='viridis', 
                          vmin=0, vmax=1, 
                          origin='lower',
                          extent=[-1, env.arena_size[0] + 1, 
                                  -env.arena_size[1] - 1, env.arena_size[1] + 1])
    ax1.set_title(f"Agent Trajectory")
    ax1.set_xlabel("X Position (m)")
    ax1.set_ylabel("Y Position (m)")
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Plot trajectory
    trajectory, = ax1.plot([], [], 'r-', linewidth=1)
    agent_pos, = ax1.plot([], [], 'ro', markersize=6)
    source_pos, = ax1.plot([env.source_position[0]], [env.source_position[1]], 
                          'g*', markersize=12, label='Source')
    
    # Plot concentration history
    conc_line, = ax2.plot([], [], 'b-', linewidth=2)
    ax2.set_title("Odor Concentration & Rewards")
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Concentration", color='b')
    ax2.set_ylim(0, 1.1)
    ax2.set_xlim(0, len(positions))
    ax2.grid(True, linestyle='--', alpha=0.3)
    
    # Plot reward history
    reward_ax = ax2.twinx()
    reward_line, = reward_ax.plot([], [], 'g-', linewidth=1, alpha=0.7)
    reward_ax.set_ylabel("Reward", color='g')
    reward_ax.tick_params(axis='y', labelcolor='g')
    reward_ax.set_ylim(np.min(rewards)-0.1, np.max(rewards)+0.1)
    
    # Add legend
    ax1.legend([source_pos, agent_pos], ['Odor Source', 'Agent Position'], 
              loc='upper right')
    
    # Animation update function
    def update(frame):
        # Update trajectory
        x = positions[:frame, 0]
        y = positions[:frame, 1]
        trajectory.set_data(x, y)
        agent_pos.set_data(positions[frame, 0], positions[frame, 1])
        
        # Update concentration plot
        conc_line.set_data(np.arange(frame), concentrations[:frame])
        
        # Update reward plot
        reward_line.set_data(np.arange(frame), rewards[:frame])
        
        # Update title with current position
        ax1.set_title(f"Agent Trajectory (Step: {frame+1}/{len(positions)})")
        
        return trajectory, agent_pos, conc_line, reward_line
    
    # Create animation
    anim = FuncAnimation(fig, update, frames=len(positions),
                        interval=100, blit=True)
    
    # Save animation
    anim.save(out_path, writer='ffmpeg', fps=15, dpi=100)
    plt.close(fig)
    print(f"Animation saved to {out_path}")

def main():
    parser = argparse.ArgumentParser(description='Animate agent in plume environment')
    parser.add_argument('--model_path', type=str, required=True, 
                       help='Path to trained model')
    parser.add_argument('--env_name', type=str, default='constantx5b5',
                       help='Plume environment name')
    parser.add_argument('--n_episodes', type=int, default=1,
                       help='Number of episodes to animate')
    parser.add_argument('--out_dir', type=str, default='animations',
                       help='Output directory for animations')
    parser.add_argument('--diffusionx', type=float, default=1.0,
                       help='Diffusion parameter for environment')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    actor_critic, ob_rms = torch.load(args.model_path, map_location=device)
    actor_critic.eval()
    
    # Create environment
    env = PlumeEnvironment(
        plume_type=args.env_name,
        arena_size=config['env']['arena_size'],
        source_position=config['env']['source_position'],
        wind_angle=config['env']['wind_angle'],
        wind_speed=config['env']['wind_speed'],
        diffusion_rate=args.diffusionx,
        puff_release_rate=config['env']['puff_release_rate'],
        puff_density=config['env']['puff_density'],
        odor_threshold=config['env']['odor_threshold']
    )
    
    # Run episodes and create animations
    for i in range(args.n_episodes):
        print(f"Running episode {i+1}/{args.n_episodes}")
        positions, concentrations, rewards, infos = run_agent_episode(env, actor_critic, device)
        
        # Create animation
        out_path = os.path.join(args.out_dir, f'episode_{i+1}.mp4')
        create_animation(positions, concentrations, rewards, infos, out_path, env)
        print(f"Saved animation to {out_path}")

if __name__ == "__main__":
    main()