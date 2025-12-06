#!/usr/bin/env python
"""
Locomotor Mode Comparison Utilities

This module provides utilities for comparing flying vs walking agents
in odor plume tracking experiments.
"""

import os
import sys

# Add the oneplume directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../oneplume'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../oneplume/ppo'))

# Locomotor mode configuration
LOCOMOTOR_CONFIGS = {
    'flying': {
        'walking': False,
        'move_capacity': 2.0,  # m/s
        'turn_capacity': 6.25 * 3.14159,  # rad/s
        'homed_radius': 0.2,  # m
        'stray_max': 2.0,  # m
        'wind_advection': True,
        'tick_penalty_factor': 10,
        'description': 'Flying agent with wind advection and higher movement capacity'
    },
    'walking': {
        'walking': True,
        'move_capacity': 0.05,  # m/s (5 cm/s)
        'turn_capacity': 3.14159,  # rad/s
        'homed_radius': 0.022,  # m (2.2 cm)
        'stray_max': 0.05,  # m
        'wind_advection': False,
        'tick_penalty_factor': 1.2,
        'description': 'Walking agent with controlled movement and no wind advection'
    }
}


def get_locomotor_config(mode):
    """
    Get configuration for a specific locomotor mode.
    
    Args:
        mode: 'flying' or 'walking'
        
    Returns:
        dict: Configuration parameters for the specified mode
    """
    if mode not in LOCOMOTOR_CONFIGS:
        raise ValueError(f"Unknown locomotor mode: {mode}. Must be 'flying' or 'walking'")
    return LOCOMOTOR_CONFIGS[mode]


def print_comparison_table():
    """Print a comparison table of flying vs walking parameters."""
    print("\nLocomotor Mode Comparison")
    print("=" * 60)
    print(f"{'Parameter':<25} {'Flying':<15} {'Walking':<15}")
    print("-" * 60)
    
    flying = LOCOMOTOR_CONFIGS['flying']
    walking = LOCOMOTOR_CONFIGS['walking']
    
    params = [
        ('Move Capacity', f"{flying['move_capacity']} m/s", f"{walking['move_capacity']} m/s"),
        ('Turn Capacity', f"{flying['turn_capacity']:.2f} rad/s", f"{walking['turn_capacity']:.2f} rad/s"),
        ('Homed Radius', f"{flying['homed_radius']} m", f"{walking['homed_radius']} m"),
        ('Stray Max', f"{flying['stray_max']} m", f"{walking['stray_max']} m"),
        ('Wind Advection', str(flying['wind_advection']), str(walking['wind_advection'])),
        ('Tick Penalty Factor', str(flying['tick_penalty_factor']), str(walking['tick_penalty_factor'])),
    ]
    
    for param, fly_val, walk_val in params:
        print(f"{param:<25} {fly_val:<15} {walk_val:<15}")
    
    print("=" * 60)


if __name__ == '__main__':
    print_comparison_table()
    print("\nFlying mode description:", LOCOMOTOR_CONFIGS['flying']['description'])
    print("Walking mode description:", LOCOMOTOR_CONFIGS['walking']['description'])
