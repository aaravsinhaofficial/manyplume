#!/usr/bin/env python

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import argparse
import pandas as pd
import numpy as np
import config
import sim_utils
import sim_analysis

# Parse CLI arguments
parser = argparse.ArgumentParser(description='Generate walking-scale plume simulations')
parser.add_argument('--duration', metavar='d', type=int, default=150,
                    help='simulation duration in seconds')
parser.add_argument('--cores', metavar='c', type=int, default=8,
                    help='number of cores to use')
parser.add_argument('--dataset_name', type=str, default='constantx5b5_walk')
parser.add_argument('--fname_suffix', type=str, default='_walk')
parser.add_argument('--dt', type=float, default=0.005,
                    help='time per step (seconds) - finer for walking scale')
parser.add_argument('--wind_magnitude', type=float, default=0.002,
                    help='m/s - much slower for walking scale')
parser.add_argument('--wind_y_varx', type=float, default=4.0,
                    help='wind variability factor')
parser.add_argument('--birth_rate', type=float, default=1.0,
                    help='Poisson birth_rate parameter')
parser.add_argument('--outdir', type=str, default=config.datadir)

args = parser.parse_args()
print(args)

# Extract regime from dataset name for wind generation
regime = args.dataset_name.replace('_walk', '')

# Wind field generation optimized for walking scale
wind_df = sim_utils.get_wind_xyt(
    args.duration,
    dt=args.dt,
    wind_magnitude=args.wind_magnitude,
    regime=regime
)
wind_df['tidx'] = np.arange(len(wind_df), dtype=int)
wind_fname = f'{args.outdir}/wind_data_{args.dataset_name}{args.fname_suffix}.pickle'
wind_df.to_pickle(wind_fname)
print("Wind data shape:", wind_df.shape)
print("Saved", wind_fname)

# Puff simulation with walking-appropriate parameters
wind_y_var = args.wind_magnitude / np.sqrt(args.wind_y_varx)

puff_df = sim_utils.get_puffs_df_vector(
    wind_df,
    wind_y_var=wind_y_var,
    birth_rate=args.birth_rate,
    verbose=True
)

puff_fname = f'{args.outdir}/puff_data_{args.dataset_name}{args.fname_suffix}.pickle'
puff_df.to_pickle(puff_fname)
print('puff_df.shape', puff_df.shape)
print("Saved", puff_fname)

# Generate visualization
try:
    data_puffs, data_wind = sim_analysis.load_plume(f'{args.dataset_name}{args.fname_suffix}')
    t_val = data_puffs['time'].iloc[-1]
    fig, ax = sim_analysis.plot_puffs_and_wind_vectors(
        data_puffs,
        data_wind,
        t_val,
        fname='',
        plotsize=(8, 8)
    )
    fig.savefig(f'{args.outdir}/{args.dataset_name}{args.fname_suffix}_t{t_val:3.3f}.png')
    
    # Walking-scale zoom
    ax.set_xlim(-0.1, 1.2)  # Much smaller scale for walking
    ax.set_ylim(-0.18, +0.18)
    fig.savefig(f'{args.outdir}/{args.dataset_name}{args.fname_suffix}_t{t_val:3.3f}z.png')
    print(f"Visualization saved for {args.dataset_name}")
except Exception as e:
    print(f"Visualization failed: {e}")