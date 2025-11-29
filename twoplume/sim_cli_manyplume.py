import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import sim_utils_manyplume as sim_utils
import argparse
import pandas as pd
import config
import numpy as np

# Parse CLI arguments
parser = argparse.ArgumentParser(description='Generate plume simulations')
parser.add_argument('--duration', metavar='d', type=int, default=100,
                    help='simulation duration in seconds')
parser.add_argument('--cores', metavar='c', type=int, default=8,
                    help='number of cores to use')
parser.add_argument('--dataset_name', type=str, default='sixplume')
parser.add_argument('--fname_suffix', type=str, default='')
parser.add_argument('--dt', type=float, default=0.01,
                    help='time per step (seconds)')
parser.add_argument('--wind_magnitude', type=float, default=0.1,
                    help='m/s')
parser.add_argument('--wind_y_varx', type=float, default=1.0)
parser.add_argument('--birth_rate', type=float, default=2,
                    help='Poisson birth_rate parameter')
parser.add_argument('--outdir', type=str, default=config.datadir)
parser.add_argument('--source_positions', type=str, default='-1,-1;-1,1;2,5;8,5;2,-5;8,-5',  # 6 plume positions
                    help='Semicolon-separated list of x,y plume source positions')
parser.add_argument('--warmup_time', type=float, default=100.0,
                    help='Warm-up time in seconds before plume is fully developed')

args = parser.parse_args()
print(args)

# Calculate total duration with warm-up
total_duration = args.duration + args.warmup_time + 1
print(f"Generating {total_duration:.1f}s simulation ({args.warmup_time:.1f}s warm-up + {args.duration:.1f}s main)")

# Parse source positions
def parse_source_positions(s):
    return [tuple(map(float, pos.split(','))) for pos in s.strip().split(';')]

source_positions = parse_source_positions(args.source_positions)
multi_plume = len(source_positions) > 1

# Wind field (generate for total duration)
wind_df = sim_utils.get_wind_xyt(
    total_duration,
    dt=args.dt,
    wind_magnitude=args.wind_magnitude,
    regime=args.dataset_name
)
wind_df['tidx'] = np.arange(len(wind_df), dtype=int)
wind_fname = f'{args.outdir}/wind_data_{args.dataset_name}{args.fname_suffix}.pickle'
wind_df.to_pickle(wind_fname)
print(wind_df.head(n=5))
print(wind_df.tail(n=5))
print("Saved", wind_fname)

# Puff simulation
wind_y_var = args.wind_magnitude / np.sqrt(args.wind_y_varx)

if multi_plume:
    print("Generating MASTER plume at (0,0)...")
    master_plume = sim_utils.get_puffs_df_vector(
        wind_df,
        wind_y_var=wind_y_var,
        birth_rate=args.birth_rate,
        source_position=(0,0),
        source_id=0,
        verbose=True
    )
    puff_dfs = []
    max_puff = master_plume['puff_number'].max() + 1
    for idx, src_pos in enumerate(source_positions):
        # Copy master plume and offset positions
        plume_df = master_plume.copy()
        plume_df['x'] = plume_df['x'] + (src_pos[0] - 0)
        plume_df['y'] = plume_df['y'] + (src_pos[1] - 0)
        plume_df['source_id'] = idx
        plume_df['puff_number'] += idx * max_puff
        puff_dfs.append(plume_df)
    puff_df = pd.concat(puff_dfs)
else:
    # Single plume case
    puff_df = sim_utils.get_puffs_df_vector(
        wind_df,
        wind_y_var=wind_y_var,
        birth_rate=args.birth_rate,
        source_position=source_positions[0],
        source_id=0,
        verbose=True
    )

puff_fname = f'{args.outdir}/puff_data_{args.dataset_name}{args.fname_suffix}.pickle'
puff_df.to_pickle(puff_fname)
print('puff_df.shape', puff_df.shape)
print(puff_df.tail())
print(puff_df.head())
print("Saved", puff_fname)

# Plotting
import sim_analysis
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
ax.set_xlim(-1, 12)
ax.set_ylim(-1.8, +1.8)
if 'switch' in args.dataset_name:
    ax.set_xlim(-1, +10)
    ax.set_ylim(-5, +5)
fig.savefig(f'{args.outdir}/{args.dataset_name}{args.fname_suffix}_t{t_val:3.3f}z.png')