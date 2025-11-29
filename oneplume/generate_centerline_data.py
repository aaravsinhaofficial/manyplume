import os
import subprocess
import sys

# Add the current directory to Python path
sys.path.append('.')

# Import the centerline generation functions
from centerline_cli import get_puffs_df_vector_centerline
import config
import pandas as pd
import numpy as np

def generate_centerline_for_dataset(dataset):
    """Generate centerline data for a given dataset"""
    print(f"Generating centerline data for {dataset}...")
    
    data_dir = config.datadir
    wind_filename = f'{data_dir}/wind_data_{dataset}.pickle'
    
    # Check if wind data exists
    if not os.path.exists(wind_filename):
        print(f"Warning: Wind data not found at {wind_filename}")
        return False
    
    try:
        # Load wind data
        wind_df = pd.read_pickle(wind_filename)
        
        # Generate centerline data
        centerline_df = get_puffs_df_vector_centerline(wind_df, verbose=True)
        
        # Sort and process
        centerline_df = centerline_df.sort_values(by=['tidx', 'puff_number']).reset_index(drop=True)
        
        # Calculate slope and angle
        y_diff = centerline_df['y'].rolling(8).mean().diff()
        x_diff = centerline_df['x'].rolling(8).mean().diff()
        centerline_df['slope'] = y_diff/x_diff
        centerline_df['angle'] = np.arctan(centerline_df['slope'])/np.pi
        centerline_df['angle'] = (centerline_df['angle'] + 1)/2  # shift scale
        
        # Save centerline data
        centerline_filename = f'{data_dir}/centerline_data_{dataset}.pickle'
        centerline_df.to_pickle(centerline_filename)
        print(f"Saved centerline data to {centerline_filename}")
        return True
        
    except Exception as e:
        print(f"Error generating centerline for {dataset}: {e}")
        return False

def main():
    """Generate centerline data for common datasets"""
    datasets = [
        'constantx5b5',
        'noisy3x5b5', 
        'noisy6x5b5',
        'switch45x5b5',
        'switch15x5b5',
        'switch30x5b5',
        'noisy1x5b5',
        'noisy2x5b5',
        'noisy4x5b5',
        'noisy5x5b5'
    ]
    
    success_count = 0
    for dataset in datasets:
        if generate_centerline_for_dataset(dataset):
            success_count += 1
    
    print(f"\nGenerated centerline data for {success_count}/{len(datasets)} datasets")

if __name__ == '__main__':
    main()
