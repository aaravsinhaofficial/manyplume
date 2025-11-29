# check_plume_data.py
import pandas as pd
import sys

if len(sys.argv) < 2:
    print("Usage: python check_plume_data.py <dataset_name>")
    exit(1)

dataset = sys.argv[1]
DATA_DIR = "/Users/aaravsinha/plume/plumedata/"

try:
    puff_df = pd.read_pickle(f"{DATA_DIR}/puff_data_{dataset}.pickle")
    wind_df = pd.read_pickle(f"{DATA_DIR}/wind_data_{dataset}.pickle")
    
    print("=== Wind Data ===")
    print(f"Shape: {wind_df.shape}")
    print(f"Time range: {wind_df['time'].min():.2f} to {wind_df['time'].max():.2f}")
    print(f"Time step: {wind_df['time'].diff().mean():.4f}")
    print(f"Columns: {wind_df.columns}")
    
    print("\n=== Puff Data ===")
    print(f"Shape: {puff_df.shape}")
    print(f"Time range: {puff_df['time'].min():.2f} to {puff_df['time'].max():.2f}")
    print(f"Time step: {puff_df['time'].diff().mean():.4f}")
    print(f"Columns: {puff_df.columns}")
    print(f"Unique puff counts: {puff_df['puff_number'].nunique()}")
    
    # Check warm-up coverage
    warmup = 20.0
    puff_coverage = puff_df[puff_df['time'] >= warmup].shape[0] / puff_df.shape[0]
    print(f"\nPuff data after {warmup}s: {puff_coverage:.1%} of total")
    
    # Check if source positions match
    if 'source_x' in puff_df.columns:
        print(f"Sources: {puff_df[['source_x', 'source_y']].drop_duplicates().values.tolist()}")
    
except Exception as e:
    print(f"Error loading data: {e}")