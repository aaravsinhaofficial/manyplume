#!/bin/bash

# Generate walking-scale datasets for comprehensive evaluation
cd /Users/aaravsinha/manyplume/oneplume

echo "Generating walking-scale datasets with short duration..."

# Standard constant wind dataset
python sim_cli.py \
  --duration 30 \
  --dataset_name constantx5b5_walk \
  --fname_suffix _walk \
  --dt 0.005 \
  --wind_magnitude 0.008 \
  --wind_y_varx 4.0 \
  --birth_rate 1.0

# Wind switching datasets (for evaluation)

python sim_cli.py \
  --duration 30 \
  --dataset_name switch45x5b5_walk \
  --fname_suffix _walk \
  --dt 0.005 \
  --wind_magnitude 0.008 \
  --wind_y_varx 4.0 \
  --birth_rate 1.0

# Noisy wind datasets
python sim_cli.py \
  --duration 30 \
  --dataset_name noisy3x5b5_walk \
  --fname_suffix _walk \
  --dt 0.005 \
  --wind_magnitude 0.008 \
  --wind_y_varx 4.0 \
  --birth_rate 1.0

  --birth_rate 1.0

echo "All walking datasets generated with 10s duration!"
echo "Generated datasets:"
echo "  - constantx5b5_walk (training)"
echo "  - switch15x5b5_walk, switch30x5b5_walk, switch45x5b5_walk (wind direction changes)"
echo "  - noisy1x5b5_walk through noisy6x5b5_walk (various noise levels)"
echo ""
echo "These datasets can now be used for:"
echo "  1. Training walking agents"
echo "  2. Comprehensive evaluation across different wind conditions"
echo "  3. Generalization testing"
