#!/bin/bash

# Evaluate walking agents on all available datasets
ZOODIR="./trained_models/ExptMemory20250724/"

echo "Evaluating walking agents..."

# Find all walking model files
WALKING_MODELS=$(find $ZOODIR -name "*.pt" 2>/dev/null)

if [ -z "$WALKING_MODELS" ]; then
    echo "No walking models found in $ZOODIR. Train walking agents first with train_agents_walking.sh"
    exit 1
fi

echo "Found walking models:"
echo "$WALKING_MODELS"

# All walking datasets for evaluation
DATASETS="constantx5b5_walk switch15x5b5_walk switch30x5b5_walk switch45x5b5_walk noisy1x5b5_walk noisy2x5b5_walk noisy3x5b5_walk noisy4x5b5_walk noisy5x5b5_walk noisy6x5b5_walk"

MAXJOBS=4

for DATASET in $DATASETS; do
    for MODEL in $WALKING_MODELS; do
        while (( $(jobs -p | wc -l) >= MAXJOBS )); do 
            echo "Waiting for evaluation jobs to complete..."
            sleep 10
        done
        
        # Fixed sed command - replace .pt with _${DATASET}.evallog
        LOGFILE=$(echo "$MODEL" | sed "s|\.pt|_${DATASET}.evallog|g")
        echo "Evaluating $MODEL on $DATASET"
        echo "Log file: $LOGFILE"
        
        nice python -u evalCli.py \
            --dataset $DATASET \
            --fixed_eval \
            --test_episodes 240 \
            --viz_episodes 10 \
            --walking True \
            --diffusionx 1.0 \
            --model_fname $MODEL >> "$LOGFILE" 2>&1 &
    done
done

echo "All evaluation jobs submitted. Monitor with:"
echo "tail -f ${ZOODIR}*.evallog"

wait
echo "All walking agent evaluations completed!"
