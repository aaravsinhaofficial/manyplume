#!/bin/bash

MODEL="/Users/aaravsinha/manyplume/oneplume/ppo/trained_models/Walking20250807/plume_Walk-v17_20250807_VRNN_constantx5b5noisy3x5b5_stepoob_bx2.72_t100000010000_q0.40.5_dmx0.80.8_dmn0.70.4_h64_wd0.00012_n1_walking_seed30970f1.pt"
MAXJOBS=3
DATASETS="constantx5b5"

for DATASET in $DATASETS; do
    while (( $(jobs -p | wc -l) >= MAXJOBS )); do echo "Sleeping..."; sleep 10; done

    LOGFILE=$(basename $MODEL .pt)_${DATASET}.evallog
    SPARSE_MODIFIER=""
    if [[ $DATASET == "constantx5b5" ]]; then
        SPARSE_MODIFIER="--test_sparsity"
    fi
    nice python -u evalCli.py --dataset constantx5b5 --viz_episodes 20 --model_fname $MODEL 
done

tail -f *.evalloge