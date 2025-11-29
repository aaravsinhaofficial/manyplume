#!/bin/bash

MODEL_DIR="/Users/aaravsinha/manyplume/oneplume/ppo/trained_models/Walking20250807/plume_Walk-v17_20250807_VRNN_constantx5b5noisy3x5b5_stepoob_bx2.72_t100000010000_q0.40.5_dmx0.80.8_dmn0.70.4_h64_wd0.00012_n1_walking_seed30970f1"
LOGFILE="${MODEL_DIR}/posteval.log"

# Run postEvalCli.py for all datasets
python -u postEvalCli.py --model_dir $MODEL_DIR --viz_episodes 7 --walking True>> $LOGFILE 2>&1 &

# Monitor log output
tail -f $LOGFILE
