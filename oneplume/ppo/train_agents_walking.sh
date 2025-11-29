#!/bin/bash

NUMPROC=1
SHAPE="step oob"  # Add walking-specific reward shaping- metabolic found later
BIRTHX="2.7 2"
DATASET="constantx5b5 noisy3x5b5"  # Use walking datasets
STEPS="1000000 10000"
VARX="0.4 0.5"
DMAX="0.8 0.8"
DMIN="0.7 0.4"
ALGO=ppo
HIDDEN=64
DECAY=0.00012

EXPT=Walking$(date '+%Y%m%d')
SAVEDIR=./trained_models/${EXPT}/
mkdir -p $SAVEDIR

MAXJOBS=3

for SEED in $(seq 3); do  
  while (( $(jobs -p | wc -l) >= MAXJOBS )); do sleep 10; done 
  RNNTYPE=VRNN
  SEED=$RANDOM

  DATASTRING=$(echo -e $DATASET | tr -d ' ')
  SHAPESTRING=$(echo -e $SHAPE | tr -d ' ')
  BXSTRING=$(echo -e $BIRTHX | tr -d ' ')
  TSTRING=$(echo -e $STEPS | tr -d ' ')
  QVARSTR=$(echo -e $VARX | tr -d ' ')
  DMAXSTR=$(echo -e $DMAX | tr -d ' ')
  DMINSTR=$(echo -e $DMIN | tr -d ' ')

  OUTSUFFIX=Walk-v17_$(date '+%Y%m%d')_${RNNTYPE}_${DATASTRING}_${SHAPESTRING}_bx${BXSTRING}_t${TSTRING}_q${QVARSTR}_dmx${DMAXSTR}_dmn${DMINSTR}_h${HIDDEN}_wd${DECAY}_n${NUMPROC}_walking_seed${SEED}$(openssl rand -hex 1)
  echo $OUTSUFFIX

  nice python -u main.py --env-name plume \
    --recurrent-policy \
    --dataset $DATASET \
    --num-env-steps ${STEPS} \
    --birthx $BIRTHX  \
    --flipping True \
    --qvar $VARX \
    --save-dir $SAVEDIR \
    --log-interval 1 \
    --r_shaping $(echo -e $SHAPE) \
    --algo $ALGO \
    --seed ${SEED} \
    --squash_action True \
    --diff_max $DMAX \
    --diff_min $DMIN \
    --num-processes $NUMPROC \
    --num-mini-batch $NUMPROC \
    --odor_scaling True \
    --rnn_type ${RNNTYPE} \
    --hidden_size $HIDDEN \
    --weight_decay ${DECAY} \
    --walking True \
    --use-gae --num-steps 2048 --lr 3e-4 --entropy-coef 0.005 --value-loss-coef 0.5 --ppo-epoch 10 --gamma 0.99 --gae-lambda 0.95 --use-linear-lr-decay \
    --outsuffix ${OUTSUFFIX} > ${SAVEDIR}/${OUTSUFFIX}.log 2>&1 &
  echo "Sleeping.."
  sleep 0.5
done

tail -f ${SAVEDIR}/*.log
