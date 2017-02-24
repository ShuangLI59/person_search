#!/usr/bin/env bash

if [[ $# -ne 2 ]]; then
  echo "$(basename $0) NET ITER"
  exit
fi

NET=$1
ITER=$2

mpirun -n 8 python tools/eval_train.py \
  --gpu 0,1,2,3,4,5,6,7 \
  --def models/psdb/${NET}/eval_train.prototxt \
  --net output/psdb_train/${NET}_iter_${ITER}.caffemodel \
  --cfg experiments/cfgs/train.yml \
  --imdb psdb_train \
  --det_thresh 0.5 \
  2>&1 | tee experiments/logs/${NET}_iter_${ITER}_eval_train.log

