#!/usr/bin/env bash

if [[ $# -ne 3 ]]; then
  echo "$(basename $0) NET ITER EXP_DIR"
  exit
fi

NET=$1
ITER=$2
EXP_DIR=$3

mkdir -p experiments/logs/${EXP_DIR}

mpirun -n 8 python2 tools/eval_test.py \
  --gpu 0,1,2,3,4,5,6,7 \
  --gallery_def models/psdb/${NET}/eval_gallery.prototxt \
  --probe_def models/psdb/${NET}/eval_probe.prototxt \
  --net output/psdb_train/${EXP_DIR}/${NET}_iter_${ITER}.caffemodel \
  --cfg experiments/cfgs/${NET}.yml \
  --imdb psdb_test \
  --gallery_size 100 \
  --det_thresh 0.5 \
  --use_gt \
  --set EXP_DIR "${EXP_DIR}" \
  2>&1 | tee experiments/logs/${EXP_DIR}/${NET}_iter_${ITER}_eval_test_with_gt.log

