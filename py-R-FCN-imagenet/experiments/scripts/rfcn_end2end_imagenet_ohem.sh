#!/bin/bash
# Usage:
# ./experiments/scripts/rfcn_end2end_ohem.sh GPU NET DATASET [options args to {train,test}_net.py]
# DATASET is either pascal_voc or coco.
#
# Example:
# ./experiments/scripts/rfcn_end2end_ohem.sh 0 ResNet50 pascal_voc \
#   --set EXP_DIR foobar RNG_SEED 42 TRAIN.SCALES "[400, 500, 600, 700]"

set -x
set -e

export PYTHONUNBUFFERED="True"

#GPU_ID=$1
#NET=$2
#NET_lc=${NET,,}
#DATASET=$3
GPU_ID=$1
NET=$2
NET_lc=${NET,,}
ITERS=999999999
DATASET_TRAIN=imagenet_DET_train
DATASET_TEST=imagenet_DET_val
PT_DIR="imagenet"
array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:2:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

LOG="experiments/logs/rfcn_end2end_${NET}_${EXTRA_ARGS_SLUG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_net_imagenet.py --gpu ${GPU_ID} \
  --solver models/${PT_DIR}/${NET}/rfcn_end2end/solver_ohem.prototxt \
  --weights data/imagenet_models/${NET}-model.caffemodel \
  --imdb ${DATASET_TRAIN} \
  --iters ${ITERS} \
  --cfg experiments/cfgs/rfcn_end2end_ohem.yml \
  ${EXTRA_ARGS}


set +x
NET_FINAL=`tail -n 100 ${LOG} | grep -B 1 "done solving" | grep "Wrote snapshot" | awk '{print $4}'`
set -x

time ./tools/test_net_imagenet.py --gpu ${GPU_ID} \
  --def models/${PT_DIR}/${NET}/rfcn_end2end/test_agnostic.prototxt \
  --net ${NET_FINAL} \
  --imdb ${DATASET_TEST} \
  --cfg experiments/cfgs/rfcn_end2end_ohem.yml \
  ${EXTRA_ARGS}
