#!/usr/bin/env bash

set -x
NGPUS=$1
PY_ARGS=${@:2}

nohup python -m torch.distributed.launch --nproc_per_node=${NGPUS} train.py --launcher pytorch ${PY_ARGS} &

