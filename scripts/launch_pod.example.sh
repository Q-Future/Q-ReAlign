#!/usr/bin/env bash
# ===========================================================================
# OPTIONAL example: launch training on a GPU pod (Slurm / k8s / your scheduler).
#
# qalign itself does NOT need a scheduler — `qalign train --config ...` runs
# anywhere ms-swift runs. This file is just a template for submitting that
# command to a multi-GPU box. Adapt the submit line to your cluster, or delete it.
# ===========================================================================
set -euo pipefail

CONFIG=${CONFIG:-configs/onealign.yaml}
GPUS=${GPUS:-2}
REPO=$(cd "$(dirname "$0")/.." && pwd)

# --- single-node, N GPUs, run directly -------------------------------------
# `qalign train` reads `train.gpus` from the config; keep them consistent or
# override on the CLI:  qalign train --config $CONFIG --set train.gpus=$GPUS
cd "$REPO"
qalign train --config "$CONFIG" --set "train.gpus=$GPUS" full

# --- example: submit to a scheduler instead (uncomment + adapt) ------------
# sbatch --gres=gpu:$GPUS --cpus-per-task=$((GPUS*24)) --mem=$((GPUS*400))G \
#   --wrap "cd $REPO && qalign train --config $CONFIG full"
#
# Notes for QA-scale data:
#   * the dataloader is image-decode bound — give it many CPU workers + RAM
#     (the packed blob cache from `qalign cache` keeps the hot bytes resident);
#   * for large models use train.deepspeed: zero3 (or zero3_offload).
