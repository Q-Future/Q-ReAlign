#!/usr/bin/env bash
# Thin convenience wrapper around `qalign train`.
#   bash scripts/train.sh configs/onealign.yaml          # full run
#   bash scripts/train.sh configs/example_iqa.yaml mini  # 10-step smoke test
set -euo pipefail
CFG=${1:?usage: train.sh <config.yaml> [full|mini]}
MODE=${2:-full}
exec qalign train --config "$CFG" "$MODE"
