#!/bin/bash
# Runs ON the node. Evaluate one finished condition on both axes.
#   GPU=2 node_eval_cond.sh V
set -u
PROJ=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export LD_LIBRARY_PATH="$PROJ/.runtime/cudafix:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=${GPU:-5}   # 0-4 are the training runs
cd "$PROJ" || exit 1
TAG=${1:?usage: node_eval_cond.sh <tag> [args...]}; shift

python3 -u conditioning/scripts/eval_cond.py \
  --jdm "$PROJ/third_party/Task_specific_JDM" --dataset waymo \
  --tag "$TAG" --which best \
  --ckpt-dir "$PROJ/checkpoints" \
  --dataset-dir "$PROJ/data/driving_wp_f16/test" \
  --probe "$PROJ/out/probe_vaeaug/probe.pt" \
  --eval-dir "$PROJ/out/eval_$TAG" \
  --batch 16 --steps 50 "$@"
