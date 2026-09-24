#!/bin/bash
# Runs ON the node. Evaluate one finished condition on both axes.
#   GPU=2 node_eval_cond.sh V
# Environment, defaults in brackets: DATA [$PROJ/data/driving_wp_f16],
# EVAL_SPLIT [test], VAE_RUN [vae_50e24b], PROBE [$PROJ/out/probe_vaeaug/probe.pt].
set -u
PROJ=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
DATA=${DATA:-$PROJ/data/driving_wp_f16}
EVAL_SPLIT=${EVAL_SPLIT:-test}
VAE_RUN=${VAE_RUN:-vae_50e24b}
PROBE=${PROBE:-$PROJ/out/probe_vaeaug/probe.pt}
export LD_LIBRARY_PATH="$PROJ/.runtime/cudafix:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=${GPU:-5}
cd "$PROJ" || exit 1
TAG=${1:?usage: node_eval_cond.sh <tag> [args...]}; shift

python3 -u conditioning/scripts/eval_cond.py \
  --jdm "$PROJ/third_party/Task_specific_JDM" --dataset waymo \
  --tag "$TAG" --which best \
  --ckpt-dir "$PROJ/checkpoints" \
  --dataset-dir "$DATA/$EVAL_SPLIT" \
  --probe "$PROBE" \
  --vae-run "$VAE_RUN" \
  --eval-dir "$PROJ/out/eval_$TAG" \
  --batch 16 --steps 50 "$@"
