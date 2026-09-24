#!/bin/bash
# Runs ON the node. Train one condition from the fixed-k cache built by
# conditioning/scripts/build_cache.py.
#
#   GPU=0 node_train_cond.sh T drivejepa drivejepa_T_vitl256
#   GPU=1 node_train_cond.sh N none
#
# GPU=<n> pins the run to one card so conditions run concurrently. Effective
# batch must stay 32 to match dit10m32b100e's recipe, and a single-card run
# therefore passes --batch 32 rather than the config's per-rank 16.
# Environment, defaults in brackets: CACHE_ROOT [$PROJ/cache],
# DATA [$PROJ/data/driving_wp_f16], VAL_SPLIT [test],
# CONFIG [conditioning/configs/dit_cond.yaml], VAE_RUN [vae_50e24b].
set -u
PROJ=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
CACHE_ROOT=${CACHE_ROOT:-$PROJ/cache}
DATA=${DATA:-$PROJ/data/driving_wp_f16}
VAL_SPLIT=${VAL_SPLIT:-test}
CONFIG=${CONFIG:-$PROJ/conditioning/configs/dit_cond.yaml}
VAE_RUN=${VAE_RUN:-vae_50e24b}
export LD_LIBRARY_PATH="$PROJ/.runtime/cudafix:${LD_LIBRARY_PATH:-}"
if [ -n "${GPU:-}" ]; then export CUDA_VISIBLE_DEVICES="$GPU"; fi
cd "$PROJ" || exit 1

TAG=${1:?usage: node_train_cond.sh <tag> <cond> [cond-run] [args...]}; shift
COND=${1:?usage: node_train_cond.sh <tag> <cond> [cond-run] [args...]}; shift
RUN_ARG=""
if [ "$COND" != "none" ]; then
  RUN_ARG="--cond-run ${1:?a cond-run is required unless cond is none}"; shift
fi

echo "condition $TAG ($COND) on GPU ${GPU:-all}"
python3 -u conditioning/scripts/train_dit_cond.py \
  --jdm "$PROJ/third_party/Task_specific_JDM" \
  --dataset waymo \
  --config "$CONFIG" \
  --cond "$COND" $RUN_ARG --tag "$TAG" \
  --ckpt-dir "$PROJ/checkpoints" \
  --dataset-dir "$DATA/train" \
  --val-dataset-dir "$DATA/$VAL_SPLIT" \
  --cache-dir "$CACHE_ROOT/$TAG" \
  --vae-run "$VAE_RUN" \
  --batch 32 --val-every 5 --workers 6 "$@"
