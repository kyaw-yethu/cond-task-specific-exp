#!/bin/bash
# Runs ON the node. Train one condition of the stage-1 screen from the fixed-k
# cache built by conditioning/scripts/build_cache.py.
#
#   GPU=0 node_train_cond.sh T drivejepa drivejepa_T_vitl256
#   GPU=1 node_train_cond.sh N none
#
# GPU=<n> pins the run to one card so conditions run concurrently. Effective
# batch must stay 32 to match dit10m32b100e's recipe, and a single-card run
# therefore passes --batch 32 rather than the config's per-rank 16.
# CONFIG=<yaml> swaps the config, e.g. conditioning/configs/dit_cond_large.yaml for the
# dit_large backbone.
set -u
PROJ=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
# CACHE_ROOT lets a cache live in /dev/shm (504 GB of RAM, already
# mounted and writable) instead of on the 98 GB disk. No faster -- the
# page cache already keeps a re-read file resident -- but it stops disk
# space serialising the conditions. Lost on a workspace stop.
CACHE_ROOT=${CACHE_ROOT:-$PROJ/cache}
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
  --config "${CONFIG:-$PROJ/configs/dit_cond.yaml}" \
  --cond "$COND" $RUN_ARG --tag "$TAG" \
  --ckpt-dir "$PROJ/checkpoints" \
  --dataset-dir "$PROJ/data/driving_wp_f16/train" \
  --val-dataset-dir "$PROJ/data/driving_wp_f16/test" \
  --cache-dir "$CACHE_ROOT/$TAG" \
  --batch 32 --val-every 5 --workers 6 "$@"
