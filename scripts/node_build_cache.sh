#!/bin/bash
# Runs ON the node. Build the fixed-k=2 cache for one condition.
#   node_build_cache.sh <tag> <cond> [cond-run]
#   node_build_cache.sh T drivejepa drivejepa_T_vitl256
#   node_build_cache.sh N none
#
# Each ViT encoder's token cache is ~63 GB, so only one fits at a time. The
# x0 latent cache (~921 MB) is shared across conditions and is written by
# whichever build runs first; later builds overwrite it identically.
set -u
PROJ=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# CACHE_ROOT lets a cache live in /dev/shm (504 GB of RAM, already
# mounted and writable) instead of on the 98 GB disk. No faster -- the
# page cache already keeps a re-read file resident -- but it stops disk
# space serialising the conditions. Lost on a workspace stop.
CACHE_ROOT=${CACHE_ROOT:-$PROJ/cache}
export LD_LIBRARY_PATH="$PROJ/.runtime/cudafix:${LD_LIBRARY_PATH:-}"
if [ -n "${GPU:-}" ]; then export CUDA_VISIBLE_DEVICES="$GPU"; fi
cd "$PROJ" || exit 1

TAG=${1:?usage: node_build_cache.sh <tag> <cond> [cond-run]}; shift
COND=${1:?usage: node_build_cache.sh <tag> <cond> [cond-run]}; shift
RUN_ARG=""
if [ "$COND" != "none" ]; then
  RUN_ARG="--cond-run ${1:?a cond-run is required unless cond is none}"; shift
fi

df -h /root | tail -1
python3 -u scripts/build_cache.py \
  --jdm "$PROJ/third_party/Task_specific_JDM" \
  --dataset waymo \
  --config "$PROJ/configs/dit_cond.yaml" \
  --ckpt-dir "$PROJ/checkpoints" \
  --cache-dir "$CACHE_ROOT/$TAG" \
  --cond "$COND" $RUN_ARG --tag "$TAG" \
  --dataset-dir "$PROJ/data/driving_wp_f16/train" \
  --val-dataset-dir "$PROJ/data/driving_wp_f16/test" \
  --batch 16 --workers 8 "$@"

echo
du -sh "$CACHE_ROOT/$TAG"
df -h /root | tail -1
