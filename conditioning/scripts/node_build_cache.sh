#!/bin/bash
# Runs ON the node. Build the fixed-k cache for one condition.
#   node_build_cache.sh <tag> <cond> [cond-run] [args...]
#   node_build_cache.sh T drivejepa drivejepa_T_vitl256
#   node_build_cache.sh N none
#
# Environment, defaults in brackets:
#   CACHE_ROOT  where the cache goes [$PROJ/cache]; /dev/shm holds several
#               encoders at once but is lost on a workspace stop
#   DATA        dataset root holding the splits [$PROJ/data/driving_wp_f16]
#   VAL_SPLIT   split cached as "val" [test]
#   CONFIG      config whose FRAMES and IMG set the latent grid
#               [conditioning/configs/dit_cond.yaml]
#   VAE_RUN     VAE under checkpoints/waymo/ [vae_50e24b]
#   GPU         pins the build to one card
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

TAG=${1:?usage: node_build_cache.sh <tag> <cond> [cond-run]}; shift
COND=${1:?usage: node_build_cache.sh <tag> <cond> [cond-run]}; shift
RUN_ARG=""
if [ "$COND" != "none" ]; then
  RUN_ARG="--cond-run ${1:?a cond-run is required unless cond is none}"; shift
fi

mkdir -p "$CACHE_ROOT"
df -h "$CACHE_ROOT" | tail -1
python3 -u conditioning/scripts/build_cache.py \
  --jdm "$PROJ/third_party/Task_specific_JDM" \
  --dataset waymo \
  --config "$CONFIG" \
  --ckpt-dir "$PROJ/checkpoints" \
  --cache-dir "$CACHE_ROOT/$TAG" \
  --cond "$COND" $RUN_ARG --tag "$TAG" \
  --dataset-dir "$DATA/train" \
  --val-dataset-dir "$DATA/$VAL_SPLIT" \
  --vae-run "$VAE_RUN" \
  --batch 16 --workers 8 "$@"

echo
du -sh "$CACHE_ROOT/$TAG"
df -h "$CACHE_ROOT" | tail -1
