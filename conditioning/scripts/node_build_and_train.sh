#!/bin/bash
# Runs ON the node. Build one condition's fixed-k cache, then train it.
#   node_build_and_train.sh <gpu> <tag> <cond> [cond-run]
#   node_build_and_train.sh 1 N none
#   node_build_and_train.sh 2 V traj_vae waypoint_vae_lr5x
#
# Sequential on purpose: training reads the cache this writes. The GPU is a
# positional rather than an env var, because launch_on_node.sh does not forward
# the environment across ssh.
set -u
PROJ=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export GPU=${1:?usage: node_build_and_train.sh <gpu> <tag> <cond> [cond-run] [cache-root]}
TAG=${2:?usage: node_build_and_train.sh <gpu> <tag> <cond> [cond-run]}
COND=${3:?usage: node_build_and_train.sh <gpu> <tag> <cond> [cond-run]}
RUN=${4:-}
export CACHE_ROOT=${5:-$PROJ/cache}

echo "############ building cache for $TAG ($COND) on GPU $GPU in $CACHE_ROOT ############"
mkdir -p "$CACHE_ROOT"
bash "$PROJ/scripts/node_build_cache.sh" "$TAG" "$COND" $RUN || exit 1

echo
echo "############ training $TAG on GPU $GPU ############"
bash "$PROJ/scripts/node_train_cond.sh" "$TAG" "$COND" $RUN
