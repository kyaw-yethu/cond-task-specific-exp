#!/bin/bash
# Runs ON the node. One GPU's share of the k=1 dit_large screen: for each
# condition in turn, build its fixed-k=1 cache, train it, evaluate it.
#
#   node_k1_chain.sh <gpu> <tag:cond:cond-run:width> [...]
#   node_k1_chain.sh 3 V1:traj_vae:waypoint_vae_lr5x:1 N1:none:-:0
#
# width is the raw frames the encoder sees: 2 for the ViTs (raw 0 twice, for the
# tubelet), 1 for the causal Traj-VAE (raw 0 alone). Ignored for N.
#
# Each step is skipped when its output is already complete, so relaunching after
# an interruption resumes rather than redoes. An interrupted training restarts
# from epoch 1, since train_dit_cond.py does not resume.
set -u
PROJ=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export LD_LIBRARY_PATH="$PROJ/.runtime/cudafix:${LD_LIBRARY_PATH:-}"
export GPU=${1:?usage: node_k1_chain.sh <gpu> <tag:cond:run:width> [...]}; shift
export CACHE_ROOT=/dev/shm/k1cache
export CONFIG=$PROJ/configs/dit_cond_large.yaml
cd "$PROJ" || exit 1
mkdir -p "$CACHE_ROOT" out/logs

log () { echo "[gpu$GPU $(date -u '+%m-%d %H:%M:%SZ')] $*"; }

epochs_done () {
  python3 - "$1" <<'PY'
import json, sys, pathlib
p = pathlib.Path(f"checkpoints/waymo/{sys.argv[1]}/history.json")
try:
    print(len(json.loads(p.read_text())))
except Exception:
    print(0)
PY
}

evaluated () {
  python3 - "$1" <<'PY'
import json, sys
try:
    m = json.load(open(f"out/eval_{sys.argv[1]}/metrics.json"))
    print(1 if m.get("num_samples") == 4000 else 0)
except Exception:
    print(0)
PY
}

for SPEC in "$@"; do
  IFS=: read -r TAG COND RUN WIDTH <<< "$SPEC"
  RUNARG=""; [ "$COND" != "none" ] && RUNARG="$RUN"

  # ---- cache ----
  if [ -f "$CACHE_ROOT/$TAG/cache_meta.json" ]; then
    log "$TAG cache present"
  else
    log "$TAG building cache (cond $COND, width $WIDTH)"
    rm -rf "$CACHE_ROOT/$TAG"
    WARG=""; [ "$COND" != "none" ] && WARG="--context-width $WIDTH"
    bash scripts/node_build_cache.sh "$TAG" "$COND" $RUNARG --fixed-k 1 $WARG \
      > "out/logs/k1_cache_$TAG.log" 2>&1 \
      || { log "$TAG cache FAILED, see out/logs/k1_cache_$TAG.log"; continue; }
  fi

  # ---- train ----
  if [ "$(epochs_done "$TAG")" -ge 100 ]; then
    log "$TAG already trained"
  else
    log "$TAG training"
    bash scripts/node_train_cond.sh "$TAG" "$COND" $RUNARG \
      > "out/logs/k1_train_$TAG.log" 2>&1 \
      || { log "$TAG training FAILED, see out/logs/k1_train_$TAG.log"; continue; }
    log "$TAG trained, $(epochs_done "$TAG") epochs"
  fi

  # ---- eval ----
  if [ "$(evaluated "$TAG")" = "1" ]; then
    log "$TAG already evaluated"
  else
    log "$TAG evaluating"
    bash scripts/node_eval_cond.sh "$TAG" > "out/logs/k1_eval_$TAG.log" 2>&1 \
      && log "$TAG eval done" \
      || { log "$TAG eval FAILED, see out/logs/k1_eval_$TAG.log"; continue; }
  fi
  # the cache lives in RAM and is only read by training
  rm -rf "$CACHE_ROOT/$TAG"
done
log "chain finished"
