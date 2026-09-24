#!/bin/bash
# Runs ON the node, unattended. Waits for every condition to finish training,
# evaluates each on both axes, builds the qualitative comparison, and writes
# RESULTS.md.
#
# Designed to be launched detached and left alone. Each stage is idempotent and
# skipped if its output already exists, so a rerun after an interruption picks
# up where it stopped rather than redoing hours of work.
set -u
PROJ=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export LD_LIBRARY_PATH="$PROJ/.runtime/cudafix:${LD_LIBRARY_PATH:-}"
cd "$PROJ" || exit 1
TAGS="N T V D G"
EVAL_GPU=5
COMP_GPU=6

log () { echo "[orch $(date -u +%H:%M:%SZ)] $*"; }

# ---------- 1. wait for training, evaluating each as it lands ----------
for TAG in $TAGS; do
  log "waiting for $TAG to reach 100 epochs"
  while true; do
    DONE=$(python3 - "$TAG" <<'PY'
import json, sys, pathlib
p = pathlib.Path(f"checkpoints/waymo/{sys.argv[1]}/history.json")
try:
    print(1 if len(json.loads(p.read_text())) >= 100 else 0)
except Exception:
    print(0)
PY
)
    ALIVE=$(pgrep -cf "train_dit_cond.py.*--tag $TAG " || true)
    [ "$DONE" = "1" ] && break
    # if training died before 100 epochs, stop waiting and evaluate what exists
    if [ "${ALIVE:-0}" = "0" ]; then
      log "WARNING: $TAG is not running and has not reached 100 epochs; evaluating last"
      break
    fi
    sleep 120
  done

  if [ -f "$PROJ/out/eval_$TAG/metrics.json" ] && \
     [ "$(python3 -c "import json;print(json.load(open('$PROJ/out/eval_$TAG/metrics.json'))['num_samples'])" 2>/dev/null)" = "4000" ]; then
    log "$TAG already evaluated on 4000 clips, skipping"
  else
    log "evaluating $TAG on GPU $EVAL_GPU"
    GPU=$EVAL_GPU bash "$PROJ/scripts/node_eval_cond.sh" "$TAG" \
      > "$PROJ/out/logs/eval_$TAG.log" 2>&1 \
      && log "$TAG eval done" || log "$TAG eval FAILED, see out/logs/eval_$TAG.log"
  fi
done

# ---------- 2. qualitative comparison ----------
log "building the qualitative comparison on GPU $COMP_GPU"
GPU=$COMP_GPU CUDA_VISIBLE_DEVICES=$COMP_GPU python3 -u conditioning/scripts/make_comparison.py \
  --jdm "$PROJ/third_party/Task_specific_JDM" --dataset waymo \
  --ckpt-dir "$PROJ/checkpoints" \
  --dataset-dir "$PROJ/data/driving_wp_f16/test" \
  --tags N T V D G --which best --num-clips 6 --steps 50 --seed 0 \
  --out-dir "$PROJ/out/comparison" \
  > "$PROJ/out/logs/comparison.log" 2>&1 \
  && log "comparison done" || log "comparison FAILED, see out/logs/comparison.log"

# ---------- 3. results document ----------
log "writing RESULTS.md"
python3 -u conditioning/scripts/make_results.py --out-root "$PROJ/out" --proj "$PROJ" \
  > "$PROJ/out/logs/results.log" 2>&1 \
  && log "RESULTS.md written" || log "RESULTS.md FAILED, see out/logs/results.log"

# ---------- 4. check every embedded figure exists here ----------
bash "$PROJ/scripts/node_check_figs.sh" || log "WARNING: some doc figures are missing"

log "ALL DONE"
echo
grep -E '^\| \*\*' "$PROJ/RESULTS.md" 2>/dev/null | head -30
