#!/bin/bash
# Runs ON the node. Train the ego-motion probe on the 40,000-clip train split
# and report the two floors plus the two sanity controls.
#   node_train_probe.sh [extra args to scripts/train_probe.py]
#   node_train_probe.sh --epochs 3 --val-frac 0.02      # quick shakeout
set -u
PROJ=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export LD_LIBRARY_PATH="$PROJ/.runtime/cudafix:${LD_LIBRARY_PATH:-}"
cd "$PROJ" || exit 1

python3 -u scripts/train_probe.py \
  --jdm "$PROJ/third_party/Task_specific_JDM" \
  --dataset waymo \
  --ckpt-dir "$PROJ/checkpoints" \
  --dataset-dir "$PROJ/data/driving_wp_f16/train" \
  --val-dataset-dir "$PROJ/data/driving_wp_f16/test" \
  --out-dir "$PROJ/out/probe" \
  --epochs 20 --batch 64 --workers 8 "$@"
