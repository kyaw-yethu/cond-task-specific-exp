#!/bin/bash
# Runs ON the node. The whole trajectory calibration: both clip geometries
# against all three intrinsics provenances.
#
# Two ranks per GPU, because the SLAM half is single-threaded OpenCV on the CPU
# and takes three times as long as the depth pass, so a rank leaves its GPU idle
# most of the time.
set -u
PROJ=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$PROJ" || exit 1
export LD_LIBRARY_PATH="$PROJ/.runtime/cudafix:${LD_LIBRARY_PATH:-}"
export HF_HOME=/root/driving-gen/ckpt/hf
export YOLO_CONFIG_DIR=/root/driving-gen/ckpt/ultralytics
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

NGPU=$(nvidia-smi -L | wc -l)
PER_GPU=${PER_GPU:-2}
WORLD=$((NGPU * PER_GPU))

for GEOM in dg stage2; do
  for MODE in predicted calibrated_slam calibrated_both; do
    LOGS=/root/driving-gen/logs/$GEOM-$MODE
    mkdir -p "$LOGS"
    echo
    echo "######## $GEOM / $MODE  ($WORLD ranks) ########"
    date -u +"start %H:%M:%S"
    pids=()
    for r in $(seq 0 $((WORLD-1))); do
      CUDA_VISIBLE_DEVICES=$((r % NGPU)) python3 -u dg_eval/scripts/dg_extract.py \
          --clips /root/driving-gen/clips/$GEOM --out /root/driving-gen/traj/$GEOM \
          --intrinsics "$MODE" --rank "$r" --world "$WORLD" \
          > "$LOGS/rank$r.log" 2>&1 &
      pids+=($!)
    done
    wait "${pids[@]}"
    date -u +"done  %H:%M:%S"
    echo "npz: $(ls /root/driving-gen/traj/$GEOM/$MODE 2>/dev/null | grep -c '\.npz$')"
    grep -h "Traceback\|Error" "$LOGS"/rank*.log | sort | uniq -c | head -5
  done
done

echo
echo "######## totals ########"
for GEOM in dg stage2; do
  for MODE in predicted calibrated_slam calibrated_both; do
    echo "$GEOM/$MODE: $(ls /root/driving-gen/traj/$GEOM/$MODE 2>/dev/null | grep -c '\.npz$')"
  done
done
df -h /root | tail -1
