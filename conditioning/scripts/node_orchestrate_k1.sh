#!/bin/bash
# Runs ON the node, unattended. The k=1 screen on the dit_large backbone: five
# conditions over four GPUs, each built, trained and evaluated by
# node_k1_chain.sh, then the qualitative comparison and STAGE1_K1_RESULTS.md.
#
# Tags carry a 1 suffix (N1 D1 T1 V1 G1) so the k=2 checkpoints, evals and
# RESULTS are untouched. V and N share GPU 3 concurrently: each dit_large run
# peaks near 9 GB of the card's 24, and running both at once is never slower
# than running them back to back.
# Relaunching resumes: every step skips output that is already complete.
set -u
PROJ=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export LD_LIBRARY_PATH="$PROJ/.runtime/cudafix:${LD_LIBRARY_PATH:-}"
cd "$PROJ" || exit 1
mkdir -p out/logs

log () { echo "[orch-k1 $(date -u '+%m-%d %H:%M:%SZ')] $*"; }

log "launching four chains"
bash conditioning/scripts/node_k1_chain.sh 0 G1:vjepa2:vjepa2_G_vith256:2      > out/logs/k1_chain0.log 2>&1 &
bash conditioning/scripts/node_k1_chain.sh 1 D1:drivejepa:drivejepa_D_vitl256:2 > out/logs/k1_chain1.log 2>&1 &
bash conditioning/scripts/node_k1_chain.sh 2 T1:drivejepa:drivejepa_T_vitl256:2 > out/logs/k1_chain2.log 2>&1 &
bash conditioning/scripts/node_k1_chain.sh 3 V1:traj_vae:waypoint_vae_lr5x:1    > out/logs/k1_chain3.log 2>&1 &
bash conditioning/scripts/node_k1_chain.sh 3 N1:none:-:0                        > out/logs/k1_chain4.log 2>&1 &
wait
log "all chains finished"
cat out/logs/k1_chain*.log

log "building the qualitative comparison on GPU 0"
CUDA_VISIBLE_DEVICES=0 python3 -u conditioning/scripts/make_comparison.py \
  --jdm "$PROJ/third_party/Task_specific_JDM" --dataset waymo \
  --ckpt-dir "$PROJ/checkpoints" \
  --dataset-dir "$PROJ/data/driving_wp_f16/test" \
  --tags N1 T1 V1 D1 G1 --which best --num-clips 6 --steps 50 --seed 0 \
  --fixed-k 1 --out-dir "$PROJ/out/comparison_k1" \
  > out/logs/k1_comparison.log 2>&1 \
  && log "comparison done" || log "comparison FAILED, see out/logs/k1_comparison.log"

log "writing STAGE1_K1_RESULTS.md"
python3 -u conditioning/scripts/make_results.py --out-root "$PROJ/out" --proj "$PROJ" \
  --suffix 1 --out-file STAGE1_K1_RESULTS.md --comparison-dir comparison_k1 \
  > out/logs/k1_results.log 2>&1 \
  && log "STAGE1_K1_RESULTS.md written" || log "results FAILED, see out/logs/k1_results.log"

bash conditioning/scripts/node_check_figs.sh || log "WARNING: some doc figures are missing"
log "ALL DONE"
