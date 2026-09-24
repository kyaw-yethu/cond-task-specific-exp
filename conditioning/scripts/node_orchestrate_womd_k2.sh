#!/bin/bash
# Runs ON the node, unattended. The k=2 dit_large screen on womd_7hz_f33: N, D,
# T and G, one GPU each, every one built, trained and evaluated by node_chain.sh,
# then the qualitative comparison and WOMD_K2_RESULTS.md.
#
# Waits first for the ego-motion probe retrain (out/probe_womd), which holds
# GPU 0 and which the evaluation reads. Caches live in /dev/shm (about 227 GB
# for the four at k=2), are only read by training, and are deleted after each
# condition's evaluation. Checkpoint selection uses the val split; the final
# evaluation uses all 4,000 test clips.
# Relaunching resumes: every step skips output that is already complete.
set -u
PROJ=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export LD_LIBRARY_PATH="$PROJ/.runtime/cudafix:${LD_LIBRARY_PATH:-}"
cd "$PROJ" || exit 1
mkdir -p out/logs

export FIXED_K=2 LOG_PREFIX=womd_k2 CACHE_ROOT=/dev/shm/womd_k2cache
export DATA=$PROJ/data/womd_7hz_f33 VAL_SPLIT=val EVAL_SPLIT=test
export CONFIG=$PROJ/conditioning/configs/dit_cond_large_womd.yaml
export VAE_RUN=vae_miniwan_womd
export PROBE=$PROJ/out/probe_womd/probe.pt

log () { echo "[orch-womd-k2 $(date -u '+%m-%d %H:%M:%SZ')] $*"; }

log "waiting for the probe retrain to finish"
while pgrep -f "conditioning/scripts/train_probe.py" > /dev/null; do sleep 60; done
if [ -f out/probe_womd/probe_floors.json ]; then
  log "probe done: $(python3 -c 'import json; f=json.load(open("out/probe_womd/probe_floors.json")); print("real ADE %.3f m, VAE round-trip ADE %.3f m" % (f["floor_real"]["ade"], f["floor_vae_roundtrip"]["ade"]))')"
else
  log "WARNING: probe finished without probe_floors.json; training anyway, evaluation will fail without $PROBE"
fi

log "launching four chains"
bash conditioning/scripts/node_chain.sh 0 G_k2:vjepa2:vjepa2_G_vith256:6      > out/logs/womd_k2_chain0.log 2>&1 &
bash conditioning/scripts/node_chain.sh 1 D_k2:drivejepa:drivejepa_D_vitl256:6 > out/logs/womd_k2_chain1.log 2>&1 &
bash conditioning/scripts/node_chain.sh 2 T_k2:drivejepa:drivejepa_T_vitl256:6 > out/logs/womd_k2_chain2.log 2>&1 &
bash conditioning/scripts/node_chain.sh 3 N_k2:none:-:0                        > out/logs/womd_k2_chain3.log 2>&1 &
wait
log "all chains finished"
cat out/logs/womd_k2_chain*.log

log "building the qualitative comparison on GPU 0"
CUDA_VISIBLE_DEVICES=0 python3 -u conditioning/scripts/make_comparison.py \
  --jdm "$PROJ/third_party/Task_specific_JDM" --dataset waymo \
  --ckpt-dir "$PROJ/checkpoints" \
  --dataset-dir "$DATA/$EVAL_SPLIT" --vae-run "$VAE_RUN" \
  --tags N_k2 T_k2 D_k2 G_k2 --which best --num-clips 6 --steps 50 --seed 0 \
  --fixed-k 2 --out-dir "$PROJ/out/comparison_womd_k2" \
  > out/logs/womd_k2_comparison.log 2>&1 \
  && log "comparison done" || log "comparison FAILED, see out/logs/womd_k2_comparison.log"

log "writing WOMD_K2_RESULTS.md"
python3 -u conditioning/scripts/make_results.py --out-root "$PROJ/out" --proj "$PROJ" \
  --suffix _k2 --out-file WOMD_K2_RESULTS.md --comparison-dir comparison_womd_k2 \
  > out/logs/womd_k2_results.log 2>&1 \
  && log "WOMD_K2_RESULTS.md written" || log "results FAILED, see out/logs/womd_k2_results.log"
log "ALL DONE"
