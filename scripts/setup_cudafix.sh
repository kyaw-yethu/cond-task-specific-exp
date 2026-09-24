#!/bin/bash
# The cluster-3090 image ships a forward-compat libcuda (530) that the host's 525
# driver cannot use, so torch reports CUDA error 804 although nvidia-smi lists every
# GPU. Point .runtime/cudafix at the host's own libcuda; the node_*.sh drivers put it
# first on LD_LIBRARY_PATH. Do not run ldconfig afterwards, it restores the fault.
set -eu
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
LIB=/usr/lib/x86_64-linux-gnu/libcuda.so.$VER
[ -e "$LIB" ] || { echo "no $LIB"; exit 1; }
mkdir -p "$ROOT/.runtime/cudafix"
ln -sf "$LIB" "$ROOT/.runtime/cudafix/libcuda.so.1"
ln -sf "$LIB" "$ROOT/.runtime/cudafix/libcuda.so"
LD_LIBRARY_PATH="$ROOT/.runtime/cudafix:${LD_LIBRARY_PATH:-}" \
  python3 -c "import torch; print('cuda available:', torch.cuda.is_available())"
