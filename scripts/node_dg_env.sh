#!/bin/bash
# Runs ON the node. Build the environment DrivingGen's trajectory extractor needs,
# on top of the system python (torch 2.3.1+cu121) rather than a fresh conda env,
# because the two heavy packages both install --no-deps:
#
#   * UniDepth pins torch>=2.4 / numpy>=2 in requirements.txt, but nothing in
#     unidepth/ uses either. xformers is imported behind try/except and its own
#     attention.py says new torch SDPA makes it unnecessary.
#   * yolov10 is an ultralytics fork, so its declared deps would pull a second
#     torch.
#
# cv2 needs libGL, which the image lacks; that is what makes `import cv2` fail.
set -u
DG=${DG:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/third_party/DrivingGen/third_parties}
export DEBIAN_FRONTEND=noninteractive
export PIP_DISABLE_PIP_VERSION_CHECK=1

echo "############ 1. system libs ############"
# libGL for cv2, which is why `import cv2` fails on a fresh image; ffmpeg for
# reading generated video in dg_prepare_generated.py
apt-get update -qq 2>&1 | tail -2
apt-get install -y -qq libgl1 libglib2.0-0 ffmpeg 2>&1 | tail -3
python3 -c "import cv2; print('cv2', cv2.__version__, 'SIFT', hasattr(cv2,'SIFT_create'))"
ffmpeg -version 2>/dev/null | head -1 || echo "  ffmpeg MISSING"

echo
echo "############ 2. leaf python deps ############"
pip3 install -q \
  einops timm huggingface_hub h5py tables wandb termcolor tabulate \
  imageio trimesh pandas py-cpuinfo psutil thop pyyaml tqdm requests \
  2>&1 | grep -viE "already satisfied" | tail -15

echo
echo "############ 3. UniDepth + yolov10, no deps ############"
pip3 install -q --no-deps -e "$DG/UniDepth"  2>&1 | tail -3
pip3 install -q --no-deps -e "$DG/yolov10"   2>&1 | tail -3

echo
echo "############ 4. import check ############"
export LD_LIBRARY_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/.runtime/cudafix:${LD_LIBRARY_PATH:-}
python3 - <<'PY'
import traceback
for mod, what in [("torch","torch"), ("cv2","cv2"), ("scipy","scipy"),
                  ("unidepth.models","UniDepth"), ("ultralytics","yolov10 fork")]:
    try:
        m = __import__(mod, fromlist=["x"])
        print("  OK   %-16s %s" % (what, getattr(m, "__version__", "")))
    except Exception:
        print("  FAIL %-16s" % what); traceback.print_exc(limit=2)
import torch
print("  cuda available:", torch.cuda.is_available(), "| devices:", torch.cuda.device_count())
from unidepth.models import UniDepthV2
print("  UniDepthV2 imported")
from ultralytics import YOLOv10
print("  YOLOv10 imported")
PY
