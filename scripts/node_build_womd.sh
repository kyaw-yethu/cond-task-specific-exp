#!/bin/bash
# Render all three womd_7hz_f33 splits, then finalize. Resumable.
cd "$(dirname "${BASH_SOURCE[0]}")/.."
R=/opt/miniforge3/envs/render/bin/python
for s in val test train; do
  $R -u scripts/build_womd.py render --split $s --workers 96 2>&1 | grep --line-buffered -v -E "INFO\]|gobj|Warning|/dev/input|inotify|warn|Can not find" || exit 1
done
python3 -u scripts/build_womd.py finalize 2>&1 | grep -v -i warn
