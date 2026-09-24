#!/bin/bash
# Fetch the parts of DrivingGen that are not vendored: UniDepth and yolov10 for the
# ego-trajectory extractor, MTR for FTD. Extra third_parties names can be passed as
# arguments, e.g. `setup_third_party.sh stylegan-v` for FVD.
#
# They land in third_party/DrivingGen/third_parties/, which is gitignored and is
# where scripts/node_dg_env.sh and scripts/dg_ftd.py look by default.
set -eu
DG_URL=https://github.com/youngzhou1999/DrivingGen.git
DG_REV=48ed35695855ef17d7a7cbd4adc0e8bd5fcc8223
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DST=$ROOT/third_party/DrivingGen/third_parties
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

PARTS="UniDepth yolov10 MTR $*"
git clone -q --filter=blob:none --no-checkout "$DG_URL" "$TMP/dg"
cd "$TMP/dg"
git sparse-checkout set $(for p in $PARTS; do echo "third_parties/$p"; done)
git checkout -q "$DG_REV"
mkdir -p "$DST"
for p in $PARTS; do
  rm -rf "$DST/$p"
  cp -r "third_parties/$p" "$DST/$p"
  echo "fetched $p"
done
