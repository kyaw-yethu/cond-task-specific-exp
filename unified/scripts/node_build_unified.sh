#!/bin/bash
# Runs ON the node, detached. Both datasets into the unified clip set.
# One log per dataset, and a lock so a second master cannot start beside the
# first and fight it for the same records cache.
set -u
PROJ=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
LOGDIR=/root/unified
MASTER=$LOGDIR/build.log

if [ "${DETACHED:-0}" != "1" ]; then
  mkdir -p "$LOGDIR"
  if [ -e "$LOGDIR/master.pid" ] && kill -0 "$(cat $LOGDIR/master.pid)" 2>/dev/null; then
    echo "a master is already running as pid $(cat $LOGDIR/master.pid)"; exit 1
  fi
  DETACHED=1 setsid nohup bash "$0" "$@" >> "$MASTER" 2>&1 < /dev/null &
  echo $! > "$LOGDIR/master.pid"
  sleep 2
  echo "detached as pid $(cat $LOGDIR/master.pid), logs in $LOGDIR"
  exit 0
fi

cd "$PROJ" || exit 1
export PYTHONWARNINGS=ignore
W=${W:-16}
trap 'rm -f '"$LOGDIR"'/master.pid' EXIT

for DS in waymo nuscenes; do
  echo "################ $DS $(date -u +%H:%M:%S) ################"
  python3 -u unified/scripts/build_unified.py --dataset "$DS" --workers "$W" \
    > "$LOGDIR/$DS.log" 2>&1
  echo "  $DS finished $(date -u +%H:%M:%S) rc=$?"
  tail -4 "$LOGDIR/$DS.log"
done
echo "UNIFIED BUILD DONE $(date -u +%H:%M:%S)"
df -h /root /dev/shm | tail -2
