#!/bin/bash
# Runs ON the node. Does every figure the docs embed actually exist here?
#
# Worth its own script: out/ is excluded from the code sync by design, so a
# figure generated locally never reaches the node and a doc that references it
# renders a broken image there. Anything the docs embed must be produced BY a
# node script, not fetched and forgotten.
set -u
PROJ=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$PROJ" || exit 1

echo "=== figures referenced by the docs ==="
MISSING=0
for f in $(grep -ho '](out/[^)]*)' ./*.md | sed 's/^](//; s/)$//' | sort -u); do
  if [ -f "$f" ]; then
    printf 'OK    %-42s %s\n' "$f" "$(du -h "$f" | cut -f1)"
  else
    printf 'MISS  %-42s\n' "$f"
    MISSING=$((MISSING + 1))
  fi
done
echo
echo "missing: $MISSING"
exit $MISSING
