#!/usr/bin/env bash
# One-time rsync of GLM-5.1 checkpoint to /tmp for faster loads (BF16 or FP8).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/glm51_env.sh
source "${REPO_ROOT}/scripts/glm51_env.sh"

SRC="${SRC:-$GLM51_ARTIFACTS_DIR}"
DST="${DST:-$GLM51_LOCAL_DIR}"

if [[ ! -d "$SRC" ]]; then
  echo "Source missing: $SRC (run: python scripts/download_glm_5_1.py --variant $GLM51_VARIANT)" >&2
  exit 1
fi

SRC_BYTES=$(du -sb "$SRC" | awk '{print $1}')
if [[ -d "$DST" ]]; then
  DST_BYTES=$(du -sb "$DST" 2>/dev/null | awk '{print $1}' || echo 0)
  if [[ "$DST_BYTES" -ge "$(( SRC_BYTES * 99 / 100 ))" ]]; then
    echo "Staged copy already present at $DST (${DST_BYTES} bytes ≈ source)"
    exit 0
  fi
  echo "Incomplete staged copy ($DST_BYTES / $SRC_BYTES bytes); resuming rsync"
fi

echo "Staging $GLM51_VARIANT: $SRC -> $DST"
START=$(date +%s)
mkdir -p "$(dirname "$DST")"
if command -v rsync >/dev/null 2>&1; then
  rsync -a --info=progress2 "$SRC/" "$DST/"
else
  mkdir -p "$DST"
  cp -a "$SRC/." "$DST/"
fi
ELAPSED=$(( $(date +%s) - START ))
echo "stage copy done in ${ELAPSED}s"

SHARD=$(find "$DST" -name '*.safetensors' | head -1)
if [[ -n "$SHARD" ]]; then
  echo "Spot-check read: $SHARD"
  dd if="$SHARD" of=/dev/null bs=1M count=512 status=progress 2>&1 || true
fi

du -sh "$SRC" "$DST"
