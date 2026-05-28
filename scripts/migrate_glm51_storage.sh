#!/usr/bin/env bash
# Move GLM-5.1-FP8 off Lustre to /tmp, free space, wire BF16 on Lustre (symlink if full model won't fit).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FP8_SRC="${REPO_ROOT}/artifacts/models/GLM-5.1-FP8"
FP8_DST="/tmp/GLM-5.1-FP8"
BF16_TMP="/tmp/GLM-5.1"
BF16_LUSTRE="${REPO_ROOT}/artifacts/models/GLM-5.1"
LOG="${LOG:-/tmp/glm51-migrate-storage.log}"

log() { echo "$(date -Is) $*" | tee -a "$LOG"; }

_bytes() { du -sb "$1" 2>/dev/null | awk '{print $1}'; }

_verify_rsync() {
  local src=$1 dst=$2
  local sb db
  sb=$(_bytes "$src")
  db=$(_bytes "$dst")
  if [[ "$db" -lt "$(( sb * 99 / 100 ))" ]]; then
    log "ERROR: $dst too small ($db vs $sb bytes)"
    return 1
  fi
  log "OK: $dst matches $src ($db bytes)"
}

log "START migrate (log=$LOG)"
df -h /home/coder /tmp | tee -a "$LOG"

if [[ -L "$FP8_SRC" ]]; then
  log "FP8 already symlinked: $FP8_SRC -> $(readlink -f "$FP8_SRC")"
elif [[ -d "$FP8_SRC" ]]; then
  log "copy FP8: $FP8_SRC -> $FP8_DST"
  mkdir -p "$FP8_DST"
  cp -a "$FP8_SRC/." "$FP8_DST/"
  _verify_rsync "$FP8_SRC" "$FP8_DST"
  rm -rf "$FP8_SRC"
  ln -sfn "$FP8_DST" "$FP8_SRC"
  log "FP8 on Lustre replaced with symlink -> $FP8_DST"
else
  log "SKIP FP8 (no directory at $FP8_SRC)"
fi

df -h /home/coder | tee -a "$LOG"
AVAIL=$(df -B1 /home/coder | awk 'NR==2 {print $4}')
BF16_BYTES=$(_bytes "$BF16_TMP" 2>/dev/null || echo 0)

if [[ ! -d "$BF16_TMP" ]]; then
  log "ERROR: BF16 not found at $BF16_TMP — run download first"
  exit 1
fi

if [[ -e "$BF16_LUSTRE" ]] && [[ ! -L "$BF16_LUSTRE" ]]; then
  log "Removing incomplete Lustre BF16 tree: $BF16_LUSTRE"
  rm -rf "$BF16_LUSTRE"
fi

if [[ "$AVAIL" -gt "$(( BF16_BYTES * 105 / 100 ))" ]]; then
  log "Moving BF16 to Lustre ($BF16_BYTES bytes, avail $AVAIL)"
  mkdir -p "$(dirname "$BF16_LUSTRE")"
  mkdir -p "$BF16_LUSTRE"
  cp -a "$BF16_TMP/." "$BF16_LUSTRE/"
  _verify_rsync "$BF16_TMP" "$BF16_LUSTRE"
  log "BF16 copied to $BF16_LUSTRE (keeping $BF16_TMP)"
else
  log "Lustre avail $AVAIL < BF16 $BF16_BYTES — symlink $BF16_LUSTRE -> $BF16_TMP"
  ln -sfn "$BF16_TMP" "$BF16_LUSTRE"
fi

df -h /home/coder /tmp | tee -a "$LOG"
log "DONE"
