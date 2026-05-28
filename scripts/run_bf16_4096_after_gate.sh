#!/usr/bin/env bash
# Wait for in-flight 4-batch eval, then run the full 4096 cycle (archive + prune + eval).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="${LOG:-/tmp/glm51-bf16-4096-after-gate.log}"

log() { echo "$(date -Is) $*" | tee -a "$LOG"; }

: >"$LOG"
log "Waiting for 4-batch eval to finish..."
while pgrep -f 'src/reap/eval.py.*reap-renorm_true-seed_42-0.25' >/dev/null 2>&1; do
  sleep "${WAIT_INTERVAL_SEC:-300}"
done
log "4-batch eval finished; starting 4096 cycle"
exec bash "${REPO_ROOT}/scripts/run_bf16_4096_cycle.sh"
