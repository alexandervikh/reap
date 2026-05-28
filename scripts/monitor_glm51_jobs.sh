#!/usr/bin/env bash
# Unified monitor for GLM-5.1 jobs (BF16 smoke/full, reprune, downloads, migrations).
# Usage: MONITOR_INTERVAL_SEC=120 bash scripts/monitor_glm51_jobs.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INTERVAL="${MONITOR_INTERVAL_SEC:-120}"
DETAIL_LOG="${DETAIL_LOG:-/tmp/glm51-progress-detail.log}"
STATUS_FILE="${STATUS_FILE:-/tmp/glm51-progress.txt}"
AGENT_LOG="${AGENT_LOG:-/tmp/glm51-agent-updates.log}"

JOB_LOGS=(
  "/tmp/glm51-bf16-smoke.log"
  "${REPO_ROOT}/artifacts/glm_5_1_monolithic_smoke.log"
  "${REPO_ROOT}/artifacts/glm_5_1_monolithic.log"
  "/tmp/glm51-10h-plan.log"
  "/tmp/glm51-reprune-timed.log"
  "/tmp/glm51-migrate-storage.log"
  "/tmp/glm51-bf16-download.log"
)

_prune_pid() { pgrep -f 'src/reap/prune.py' 2>/dev/null | head -1 || true; }
_monolithic_pid() { pgrep -f 'glm_5_1_monolithic.sh' 2>/dev/null | head -1 || true; }

_parse_job_log() {
  local f=$1
  [[ -f "$f" ]] || return 0
  python3 - "$f" <<'PY'
import re, sys
from pathlib import Path
p = Path(sys.argv[1])
t = p.read_text(errors="replace")
out = []
loads = re.findall(r"(\d+)%\|", t)
if loads:
    out.append(f"load={loads[-1]}%")
if "GLM51_LOAD_MODE=fast_ram" in t or "load=fast_ram" in t:
    out.append("load_mode=fast_ram")
if "previously processed. Skipping" in t:
    out.append("observer=skipped")
elif "Processing all samples" in t:
    m = re.findall(r"Processing all samples:.*?\| (\d+)/(\d+)", t)
    if m:
        a, b = m[-1]
        out.append(f"observer={a}/{b}")
elif "Observer run completed" in t:
    out.append("observer=done")
if "Start of pruning" in t:
    out.append("prune=started")
m = re.findall(r"Pruning layers:.*?\| (\d+)/(\d+)", t)
if m:
    a, b = m[-1]
    out.append(f"prune_layers={a}/{b}")
if "Pruned model saved to" in t:
    out.append("save=done")
elif "Saving pruned model" in t:
    out.append("save=in_progress")
if "Traceback" in t or "Error" in t.splitlines()[-20:]:
    if "Traceback" in t:
        out.append("ERROR=traceback")
if "CUDA out of memory" in t:
    out.append("ERROR=oom")
print(" ".join(out) if out else "no_markers")
PY
}

_snapshot() {
  local ts pid mpid vram util_max phase_lines status_line
  ts="$(date -Is)"
  pid="$(_prune_pid)"
  mpid="$(_monolithic_pid)"

  {
    echo "======== $ts ========"
    if [[ -n "$pid" ]]; then
      echo "JOB: prune.py RUNNING pid=$pid"
      ps -p "$pid" -o etime,pcpu,pmem,rss,cmd --no-headers 2>/dev/null || true
    else
      echo "JOB: prune.py not running"
    fi
    [[ -n "$mpid" ]] && echo "JOB: glm_5_1_monolithic.sh pid=$mpid"

    if command -v nvidia-smi >/dev/null 2>&1; then
      echo "--- GPUs ---"
      nvidia-smi --query-gpu=index,utilization.gpu,memory.used,power.draw \
        --format=csv,noheader 2>/dev/null || true
      vram_mib="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
        | awk '{s+=$1} END {printf "%.0f", s}')"
      util_max="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null \
        | sort -n | tail -1 | tr -d ' ')"
      echo "VRAM sum: ${vram_mib:-?} MiB | max util: ${util_max:-?}%"
    fi

    echo "--- Log phases ---"
    for f in "${JOB_LOGS[@]}"; do
      [[ -f "$f" ]] || continue
      phase_lines="$(_parse_job_log "$f")"
      echo "  $(basename "$f"): $phase_lines ($(wc -c <"$f") bytes)"
    done
    echo ""
  } >>"$DETAIL_LOG"

  # One-line status for agents / quick tail
  status_line="$ts"
  [[ -n "$pid" ]] && status_line+=" prune:RUNNING"
  [[ -n "$mpid" ]] && status_line+=" monolithic:RUNNING"
  for f in "/tmp/glm51-bf16-smoke.log" "${REPO_ROOT}/artifacts/glm_5_1_monolithic_smoke.log" \
    "${REPO_ROOT}/artifacts/glm_5_1_monolithic.log"; do
    [[ -f "$f" ]] || continue
    phase_lines="$(_parse_job_log "$f")"
    [[ "$phase_lines" != "no_markers" ]] && status_line+=" | $(basename "$f"): $phase_lines"
  done
  echo "$status_line" >"$STATUS_FILE"
  echo "$status_line" >>"$AGENT_LOG"
}

echo "$(date -Is) monitor_glm51_jobs started interval=${INTERVAL}s" >>"$DETAIL_LOG"
echo "$(date -Is) monitor started -> $DETAIL_LOG $STATUS_FILE" >>"$AGENT_LOG"

while true; do
  _snapshot
  sleep "$INTERVAL"
done
