#!/usr/bin/env bash
# Periodic monitor for GLM-5.1 monolithic / 10h plan runs.
# Usage: MONITOR_INTERVAL_SEC=300 bash scripts/monitor_glm51_run.sh
set -euo pipefail

INTERVAL="${MONITOR_INTERVAL_SEC:-300}"
LOG="${MONITOR_LOG:-/tmp/glm51-monitor.log}"
PLAN_LOG="${PLAN_LOG:-/tmp/glm51-10h-plan.log}"
STUCK_LOAD_SEC="${STUCK_LOAD_SEC:-7200}"

_prune_pid() {
  pgrep -f 'src/reap/prune.py' 2>/dev/null | head -1 || true
}

_snapshot() {
  local ts phase vram_mib util_max
  ts="$(date -Is)"
  {
    echo "======== $ts ========"
    local pid
    pid="$(_prune_pid)"
    if [[ -n "$pid" ]]; then
      echo "prune.py: RUNNING (pid=$pid)"
      ps -p "$pid" -o etime,pcpu,pmem,rss,cmd --no-headers 2>/dev/null || true
    else
      echo "prune.py: NOT RUNNING"
    fi

    if command -v nvidia-smi >/dev/null 2>&1; then
      echo "--- GPUs (index, util%, mem_used_MiB, power_W) ---"
      nvidia-smi --query-gpu=index,utilization.gpu,memory.used,power.draw \
        --format=csv,noheader 2>/dev/null || nvidia-smi
      vram_mib="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
        | awk -F' ' '{s+=$1} END {printf "%.0f", s}')"
      util_max="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null \
        | sort -n | tail -1 | tr -d ' ')"
      echo "VRAM sum: ${vram_mib:-?} MiB | max GPU util: ${util_max:-?}%"
      if [[ -n "$pid" ]] && [[ "${vram_mib:-0}" -gt 100000 ]]; then
        echo "OK: model shards appear loaded on GPUs (~${vram_mib} MiB)"
      elif [[ -n "$pid" ]]; then
        echo "NOTE: low VRAM while prune running — may still be loading to CPU first"
      fi
    fi

    echo "--- Plan log phase (last matches) ---"
    if [[ -f "$PLAN_LOG" ]]; then
      phase="$(grep -E 'Loading weights:.*%|Processing all samples:|Start of pruning|Observer run completed|Pruned model saved|Traceback|CUDA out of memory' "$PLAN_LOG" 2>/dev/null | tail -3)"
      if [[ -n "$phase" ]]; then
        echo "$phase"
      else
        echo "(no phase markers yet; file $(wc -c <"$PLAN_LOG") bytes)"
      fi
    else
      echo "(no $PLAN_LOG)"
    fi
    echo ""
  } >>"$LOG"
  echo "$ts snapshot -> $LOG"
}

{
  echo "GLM-5.1 monitor started (interval=${INTERVAL}s, log=$LOG)"
} >>"$LOG"
_last_load_line=""
_last_load_change="$(date +%s)"

while true; do
  _snapshot
  if [[ -f "$PLAN_LOG" ]]; then
    load_line="$(grep -oE 'Loading weights:.*[0-9]+%' "$PLAN_LOG" 2>/dev/null | tail -1 || true)"
    if [[ -n "$load_line" ]] && [[ "$load_line" != "$_last_load_line" ]]; then
      _last_load_line="$load_line"
      _last_load_change="$(date +%s)"
      echo "$(date -Is) load progress: $load_line" >>"$LOG"
    fi
    if grep -q 'Processing all samples:' "$PLAN_LOG" 2>/dev/null; then
      echo "$(date -Is) PHASE: observer calibration (expect higher GPU util)" >>"$LOG"
    fi
    now="$(date +%s)"
    if [[ -n "$(_prune_pid)" ]] && ! grep -q 'Processing all samples:' "$PLAN_LOG" 2>/dev/null; then
      if (( now - _last_load_change > STUCK_LOAD_SEC )); then
        echo "$(date -Is) WARN: no load progress for ${STUCK_LOAD_SEC}s — consider checking Lustre/disk or restart" >>"$LOG"
      fi
    fi
  fi
  sleep "$INTERVAL"
done
