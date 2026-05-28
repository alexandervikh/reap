#!/usr/bin/env bash
# 10h babysitter: periodic checks, decision log, optional post-monolithic eval.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

INTERVAL="${BABYSIT_INTERVAL_SEC:-300}"
PLAN_LOG="${PLAN_LOG:-/tmp/glm51-10h-plan.log}"
MONITOR_LOG="${MONITOR_LOG:-/tmp/glm51-monitor.log}"
DECISION_LOG="${DECISION_LOG:-/tmp/glm51-babysit-decisions.log}"
STATE="${BABYSIT_STATE:-/tmp/glm51-babysit-state}"
STUCK_LOAD_SEC="${STUCK_LOAD_SEC:-7200}"
OBS_64="${REPO_ROOT}/artifacts/GLM-5.1-FP8/evol-codealpaca-v1/all/observations_64_cosine-seed_42.pt"
PRUNED_DIR="${REPO_ROOT}/artifacts/GLM-5.1-FP8/evol-codealpaca-v1/pruned_models/reap-renorm_true-seed_42-0.25"

_prune_pid() {
  pgrep -f 'python src/reap/prune.py' 2>/dev/null | head -1 || true
}
_plan_pid() { pgrep -f 'run_10h_glm51_plan.sh' 2>/dev/null | head -1 || true; }

_decide() {
  echo "$(date -Is) $*" | tee -a "$DECISION_LOG"
}

_load_pct() {
  if [[ ! -f "$PLAN_LOG" ]]; then
    echo "?"
    return
  fi
  python3 - "$PLAN_LOG" <<'PY'
import re, sys
t = open(sys.argv[1], errors="replace").read()
m = re.findall(r"(\d+)%\|", t)
print(f"{m[-1]}%" if m else "?")
PY
}

touch "$STATE"
_last_load=""
_last_load_ts="$(date +%s)"

_decide "BABYSIT start interval=${INTERVAL}s plan_log=$PLAN_LOG"

while true; do
  pid="$(_prune_pid)"
  plan_pid="$(_plan_pid)"
  load_pct="$(_load_pct)"

  if [[ -n "$pid" ]]; then
    if grep -q 'Processing all samples:' "$PLAN_LOG" 2>/dev/null; then
      _decide "RUNNING prune pid=$pid — CALIBRATION (64-batch observer); GPU util should rise"
    elif [[ "$load_pct" != "?" ]]; then
      _decide "RUNNING prune pid=$pid — LOADING weights ($load_pct); 0% GPU util during shard load is normal"
      load_line="$(grep -oE 'Loading weights:.*\| [0-9]+%.*\| [0-9]+/2559' "$PLAN_LOG" 2>/dev/null | tail -1 || true)"
      if [[ -n "$load_line" ]] && [[ "$load_line" != "$_last_load" ]]; then
        _last_load="$load_line"
        _last_load_ts="$(date +%s)"
      fi
      now="$(date +%s)"
      if (( now - _last_load_ts > STUCK_LOAD_SEC )); then
        _decide "WARN load stalled >${STUCK_LOAD_SEC}s at $load_pct — check disk/Lustre; do not kill unless no progress 3+ h"
      fi
    else
      _decide "RUNNING prune pid=$pid — phase unknown (no load line yet)"
    fi
    if grep -q 'CUDA out of memory' "$PLAN_LOG" 2>/dev/null; then
      if ! grep -q 'DECISION: OOM logged' "$STATE" 2>/dev/null; then
        _decide "ACTION CUDA OOM during prune — inspect $PLAN_LOG"
        echo "DECISION: OOM logged" >>"$STATE"
      fi
    fi
    if awk '/=== 2\/3 Monolithic/,0' "$PLAN_LOG" 2>/dev/null | grep -q 'Traceback'; then
      if ! grep -q 'DECISION: monolithic traceback' "$STATE" 2>/dev/null; then
        _decide "ACTION traceback in monolithic section — inspect $PLAN_LOG"
        echo "DECISION: monolithic traceback" >>"$STATE"
      fi
    fi
  else
    if grep -q 'Pruned model saved to' "$PLAN_LOG" 2>/dev/null; then
      _decide "DONE monolithic — Pruned model saved (see plan log)"
      if [[ -f "$OBS_64" ]]; then
        _decide "VERIFY observations_64 present: $OBS_64"
      else
        _decide "WARN observations_64 missing — check artifacts/GLM-5.1-FP8/evol-codealpaca-v1/all/"
      fi
      if ! grep -q 'DECISION: eval triggered' "$STATE" 2>/dev/null; then
        if grep -q 'WARN: eval failed' "$PLAN_LOG" 2>/dev/null \
          && ! grep -q 'Finished evaluating lm-eval' "$PLAN_LOG" 2>/dev/null; then
          if [[ -z "$(pgrep -f 'src/reap/eval.py' || true)" ]]; then
            _decide "ACTION starting HF eval on pruned checkpoint (prior eval failed)"
            echo "DECISION: eval triggered" >>"$STATE"
            nohup bash scripts/run_eval_pruned_only.sh >>/tmp/glm51-eval-pruned.nohup 2>&1 &
          fi
        fi
      fi
    elif grep -q 'Observer run completed' "$PLAN_LOG" 2>/dev/null; then
      _decide "DONE observer-only — no prune save expected"
    elif [[ -n "$plan_pid" ]]; then
      _decide "prune exited; 10h plan still running (pid=$plan_pid) — waiting for collate/finish"
    else
      _decide "IDLE no prune.py and no 10h plan — check if run crashed"
    fi
  fi

  sleep "$INTERVAL"
done
