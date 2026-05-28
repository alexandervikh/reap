#!/usr/bin/env bash
# Re-prune GLM-5.1 from existing observations (skip observer), with per-phase timing.
# Default: BF16 + GLM51_LOAD_MODE=local (staged under /tmp/GLM-5.1).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export PATH="${HOME}/.local/bin:${PATH}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

# shellcheck source=scripts/glm51_env.sh
source "${REPO_ROOT}/scripts/glm51_env.sh"

PYTHON="${REPO_ROOT}/.venv/bin/python"
MODEL="${MODEL:-$GLM51_MODEL_PATH}"
if [[ "$GLM51_LOAD_MODE" != "lustre" ]] && [[ -d "$GLM51_LOCAL_DIR" ]]; then
  MODEL="$GLM51_LOCAL_DIR"
elif [[ -d "$GLM51_ARTIFACTS_DIR" ]]; then
  MODEL="$GLM51_ARTIFACTS_DIR"
fi

DATASET="${DATASET:-theblackcat102/evol-codealpaca-v1}"
SEED="${SEED:-42}"
BATCHES="${BATCHES:-64}"
LOG="${LOG:-/tmp/glm51-reprune-timed.log}"
TIMING="${TIMING:-/tmp/glm51-reprune-timing.log}"
PROGRESS="${PROGRESS:-/tmp/glm51-reprune-progress.txt}"

: >"$LOG"
: >"$TIMING"
: >"$PROGRESS"

START_SEC=$(date +%s)
_mark() {
  local now elapsed
  now=$(date +%s)
  elapsed=$((now - START_SEC))
  printf '%s +%ds %s\n' "$(date -Is)" "$elapsed" "$*" | tee -a "$TIMING"
}

_update_progress() {
  python3 - "$LOG" "$PROGRESS" <<'PY'
import re, sys
from pathlib import Path
log = Path(sys.argv[1]).read_text(errors="replace") if Path(sys.argv[1]).exists() else ""
out = []

loads = re.findall(r"(\d+)%\|", log)
if loads:
    out.append(f"model_load={loads[-1]}%")

if "previously processed. Skipping" in log:
    out.append("observer=skipped(100%)")
elif "Processing all samples" in log:
    m = re.findall(r"Processing all samples:.*?(\d+)/(\d+)", log)
    if m:
        a, b = m[-1]
        out.append(f"observer={100*int(a)//int(b)}% ({a}/{b})")

if "Pruned model saved to" in log:
    out.append("save=100%")
elif "Saving pruned model" in log:
    out.append("save=in_progress")

m = re.findall(r"Pruning layers:.*?\| (\d+)/(\d+)", log)
if m:
    a, b = m[-1]
    out.append(f"prune_layers={100*int(a)//int(b)}% ({a}/{b})")
elif "Start of pruning" in log:
    out.append("prune_layers=0% (started)")
elif "observer=skipped" in " ".join(out) or "observer=100%" in " ".join(out):
    if "prune_layers" not in " ".join(out) and "Start of pruning" in log:
        out.append("prune_layers=0%")

if not out:
    out.append("phase=init_or_loading")

pct = 0
for line in out:
    if line.startswith("model_load="):
        pct += 70 * int(line.split("=")[1].rstrip("%")) // 100
    elif "observer=" in line and "skipped" in line:
        pct += 5
    elif line.startswith("observer="):
        pct += 5 * int(re.search(r"(\d+)%", line).group(1)) // 100
    elif line.startswith("prune_layers="):
        pct += 15 * int(re.search(r"(\d+)%", line).group(1)) // 100
    elif line.startswith("save=100"):
        pct += 10

Path(sys.argv[2]).write_text(
    f"overall≈{pct}%\n" + "\n".join(out) + "\n",
)
PY
}

_watch_log() {
  local pid=$1
  _last_phase=""
  while kill -0 "$pid" 2>/dev/null; do
    _update_progress
    if [[ -f "$LOG" ]]; then
      while IFS= read -r line; do
        case "$line" in
          *"Loading weights:"*)
            [[ "$_last_phase" == load ]] || { _last_phase=load; _mark "PHASE model_load"; }
            ;;
          *"previously processed. Skipping"*)
            [[ "$_last_phase" == obs ]] || { _last_phase=obs; _mark "PHASE observer_skipped"; }
            ;;
          *"Start of pruning"*)
            [[ "$_last_phase" == prune ]] || { _last_phase=prune; _mark "PHASE prune_cluster_start"; }
            ;;
          *"Saving pruned model"*)
            [[ "$_last_phase" == save ]] || { _last_phase=save; _mark "PHASE save_checkpoint_start"; }
            ;;
          *"Pruned model saved to"*)
            _mark "PHASE save_checkpoint_done | $line"
            ;;
        esac
      done < <(tail -c +"$(( _watch_offset + 1 ))" "$LOG" 2>/dev/null || true)
      _watch_offset=$(wc -c <"$LOG" 2>/dev/null || echo 0)
    fi
    sleep 30
  done
}

_mark "START re-prune variant=$GLM51_VARIANT load=$GLM51_LOAD_MODE model=$MODEL (observations_${BATCHES})"
_watch_offset=0

"$PYTHON" src/reap/prune.py \
  --model-name "$MODEL" \
  --dataset-name "$DATASET" \
  --compression-ratio 0.25 \
  --prune-method reap \
  --profile false \
  --do-eval false \
  --distance_measure cosine \
  --seed "$SEED" \
  --output_file_name "observations_${BATCHES}_cosine-seed_${SEED}.pt" \
  --batch_size 1 \
  --model_max_length 2048 \
  --batches_per_category "$BATCHES" \
  --record_pruning_metrics_only true \
  --run_observer_only false \
  --overwrite_observations false \
  --overwrite_pruned_model true \
  2>&1 | stdbuf -oL tr '\r' '\n' >>"$LOG" &
PRUNE_PID=$!

_watch_log "$PRUNE_PID" &
WATCH_PID=$!

wait "$PRUNE_PID" || EXIT=$?
EXIT=${EXIT:-0}
kill "$WATCH_PID" 2>/dev/null || true
_update_progress
_mark "END re-prune exit_code=$EXIT"
exit "$EXIT"
