#!/usr/bin/env bash
# BF16 full cycle: 4096-sample observer+prune+save on evol-codealpaca-v1, then HF eval.
# Prerequisite: 4-batch gate reviewed; archive 4-batch checkpoint before launch.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export GLM51_VARIANT=bf16
export GLM51_LOAD_MODE="${GLM51_LOAD_MODE:-fast_ram}"
# shellcheck source=scripts/glm51_env.sh
source "${REPO_ROOT}/scripts/glm51_env.sh"
export HF_HOME="${HF_HOME:-/tmp/hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTHONUNBUFFERED=1

SEED="${SEED:-42}"
COMPRESSION="${COMPRESSION:-0.25}"
DATASET_SLUG="${DATASET_SLUG:-evol-codealpaca-v1}"
LOG_DIR="${LOG_DIR:-/tmp}"
PIPELINE_LOG="${PIPELINE_LOG:-${LOG_DIR}/glm51-bf16-4096-cycle.log}"
PRUNE_LOG="${PRUNE_LOG:-${LOG_DIR}/glm51-bf16-4096-prune.log}"
EVAL_LOG="${EVAL_LOG:-${LOG_DIR}/glm51-bf16-4096-eval.log}"
PROGRESS_LOG="${PROGRESS_LOG:-${LOG_DIR}/glm51-bf16-4096-progress.log}"

PRUNED_DIR="${PRUNED_MODEL:-${REPO_ROOT}/artifacts/GLM-5.1/${DATASET_SLUG}/pruned_models/reap-renorm_true-seed_${SEED}-${COMPRESSION}}"
PRUNED_NAME="$(basename "$PRUNED_DIR")"
PRUNED_BACKING_DIR="${PRUNED_BACKING_DIR:-/tmp/reap-glm51-pruned/${DATASET_SLUG}/${PRUNED_NAME}}"
OBS4096="${REPO_ROOT}/artifacts/GLM-5.1/${DATASET_SLUG}/all/observations_4096_cosine-seed_${SEED}.pt"

log() {
  echo "$(date -Is) $*" | tee -a "$PIPELINE_LOG"
}

has_pruned_safetensors() {
  compgen -G "${PRUNED_DIR}/*.safetensors" >/dev/null
}

archive_pruned_dir() {
  local reason="$1"
  local stamp
  stamp="$(date +%Y%m%dT%H%M%S)"
  if [[ -L "$PRUNED_DIR" ]]; then
    local target archive
    target="$(readlink -f "$PRUNED_DIR")"
    archive="${target}-${reason}-${stamp}"
    mv "$target" "$archive"
    rm "$PRUNED_DIR"
    log "Archived pruned backing ${archive}"
  elif [[ -e "$PRUNED_DIR" ]]; then
    local archive="${PRUNED_DIR}-${reason}-${stamp}"
    mv "$PRUNED_DIR" "$archive"
    log "Archived pruned dir ${archive}"
  fi
}

ensure_pruned_backing() {
  if [[ -L "$PRUNED_DIR" ]]; then
    return
  fi
  if [[ -e "$PRUNED_DIR" ]]; then
    return
  fi
  mkdir -p "$(dirname "$PRUNED_DIR")" "$PRUNED_BACKING_DIR"
  ln -s "$PRUNED_BACKING_DIR" "$PRUNED_DIR"
  log "Linked ${PRUNED_DIR} -> ${PRUNED_BACKING_DIR}"
}

prepare_4096_output_dir() {
  if has_pruned_safetensors; then
    if [[ "${ARCHIVE_EXISTING_PRUNED:-true}" == "true" ]]; then
      archive_pruned_dir "pre-4096"
    else
      log "Refusing to overwrite ${PRUNED_DIR}; set ARCHIVE_EXISTING_PRUNED=true"
      exit 3
    fi
  fi
  ensure_pruned_backing
}

progress_snapshot() {
  {
    echo "=== $(date -Is) ==="
    pgrep -af 'src/reap/prune.py' || echo "no prune.py"
    if [[ -f "${REPO_ROOT}/artifacts/glm_5_1_monolithic_4096.log" ]]; then
      grep -oE '[0-9]+%\|' "${REPO_ROOT}/artifacts/glm_5_1_monolithic_4096.log" 2>/dev/null | tail -1 \
        | xargs -I{} echo "load {}" || true
      grep 'Processing all samples' "${REPO_ROOT}/artifacts/glm_5_1_monolithic_4096.log" 2>/dev/null \
        | tr '\r' '\n' | tail -1 || true
    fi
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | head -2 || true
  } >>"$PROGRESS_LOG"
}

run_4096_prune() {
  log "START 4096-batch BF16 observer+prune+save"
  prepare_4096_output_dir
  (
    while true; do
      progress_snapshot
      sleep "${MONITOR_INTERVAL_SEC:-3600}"
      pgrep -f 'src/reap/prune.py.*observations_4096' >/dev/null || break
    done
  ) &
  MON_PID=$!
  SMOKE=false BATCHES=4096 RUN_OBSERVER_ONLY=false OVERWRITE_OBSERVATIONS=true \
    LOG="${REPO_ROOT}/artifacts/glm_5_1_monolithic_4096.log" \
    bash scripts/glm_5_1_monolithic.sh 2>&1 | tee "$PRUNE_LOG"
  kill "$MON_PID" 2>/dev/null || true
  if [[ ! -f "$OBS4096" ]]; then
    log "Missing ${OBS4096}"
    exit 4
  fi
  if ! has_pruned_safetensors; then
    log "Missing pruned safetensors under ${PRUNED_DIR}"
    exit 5
  fi
  log "DONE 4096-batch prune"
}

run_4096_eval() {
  log "START 4096-batch HF eval (lm-eval; EvalPlus skipped for GLM MoE)"
  PRUNED_MODEL="$PRUNED_DIR" LOG="$EVAL_LOG" bash scripts/run_eval_pruned_only.sh
  log "DONE 4096-batch eval"
}

: >"$PIPELINE_LOG"
: >"$PROGRESS_LOG"
log "BF16 4096 cycle: model=${GLM51_MODEL_PATH} load=${GLM51_LOAD_MODE}"

if [[ "${RUN_EVAL_ONLY:-false}" == "true" ]]; then
  run_4096_eval
  exit 0
fi

if [[ "${RUN_PRUNE_ONLY:-false}" == "true" ]]; then
  run_4096_prune
  exit 0
fi

run_4096_prune
run_4096_eval
log "BF16 4096 cycle complete"
