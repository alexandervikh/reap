#!/usr/bin/env bash
# BF16 gate: run a 4-batch prune + HF eval before optionally launching 64 batches.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export GLM51_VARIANT=bf16
export GLM51_LOAD_MODE="${GLM51_LOAD_MODE:-fast_ram}"
if [[ "$GLM51_LOAD_MODE" != "fast_ram" ]]; then
  echo "This BF16 gate must run with GLM51_LOAD_MODE=fast_ram, got ${GLM51_LOAD_MODE}." >&2
  exit 2
fi

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
PIPELINE_LOG="${PIPELINE_LOG:-${LOG_DIR}/glm51-bf16-gate-4-then-64.log}"
PRUNE4_LOG="${PRUNE4_LOG:-${LOG_DIR}/glm51-bf16-4batch-prune.log}"
EVAL4_LOG="${EVAL4_LOG:-${LOG_DIR}/glm51-bf16-4batch-eval.log}"
FULL64_LOG="${FULL64_LOG:-${LOG_DIR}/glm51-bf16-64batch-prune.log}"
OVERWRITE_OBSERVATIONS="${OVERWRITE_OBSERVATIONS:-true}"

PRUNED_DIR="${PRUNED_MODEL:-${REPO_ROOT}/artifacts/GLM-5.1/${DATASET_SLUG}/pruned_models/reap-renorm_true-seed_${SEED}-${COMPRESSION}}"
PRUNED_NAME="$(basename "$PRUNED_DIR")"
PRUNED_BACKING_DIR="${PRUNED_BACKING_DIR:-/tmp/reap-glm51-pruned/${DATASET_SLUG}/${PRUNED_NAME}}"
OBS4="${REPO_ROOT}/artifacts/GLM-5.1/${DATASET_SLUG}/all/observations_4_cosine-seed_${SEED}.pt"

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
    log "Archived existing pruned checkpoint backing to ${archive}"
  else
    local archive="${PRUNED_DIR}-${reason}-${stamp}"
    mv "$PRUNED_DIR" "$archive"
    log "Archived existing pruned checkpoint to ${archive}"
  fi
}

ensure_pruned_backing() {
  if [[ -z "${PRUNED_BACKING_DIR}" ]]; then
    return
  fi
  if [[ -L "$PRUNED_DIR" ]]; then
    return
  fi
  if [[ -e "$PRUNED_DIR" ]]; then
    log "Using existing pruned checkpoint directory on artifacts filesystem: ${PRUNED_DIR}"
    return
  fi
  mkdir -p "$(dirname "$PRUNED_DIR")" "$PRUNED_BACKING_DIR"
  ln -s "$PRUNED_BACKING_DIR" "$PRUNED_DIR"
  log "Linked ${PRUNED_DIR} -> ${PRUNED_BACKING_DIR} for checkpoint storage"
}

prepare_4batch_output_dir() {
  if has_pruned_safetensors; then
    if [[ "${ARCHIVE_EXISTING_PRUNED_FOR_4B:-false}" == "true" ]]; then
      archive_pruned_dir "pre-4batch"
    else
      log "Refusing to overwrite existing pruned checkpoint at ${PRUNED_DIR}"
      log "Set ARCHIVE_EXISTING_PRUNED_FOR_4B=true only after confirming it is safe to move."
      exit 3
    fi
  fi
  ensure_pruned_backing
}

run_4batch_prune() {
  log "START 4-batch BF16 prune: output=${OBS4}, pruned=${PRUNED_DIR}"
  prepare_4batch_output_dir
  SMOKE=false BATCHES=4 RUN_OBSERVER_ONLY=false OVERWRITE_OBSERVATIONS="$OVERWRITE_OBSERVATIONS" \
    LOG="${REPO_ROOT}/artifacts/glm_5_1_monolithic_4.log" \
    bash scripts/glm_5_1_monolithic.sh 2>&1 | tee "$PRUNE4_LOG"

  if [[ ! -f "$OBS4" ]]; then
    log "4-batch prune did not create ${OBS4}"
    exit 4
  fi
  if ! has_pruned_safetensors; then
    log "4-batch prune did not create safetensors under ${PRUNED_DIR}"
    exit 5
  fi
  log "DONE 4-batch BF16 prune"
}

run_4batch_eval() {
  log "START 4-batch HF eval: pruned=${PRUNED_DIR}"
  PRUNED_MODEL="$PRUNED_DIR" LOG="$EVAL4_LOG" bash scripts/run_eval_pruned_only.sh
  log "DONE 4-batch HF eval"
}

prepare_64batch_output_dir() {
  if has_pruned_safetensors; then
    if [[ "${ARCHIVE_4B_BEFORE_64:-true}" == "true" ]]; then
      archive_pruned_dir "4batch-eval-ok"
    else
      log "Refusing to replace 4-batch checkpoint before 64-batch run: ${PRUNED_DIR}"
      log "Leave ARCHIVE_4B_BEFORE_64=true to archive it, or move it manually first."
      exit 6
    fi
  fi
  ensure_pruned_backing
}

run_64batch_prune() {
  log "START 64-batch BF16 prune: pruned=${PRUNED_DIR}"
  prepare_64batch_output_dir
  SMOKE=false BATCHES=64 RUN_OBSERVER_ONLY=false OVERWRITE_OBSERVATIONS=true \
    LOG="${REPO_ROOT}/artifacts/glm_5_1_monolithic_64.log" \
    bash scripts/glm_5_1_monolithic.sh 2>&1 | tee "$FULL64_LOG"
  log "DONE 64-batch BF16 prune"
}

: >"$PIPELINE_LOG"
log "BF16 gate using GLM51_LOAD_MODE=${GLM51_LOAD_MODE}, GLM51_MODEL_PATH=${GLM51_MODEL_PATH}"

if [[ "${RUN_64_ONLY_AFTER_REVIEW:-false}" == "true" ]]; then
  log "RUN_64_ONLY_AFTER_REVIEW=true; launching reviewed 64-batch prune"
  run_64batch_prune
  exit 0
fi

run_4batch_prune
run_4batch_eval

if [[ "${RUN_64_AFTER_EVAL:-false}" == "true" ]]; then
  log "RUN_64_AFTER_EVAL=true; launching 64-batch prune after successful eval exit"
  run_64batch_prune
else
  log "64-batch prune not launched. Review eval results, then run with RUN_64_AFTER_EVAL=true if acceptable."
fi
