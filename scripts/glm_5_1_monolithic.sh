#!/usr/bin/env bash
# Paper-style monolithic REAP for GLM-5.1 (BF16 default): prune.py + MoE hooks.
# Use SMOKE=true first (2 batches, observer-only). Tune load via GLM51_LOAD_MODE.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

unset PYTHONPATH
PYTHON="${REPO_ROOT}/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON=python
fi

# shellcheck source=scripts/glm51_env.sh
source "${REPO_ROOT}/scripts/glm51_env.sh"

MODEL="${MODEL:-$GLM51_MODEL_PATH}"
if [[ ! -d "$MODEL" ]] && [[ "$MODEL" == zai-org/* ]]; then
  echo "Downloading ${GLM51_HF_ID} to ${GLM51_ARTIFACTS_DIR} ..."
  "$PYTHON" scripts/download_glm_5_1.py --variant "$GLM51_VARIANT"
  MODEL="$GLM51_ARTIFACTS_DIR"
fi
if [[ "$GLM51_LOAD_MODE" == "local" || "$GLM51_LOAD_MODE" == "fast_ram" || "$GLM51_LOAD_MODE" == "cpu_dispatch" ]]; then
  if [[ ! -d "$GLM51_LOCAL_DIR" ]] && [[ -d "$GLM51_ARTIFACTS_DIR" ]]; then
    echo "GLM51_LOAD_MODE=$GLM51_LOAD_MODE: staging to $GLM51_LOCAL_DIR ..."
    GLM51_VARIANT="$GLM51_VARIANT" bash scripts/stage_glm51_local.sh
    MODEL="$GLM51_LOCAL_DIR"
  elif [[ -d "$GLM51_LOCAL_DIR" ]]; then
    MODEL="$GLM51_LOCAL_DIR"
  fi
fi

SEED="${SEED:-42}"
COMPRESSION="${COMPRESSION:-0.25}"
DATASET="${DATASET:-${COMPOSITE_DATASET:-theblackcat102/evol-codealpaca-v1}}"
SMOKE="${SMOKE:-true}"

if [[ "$SMOKE" == "true" ]]; then
  if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]] || [[ "$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)" -lt 2 ]]; then
    echo "SMOKE: defaulting CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 (GLM-5.1 needs multi-GPU load)"
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
  fi
  BATCHES=2
  RUN_OBSERVER_ONLY="true"
  MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-512}"
  LOG="${LOG:-artifacts/glm_5_1_monolithic_smoke.log}"
  OUT_NAME="observations_smoke_cosine-seed_${SEED}.pt"
  echo "Monolithic smoke (${GLM51_VARIANT}, load=${GLM51_LOAD_MODE}): ${BATCHES} batches, GPUs=${CUDA_VISIBLE_DEVICES}"
else
  MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-2048}"
  BATCHES="${BATCHES:-128}"
  RUN_OBSERVER_ONLY="${RUN_OBSERVER_ONLY:-false}"
  LOG="${LOG:-artifacts/glm_5_1_monolithic.log}"
  OUT_NAME="observations_${BATCHES}_cosine-seed_${SEED}.pt"
  echo "Monolithic full (${GLM51_VARIANT}, load=${GLM51_LOAD_MODE}): ${BATCHES} batches, run_observer_only=${RUN_OBSERVER_ONLY}"
fi

"$PYTHON" src/reap/prune.py \
  --model-name "$MODEL" \
  --dataset-name "$DATASET" \
  --compression-ratio "$COMPRESSION" \
  --prune-method reap \
  --profile false \
  --do-eval false \
  --distance_measure cosine \
  --seed "$SEED" \
  --output_file_name "$OUT_NAME" \
  --batch_size 1 \
  --model_max_length "$MODEL_MAX_LENGTH" \
  --batches_per_category "$BATCHES" \
  --record_pruning_metrics_only true \
  --run_observer_only "$RUN_OBSERVER_ONLY" \
  --overwrite_observations "${OVERWRITE_OBSERVATIONS:-true}" \
  2>&1 | tee "$LOG"
