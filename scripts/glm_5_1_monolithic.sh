#!/usr/bin/env bash
# Paper-style monolithic REAP for GLM-5.1-FP8: prune.py + MoE hooks during full forward.
# Same path as GLM-4.5 / Qwen (not layerwise). Use SMOKE=true first (2 batches, observer-only).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

unset PYTHONPATH
PYTHON="${REPO_ROOT}/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON=python
fi

if [[ ! -d "${REPO_ROOT}/artifacts/models/GLM-5.1-FP8" ]]; then
  "$PYTHON" scripts/patch_glm_5_1.py
fi

if [[ -d "${REPO_ROOT}/artifacts/models/GLM-5.1-FP8" ]]; then
  MODEL="${MODEL:-${REPO_ROOT}/artifacts/models/GLM-5.1-FP8}"
else
  MODEL="${MODEL:-zai-org/GLM-5.1-FP8}"
fi

SEED="${SEED:-42}"
COMPRESSION="${COMPRESSION:-0.25}"
DATASET="${DATASET:-theblackcat102/evol-codealpaca-v1}"
SMOKE="${SMOKE:-true}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ "$SMOKE" == "true" ]]; then
  BATCHES=2
  RUN_OBSERVER_ONLY="true"
  MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-512}"
  LOG="${LOG:-artifacts/glm_5_1_monolithic_smoke.log}"
  OUT_NAME="observations_smoke_cosine-seed_${SEED}.pt"
  echo "Monolithic smoke: ${BATCHES} batches, batch_size=1, model_max_length=${MODEL_MAX_LENGTH}, observer-only"
else
  MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-2048}"
  BATCHES="${BATCHES:-128}"
  RUN_OBSERVER_ONLY="${RUN_OBSERVER_ONLY:-false}"
  LOG="${LOG:-artifacts/glm_5_1_monolithic.log}"
  OUT_NAME="observations_${BATCHES}_cosine-seed_${SEED}.pt"
  echo "Monolithic full: ${BATCHES} batches, batch_size=1, model_max_length=${MODEL_MAX_LENGTH}, run_observer_only=${RUN_OBSERVER_ONLY}"
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
  --overwrite_observations true \
  2>&1 | tee "$LOG"
