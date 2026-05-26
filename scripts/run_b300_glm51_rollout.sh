#!/usr/bin/env bash
# B300 rollout driver: SMOKE -> monolithic full prune -> Phase A multi-seed eval.
# Requires GLM-5.1-FP8 under artifacts/models/ (or HF cache via patch_glm_5_1.py).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export PATH="${HOME}/.local/bin:${PATH}"
export HF_HOME="${HF_HOME:-/tmp/hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

PYTHON="${REPO_ROOT}/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  echo "Run bash scripts/build.sh first"
  exit 1
fi

echo "=== Phase 1: SMOKE ==="
SMOKE=true bash scripts/glm_5_1_monolithic.sh

echo "=== Phase 2: Monolithic full (FP8, 128 batches) ==="
SMOKE=false BATCHES=128 MODEL_MAX_LENGTH=2048 RUN_OBSERVER_ONLY=false \
  bash scripts/glm_5_1_monolithic.sh

echo "=== Phase 4: Phase A (monolithic calibration, SEEDS 42 11 99) ==="
SEEDS="42 11 99" CALIBRATION_MODE=monolithic MODEL=zai-org/GLM-5.1-FP8 \
  bash scripts/glm_5_1_phase_a.sh

echo "=== Collate results ==="
"$PYTHON" scripts/report_results.py || true

echo "Rollout complete."
