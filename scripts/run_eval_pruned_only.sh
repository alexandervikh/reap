#!/usr/bin/env bash
# HF-backend eval on the short-calib pruned GLM-5.1 checkpoint (no vLLM).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export PATH="${HOME}/.local/bin:${PATH}"
export HF_HOME="${HF_HOME:-/tmp/hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

PYTHON="${REPO_ROOT}/.venv/bin/python"
PRUNED="${PRUNED_MODEL:-${REPO_ROOT}/artifacts/GLM-5.1/evol-codealpaca-v1/pruned_models/reap-renorm_true-seed_42-0.25}"
SEED="${SEED:-42}"
LOG="${LOG:-/tmp/glm51-eval-pruned.log}"

: >"$LOG"
echo "=== HF eval on ${PRUNED} ===" | tee "$LOG"
"$PYTHON" src/reap/eval.py \
  --model-name "$PRUNED" \
  --vllm_port 8000 \
  --server_log_file_name "glm-5.1-eval-seed_${SEED}.log" \
  --run-lm-eval true \
  --run-evalplus true \
  --run-livecodebench true \
  --run-wildbench false \
  --run-math false \
  --use-server false \
  --results_dir "$PRUNED/eval" \
  --seed "$SEED" \
  2>&1 | tee -a "$LOG"
