#!/usr/bin/env bash
# Phase A: full paper-protocol GLM-5.1 calibration + eval (multi-GPU recommended).
# On 8x B300 (~2.3 TB VRAM) prefer monolithic calibration; layerwise remains for smaller nodes.
# BF16 (~1.5TB) or FP8 (~750GB). vLLM 0.10 may not serve glm_moe_dsa — eval uses HF backends.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

unset PYTHONPATH
PYTHON="${REPO_ROOT}/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON=python
fi

MODEL="${MODEL:-zai-org/GLM-5.1}"
GPUS="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
SEEDS="${SEEDS:-42}"
COMPRESSION="${COMPRESSION:-0.25}"
# monolithic | layerwise — monolithic is default on large multi-GPU nodes (8x B300/H200)
CALIBRATION_MODE="${CALIBRATION_MODE:-monolithic}"

export HF_HOME="${HF_HOME:-/tmp/hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Agentic calibration mix (24576 samples total)
COMPOSITE_DATASET='theblackcat102/evol-codealpaca-v1:4096,Salesforce/xlam-function-calling-60k:4096,open-r1/Mixture-of-Thoughts[code]:4096,open-r1/Mixture-of-Thoughts[math]:4096,open-r1/Mixture-of-Thoughts[science]:4096,SWE-bench/SWE-smith-trajectories(tool):4096'

_run_calibration() {
  local seed="$1"
  if [[ "$CALIBRATION_MODE" == "monolithic" ]]; then
    echo "Phase A: monolithic REAP (seed=${seed}, MODEL=${MODEL})"
    CUDA_VISIBLE_DEVICES="$GPUS" \
      SEED="$seed" \
      COMPRESSION="$COMPRESSION" \
      SMOKE=false \
      MODEL="$MODEL" \
      DATASET="$COMPOSITE_DATASET" \
      COMPOSITE_DATASET="$COMPOSITE_DATASET" \
      BATCHES=128 \
      MODEL_MAX_LENGTH=16384 \
      RUN_OBSERVER_ONLY=false \
      bash scripts/glm_5_1_monolithic.sh
  else
    echo "Phase A: layerwise REAP (seed=${seed}, MODEL=${MODEL})"
    CUDA_VISIBLE_DEVICES="$GPUS" \
      bash experiments/pruning-layerwise-cli.sh \
      "$GPUS" "$MODEL" reap "$seed" "$COMPRESSION" \
      "$COMPOSITE_DATASET" false false false false false
  fi
}

_run_eval() {
  local seed="$1"
  local pruned_path=""
  if [[ "$CALIBRATION_MODE" == "monolithic" ]]; then
    local model_slug
    model_slug="$(echo "$MODEL" | tr '/:' '_')"
    # shellcheck disable=SC2012
    pruned_path="$(ls -d "artifacts/${model_slug}/composite_"*/pruned_models/reap-renorm_true-seed_"${seed}"-"${COMPRESSION}" 2>/dev/null | head -1)"
    if [[ -z "$pruned_path" ]]; then
      pruned_path="$(ls -d "artifacts/${model_slug}/"*/pruned_models/reap-renorm_true-seed_"${seed}"-"${COMPRESSION}" 2>/dev/null | head -1)"
    fi
  else
    # shellcheck disable=SC2012
    pruned_path="$(ls -d artifacts/zai-org_GLM-5.1/composite_*/pruned_models/layerwise_reap-renorm_true-seed_"${seed}"-"${COMPRESSION}" 2>/dev/null | head -1)"
  fi

  if [[ -z "${pruned_path:-}" ]]; then
    echo "No pruned model for seed=${seed}; skip eval"
    return 1
  fi

  echo "Phase A eval (HF backends): ${pruned_path}"
  CUDA_VISIBLE_DEVICES="$GPUS" \
    "$PYTHON" src/reap/eval.py \
    --model-name "$pruned_path" \
    --vllm_port 8000 \
    --server_log_file_name "glm-5.1-paper-eval-seed_${seed}.log" \
    --run-lm-eval true \
    --run-evalplus true \
    --run-livecodebench true \
    --run-wildbench false \
    --run-math false \
    --use-server false \
    --results_dir "$pruned_path/eval" \
    --seed "$seed"
}

for seed in $SEEDS; do
  echo "======== Phase A seed=${seed} ========"
  _run_calibration "$seed"
  _run_eval "$seed" || true
done

echo "Phase A complete for SEEDS=${SEEDS}"
