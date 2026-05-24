#!/usr/bin/env bash
# Phase B: GLM-5.1-FP8 dev calibration + optional smoke eval on a single GPU.
# Requires ~750GB model weights; use layerwise observer (memory-efficient calibration).
# On a single 143GB H200, weights alone may not fit — use multi-GPU device map or a larger node.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

unset PYTHONPATH
PYTHON="${REPO_ROOT}/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON=python
fi

# 1) Download FP8 checkpoint (skip if already present)
if [[ ! -d "artifacts/models/GLM-5.1-FP8" ]]; then
  "$PYTHON" scripts/patch_glm_5_1.py
fi

if [[ -d "${REPO_ROOT}/artifacts/models/GLM-5.1-FP8" ]]; then
  MODEL="${MODEL:-${REPO_ROOT}/artifacts/models/GLM-5.1-FP8}"
else
  MODEL="${MODEL:-zai-org/GLM-5.1-FP8}"
fi
GPU="${CUDA_VISIBLE_DEVICES:-0}"
SEED="${SEED:-42}"
COMPRESSION="${COMPRESSION:-0.25}"
DATASET="${DATASET:-theblackcat102/evol-codealpaca-v1}"

echo "Phase B: layerwise REAP prune (128 batches, no eval)"
"$PYTHON" -m reap.layerwise_prune \
  --model-name "$MODEL" \
  --dataset-name "$DATASET" \
  --compression-ratio "$COMPRESSION" \
  --prune-method reap \
  --do-eval false \
  --seed "$SEED" \
  --output_file_name "observations_128_cosine-seed_${SEED}.pt" \
  --batches_per_category 128 \
  --batch_size 8 \
  --low_cpu_mem_usage True \
  2>&1 | tee artifacts/glm_5_1_layerwise_prune.log

if [[ "$MODEL" == *"GLM-5.1-FP8"* ]]; then
  SHORT_MODEL="GLM-5.1-FP8"
else
  SHORT_MODEL="zai-org_GLM-5.1-FP8"
fi
PRUNED_DIR="artifacts/${SHORT_MODEL}/theblackcat102_evol-codealpaca-v1/pruned_models/layerwise_reap-renorm_true-seed_${SEED}-${COMPRESSION}"

echo "Phase B: manual HF lm-eval smoke (limit 200; slow at 744B scale)"
if [[ -d "$PRUNED_DIR" ]]; then
  "$PYTHON" -m lm_eval --model hf \
    --model_args "pretrained=${PRUNED_DIR},trust_remote_code=True" \
    --tasks arc_easy,openbookqa \
    --batch_size 1 \
    --limit 200 \
    || echo "lm-eval failed (OOM or missing deps); see README GLM-5.1 section"
else
  echo "Pruned model not found at $PRUNED_DIR — prune step may have failed."
fi
