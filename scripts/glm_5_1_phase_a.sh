#!/usr/bin/env bash
# Phase A: full paper-protocol GLM-5.1 calibration + eval (multi-GPU recommended).
# BF16 (~1.5TB) or FP8 (~750GB); 8x H200 typical for layerwise calibration at 16384 ctx.

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
SEED="${SEED:-42}"
COMPRESSION="${COMPRESSION:-0.25}"

# Agentic calibration mix (24576 samples total), model_max_length=16384 via layerwise CLI defaults
COMPOSITE_DATASET='theblackcat102/evol-codealpaca-v1:4096,Salesforce/xlam-function-calling-60k:4096,open-r1/Mixture-of-Thoughts[code]:4096,open-r1/Mixture-of-Thoughts[math]:4096,open-r1/Mixture-of-Thoughts[science]:4096,SWE-bench/SWE-smith-trajectories(tool):4096'

echo "Phase A: layerwise REAP prune with paper calibration mix"
bash experiments/pruning-layerwise-cli.sh \
  "$GPUS" "$MODEL" reap "$SEED" "$COMPRESSION" \
  "$COMPOSITE_DATASET" true true true false false

PRUNED_DIR="artifacts/zai-org_GLM-5.1/composite_*/pruned_models/layerwise_reap-renorm_true-seed_${SEED}-${COMPRESSION}"
echo "Pruned artifact (glob): $PRUNED_DIR"

echo "Phase A: paper Table 3 eval (EvalPlus, LiveCodeBench, BFCL, MC)"
# shellcheck disable=SC2012
PRUNED_PATH="$(ls -d artifacts/zai-org_GLM-5.1/composite_*/pruned_models/layerwise_reap-renorm_true-seed_${SEED}-${COMPRESSION} 2>/dev/null | head -1)"
if [[ -n "${PRUNED_PATH:-}" ]]; then
  bash experiments/eval.sh \
    "$PRUNED_PATH" \
    "$SEED" \
    8000 \
    "glm-5.1-paper-eval.log" \
    true true true false false
else
  echo "Set PRUNED_PATH manually and run experiments/eval.sh"
fi

# Reproduce Table 3 confidence intervals: repeat with seeds 11 and 99
# for S in 42 11 99; do SEED=$S bash scripts/glm_5_1_phase_a.sh; done
