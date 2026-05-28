#!/usr/bin/env bash
# ~10h plan: HF eval on pruned checkpoint, then 64-batch evol monolithic (no PR).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export PATH="${HOME}/.local/bin:${PATH}"
export HF_HOME="${HF_HOME:-/tmp/hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

PYTHON="${REPO_ROOT}/.venv/bin/python"
PRUNED="${PRUNED_MODEL:-${REPO_ROOT}/artifacts/GLM-5.1-FP8/evol-codealpaca-v1/pruned_models/reap-renorm_true-seed_42-0.25}"
SEED="${SEED:-42}"
LOG="${LOG:-/tmp/glm51-10h-plan.log}"
SKIP_TESTS="${SKIP_TESTS:-false}"

: >"$LOG"

if [[ "$SKIP_TESTS" != "true" ]]; then
echo "=== 0/3 Cold unit tests ===" | tee "$LOG"
"$PYTHON" -m pytest \
  tests/test_pruning_e2e.py::test_glm_moe_dsa_fused_expert_count_attrs \
  tests/test_pruning_e2e.py::test_glm_moe_dsa_layerwise_fused_expert_count_attrs \
  tests/test_layerwise_model_utils.py -q --tb=short \
  2>&1 | tee -a "$LOG"
else
  echo "=== 0/3 Cold unit tests (skipped) ===" | tee "$LOG"
fi

echo "=== 1/3 HF eval on ${PRUNED} ===" | tee -a "$LOG"
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
  2>&1 | tee -a "$LOG" || echo "WARN: eval failed (see log); continuing to monolithic" | tee -a "$LOG"

echo "=== 2/3 Monolithic 64-batch evol (ctx 2048) ===" | tee -a "$LOG"
SMOKE=false BATCHES=64 MODEL_MAX_LENGTH=2048 \
  DATASET=theblackcat102/evol-codealpaca-v1 \
  RUN_OBSERVER_ONLY=false \
  OVERWRITE_OBSERVATIONS=true \
  bash scripts/glm_5_1_monolithic.sh \
  2>&1 | tee -a "$LOG"

echo "=== 3/3 Collate results ===" | tee -a "$LOG"
"$PYTHON" scripts/report_results.py 2>&1 | tee -a "$LOG" || true

echo "10h plan finished. Log: $LOG"
