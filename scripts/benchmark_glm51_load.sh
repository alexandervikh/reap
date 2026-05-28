#!/usr/bin/env bash
# Time GLM-5.1 load modes (load only; no observer). Logs to /tmp/glm51-load-bench.log
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
# shellcheck source=scripts/glm51_env.sh
source "${REPO_ROOT}/scripts/glm51_env.sh"

PYTHON="${REPO_ROOT}/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON=python
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
LOG="${LOG:-/tmp/glm51-load-bench.log}"
MODES="${MODES:-lustre local fast_ram cpu_dispatch}"

: >"$LOG"
echo "benchmark start variant=$GLM51_VARIANT model=$GLM51_MODEL_PATH" | tee -a "$LOG"

for mode in $MODES; do
  echo "=== GLM51_LOAD_MODE=$mode ===" | tee -a "$LOG"
  export GLM51_LOAD_MODE="$mode"
  START=$(date +%s)
  RSS_BEFORE=$(grep -E '^VmRSS:' /proc/self/status | awk '{print $2}')
  "$PYTHON" - <<'PY' 2>&1 | tee -a "$LOG"
import os, resource, time, torch
from reap.model_util import patched_model_map, load_causal_lm_for_prune

model_id = os.environ.get("GLM51_MODEL_PATH") or "zai-org/GLM-5.1"
model_name = patched_model_map(model_id)
mode = os.environ["GLM51_LOAD_MODE"]
t0 = time.perf_counter()
model = load_causal_lm_for_prune(model_name)
dt = time.perf_counter() - t0
rss_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2)
vram = []
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        vram.append(torch.cuda.memory_allocated(i) / 1e9)
del model
torch.cuda.empty_cache()
print(f"mode={mode} wall_s={dt:.1f} peak_rss_gib≈{rss_gib:.1f} vram_gib_per_gpu={vram}")
PY
  ELAPSED=$(( $(date +%s) - START ))
  echo "wall_clock=${ELAPSED}s rss_before_kb=$RSS_BEFORE" | tee -a "$LOG"
done

echo "Wrote $LOG"
