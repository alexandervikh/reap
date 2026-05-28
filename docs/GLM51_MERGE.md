# Merging `glm51` into `main`

Suggested PR sequence (smallest risk first):

1. **Fused prune refactor** — [src/reap/prune.py](../src/reap/prune.py) (`_set_fused_expert_counts`, `_set_fused_router_counts`) + `test_fused_prune_without_experts_num_experts_attribute`.
2. **Multi-GPU load plumbing** — [src/reap/model_util.py](../src/reap/model_util.py) (`build_auto_max_memory`, `dispatch_model_to_auto`, `get_from_pretrained_kwargs`), [src/reap/args.py](../src/reap/args.py) (`load_device_map`, `prune_on_cpu`), [src/reap/layerwise_prune.py](../src/reap/layerwise_prune.py).
3. **GLM-5.1 / GlmMoeDsa** — observer hooks, `MODEL_ATTRS`, `router_input_ndim`, tests, eval HF fallback ([src/reap/eval.py](../src/reap/eval.py)).
4. **Scripts + README** — [scripts/glm_5_1_*.sh](../scripts/), GLM-5.1 section in [README.md](../README.md).

## Pre-merge checklist

```bash
export PATH="$HOME/.local/bin:$PATH"
bash scripts/build.sh
.venv/bin/pytest tests/test_pruning_e2e.py tests/test_layerwise_model_utils.py -q
```

Confirm `transformers>=5.4.0` still passes all factories in `tests/test_layerwise_model_utils.py` (`MODEL_FACTORIES`, `MOE_LOOKUP_CASES`).

## Hardware note (8× B300)

On this node, prefer **monolithic** calibration (`CALIBRATION_MODE=monolithic bash scripts/glm_5_1_phase_a.sh`). Skip `scripts/glm_5_1_phase_b.sh` unless reproducing single-GPU behavior.

## Load speedup (BF16 default)

GLM-5.1 on Lustre can take **~80 min** to load (serial HF `from_pretrained`, low GPU util). Prefer **BF16** (`zai-org/GLM-5.1`) — no FP8 Blackwell fallback or `dequantize_fp8` at load.

### Download and stage

```bash
bash scripts/build.sh
# Lustre may be full on B300 nodes — download directly to /tmp:
.venv/bin/python scripts/download_glm_5_1.py --variant bf16 --dest /tmp/GLM-5.1
# Or from Lustre copy: GLM51_VARIANT=bf16 bash scripts/stage_glm51_local.sh
```

FP8: `GLM51_VARIANT=fp8` and `scripts/patch_glm_5_1.py` (or `download_glm_5_1.py --variant fp8`).

### `GLM51_LOAD_MODE`

Set in the environment (wired via `load_causal_lm_for_prune` in `src/reap/model_util.py`):

| Mode | Effect |
|------|--------|
| `lustre` | Use `artifacts/models/GLM-5.1` (default path resolution) |
| `local` | Prefer `/tmp/GLM-5.1` if staged; standard `low_cpu_mem_usage=True` |
| `fast_ram` | Staged path + `low_cpu_mem_usage=False` (~3.9 TiB RAM on B300 node) |
| `cpu_dispatch` | Load on CPU, then `dispatch_model_to_auto()` |

Scripts source `scripts/glm51_env.sh` (`GLM51_VARIANT=bf16`, `GLM51_LOAD_MODE=local` by default).

```bash
export GLM51_LOAD_MODE=fast_ram
SMOKE=true bash scripts/glm_5_1_monolithic.sh
bash scripts/benchmark_glm51_load.sh   # log: /tmp/glm51-load-bench.log
```

**Note:** Observations from an FP8 run are not interchangeable with BF16 prune unless you re-run the observer on BF16.
