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
