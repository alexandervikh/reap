from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Router hidden-state layout for calibration hooks: 2 = [tokens, hidden], 3 = [batch, seq, hidden].
ROUTER_INPUT_NDIM_DEFAULT = 2

# CPU spill budget for device_map="auto" (load + dispatch after layerwise observer).
_AUTO_DEVICE_MAP_CPU_HEADROOM_GIB = 1500
# Fraction of per-GPU VRAM usable for weights during load/dispatch. Fused MoE
# checkpoints spike during expert-tensor concat; 0.82 on 140GiB H200 ≈ 115GiB cap.
_AUTO_DEVICE_MAP_GPU_MEMORY_FRACTION = 0.82


MODEL_ATTRS = {
    "Qwen3MoeForCausalLM": {
        "moe_block": "mlp",
        "gate_proj": "gate_proj",
        "up_proj": "up_proj",
        "down_proj": "down_proj",
        "experts": "experts",
        "fused": False,
        "router": "gate",
        "num_experts": "num_experts",
        "num_experts_per_tok": "num_experts_per_tok",
    },
    "Qwen3-Coder-30B-A3B-Instruct": {
        "moe_block": "mlp",
        "gate_proj": "gate_proj",
        "up_proj": "up_proj",
        "down_proj": "down_proj",
        "experts": "experts",
        "fused": False,
        "router": "gate",
        "num_experts": "num_experts",
        "num_experts_per_tok": "num_experts_per_tok",
    },
    "NonUniformQwen3MoeForCausalLM": {
        "moe_block": "mlp",
        "gate_proj": "gate_proj",
        "up_proj": "up_proj",
        "down_proj": "down_proj",
        "experts": "experts",
        "fused": False,
        "router": "gate",
        "num_experts": "num_experts",
        "num_experts_per_tok": "num_experts_per_tok",
    },
    "Llama4ForCausalLM": {
        "moe_block": "feed_forward",
        "gate_proj": "gate_up_proj",
        "up_proj": "gate_up_proj",
        "down_proj": "down_proj",
        "experts": "experts",
        "fused": True,
        "router": "router",
        "num_experts": "num_local_experts",  # HuggingFace config field
        "moe_num_experts": "num_experts",  # Llama4TextMoe / Llama4Router runtime attr
        "fused_experts_count": "num_experts",  # Llama4TextExperts
        "num_experts_per_tok": "num_experts_per_tok",
    },
    "MixtralForCausalLM": {
        "moe_block": "block_sparse_moe",
        "gate_proj": "w3",
        "up_proj": "w1",
        "down_proj": "w2",
        "experts": "experts",
        "fused": False,
        "router": "gate",
        "num_experts": "num_local_experts",
        "num_experts_per_tok": "num_experts_per_tok",
    },
    "DeepseekV2ForCausalLM": {
        "moe_block": "mlp",
        "gate_proj": "gate_proj",
        "up_proj": "up_proj",
        "down_proj": "down_proj",
        "experts": "experts",
        "fused": False,
        "router": "gate",
        "router_input_ndim": 3,
        "num_experts": "n_routed_experts",
        "num_experts_per_tok": "num_experts_per_tok",
    },
    "Ernie4_5_MoEForCausalLM": {
        "moe_block": "mlp",
        "gate_proj": "gate_proj",
        "up_proj": "up_proj",
        "down_proj": "down_proj",
        "experts": "experts",
        "fused": False,
        "router": "gate",
        "num_experts": "moe_num_experts",
        "num_experts_per_tok": "num_experts_per_tok",
    },
    "Ernie4_5_MoeForCausalLM": {
        "moe_block": "mlp",
        "gate_proj": "gate_proj",
        "up_proj": "up_proj",
        "down_proj": "down_proj",
        "experts": "experts",
        "fused": False,
        "router": "gate",
        "num_experts": "moe_num_experts",
        "num_experts_per_tok": "moe_k",
    },
    "gpt-oss-20b": {
        "moe_block": "mlp",
        "gate_proj": "gate_proj",
        "up_proj": "up_proj",
        "down_proj": "down_proj",
        "experts": "experts",
        "fused": False,
        "router": "gate",
        "num_experts": "num_experts",
        "num_experts_per_tok": "num_experts_per_tok",
    },
    "Glm4MoeForCausalLM": {
        "moe_block": "mlp",
        "gate_proj": "gate_proj",
        "up_proj": "up_proj",
        "down_proj": "down_proj",
        "experts": "experts",
        "fused": False,
        "router": "gate",
        "num_experts": "n_routed_experts",
        "num_experts_per_tok": "num_experts_per_tok",
    },
    "GlmMoeDsaForCausalLM": {
        "moe_block": "mlp",
        "gate_proj": "gate_up_proj",
        "up_proj": "gate_up_proj",
        "down_proj": "down_proj",
        "experts": "experts",
        "fused": True,
        "router": "gate",
        "router_input_ndim": 2,
        "num_experts": "n_routed_experts",  # HuggingFace config field
        "moe_num_experts": "n_routed_experts",  # GlmMoeDsaMoE / GlmMoeDsaTopkRouter
        "fused_experts_count": "num_experts",  # GlmMoeDsaNaiveMoe (config.num_local_experts alias)
        "num_experts_per_tok": "num_experts_per_tok",
    },
}


def build_auto_max_memory(
    gpu_memory_fraction: float = _AUTO_DEVICE_MAP_GPU_MEMORY_FRACTION,
) -> dict[int | str, str]:
    """Per-device max_memory dict for accelerate / HF device_map='auto'."""
    max_memory: dict[int | str, str] = {}
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            total_gib = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            cap_gib = max(1, int(total_gib * gpu_memory_fraction))
            max_memory[i] = f"{cap_gib}GiB"
    max_memory["cpu"] = f"{_AUTO_DEVICE_MAP_CPU_HEADROOM_GIB}GiB"
    return max_memory


def get_model_device(model: nn.Module) -> torch.device:
    """Resolve a device for tensors when ``model.device`` is absent (sharded models)."""
    if hasattr(model, "device"):
        return model.device
    hf_map = getattr(model, "hf_device_map", None)
    if hf_map:
        for dev in hf_map.values():
            if isinstance(dev, int):
                return torch.device(f"cuda:{dev}")
            if isinstance(dev, str) and dev.startswith("cuda"):
                return torch.device(dev)
    return next(model.parameters()).device


def router_hidden_input_for_model(
    model: nn.Module,
    flat_input: torch.Tensor,
    *,
    batch_size: int,
    sequence_length: int,
    hidden_dim: int,
) -> torch.Tensor:
    """Shape hidden states for a model's router/gate (see MODEL_ATTRS router_input_ndim)."""
    ndim = MODEL_ATTRS.get(model.__class__.__name__, {}).get(
        "router_input_ndim", ROUTER_INPUT_NDIM_DEFAULT
    )
    if ndim == 3:
        return flat_input.view(batch_size, sequence_length, hidden_dim)
    if ndim != 2:
        raise ValueError(
            f"Unsupported router_input_ndim={ndim!r} for {model.__class__.__name__}"
        )
    return flat_input


def vllm_supported_for_eval(model_name: str) -> bool:
    """Whether Phase-A style eval can use a vLLM OpenAI server for this checkpoint."""
    lower = str(model_name).lower()
    if "glm-5.1" in lower or "glm_moe_dsa" in lower:
        return False
    return True


def get_model_input_device(model: nn.Module) -> torch.device:
    """Device for calibration batch tensors (embedding shard for ``device_map`` models)."""
    try:
        embed = model.get_input_embeddings()
        if embed is not None and hasattr(embed, "weight"):
            weight = embed.weight
            if str(weight.device) != "meta":
                return weight.device
    except Exception:
        pass
    return get_model_device(model)


def model_weights_all_on_cpu(model: nn.Module) -> bool:
    for param in model.parameters():
        if str(param.device) == "meta":
            continue
        if param.device.type != "cpu":
            return False
    return True


def dispatch_model_to_auto(
    model: nn.Module,
    gpu_memory_fraction: float = _AUTO_DEVICE_MAP_GPU_MEMORY_FRACTION,
) -> nn.Module:
    """Spread an in-memory (CPU) model across GPUs without reloading from disk."""
    from accelerate import dispatch_model, infer_auto_device_map
    from accelerate.hooks import remove_hook_from_submodules

    if not torch.cuda.is_available():
        return model

    try:
        remove_hook_from_submodules(model)
    except Exception:
        pass

    max_memory = build_auto_max_memory(gpu_memory_fraction)
    infer_kwargs: dict[str, Any] = {"max_memory": max_memory}
    no_split = getattr(model, "_no_split_modules", None)
    if no_split:
        infer_kwargs["no_split_module_classes"] = no_split

    device_map = infer_auto_device_map(model, **infer_kwargs)
    logger.info(
        "Dispatching model to GPUs for prune (max_memory per GPU: %s)",
        {k: v for k, v in max_memory.items() if k != "cpu"},
    )
    return dispatch_model(model, device_map=device_map)


def get_from_pretrained_kwargs(
    *,
    device_map: str = "auto",
    local_files_only: bool = False,
    low_cpu_mem_usage: bool | None = None,
    gpu_memory_fraction: float = _AUTO_DEVICE_MAP_GPU_MEMORY_FRACTION,
    dequantize_fp8: bool = False,
) -> dict[str, Any]:
    """Build kwargs for ``AutoModelForCausalLM.from_pretrained``."""
    kwargs: dict[str, Any] = {
        "torch_dtype": "auto",
        "trust_remote_code": True,
    }
    if local_files_only:
        kwargs["local_files_only"] = True
    if low_cpu_mem_usage is None and device_map == "auto" and torch.cuda.is_available():
        low_cpu_mem_usage = True
    if low_cpu_mem_usage is not None:
        kwargs["low_cpu_mem_usage"] = low_cpu_mem_usage

    if device_map == "auto" and torch.cuda.is_available():
        kwargs["device_map"] = "auto"
        kwargs["max_memory"] = build_auto_max_memory(gpu_memory_fraction)
    else:
        kwargs["device_map"] = device_map
    if dequantize_fp8:
        kwargs["dtype"] = torch.bfloat16
        kwargs["_dequantize_fp8_checkpoint"] = True
    return kwargs


def maybe_dequantize_fp8_config(model_name: str, load_kwargs: dict[str, Any]) -> None:
    """Attach ``quantization_config.dequantize=True`` for FP8 checkpoints on Blackwell."""
    if not load_kwargs.pop("_dequantize_fp8_checkpoint", False):
        return
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(
        model_name, trust_remote_code=True, local_files_only=load_kwargs.get("local_files_only", False)
    )
    qc = getattr(config, "quantization_config", None)
    if qc is None:
        return
    if isinstance(qc, dict):
        from transformers import FineGrainedFP8Config

        qc = FineGrainedFP8Config.from_dict({**qc, "dequantize": True})
    else:
        qc.dequantize = True
    load_kwargs["quantization_config"] = qc
    load_kwargs["dtype"] = torch.bfloat16
    load_kwargs.pop("torch_dtype", None)
    logger.info("Loading %s with quantization_config.dequantize=True (BF16 compute)", model_name)


def get_moe(model, layer):
    moe_attr_name = MODEL_ATTRS.get(model.__class__.__name__)["moe_block"]
    return getattr(model.model.layers[layer], moe_attr_name)


def assert_merge(model, merged_moe, cluster_label):
    model_attr = MODEL_ATTRS.get(model.__class__.__name__)
    assert hasattr(merged_moe, "experts"), (
        "The merged module must have an 'experts' attribute."
    )

    gate_proj = model_attr["gate_proj"]
    down_proj = model_attr["down_proj"]

    if model_attr["fused"]:
        for cluster_id in cluster_label.unique():
            expert_indices = torch.where(cluster_label == cluster_id)[0]
            dom_expert = expert_indices[0]
            for expert in expert_indices[1:]:
                assert torch.allclose(
                    getattr(merged_moe.experts, gate_proj)[dom_expert],
                    getattr(merged_moe.experts, gate_proj)[expert],
                ), f"Experts {expert_indices} are not merged correctly."
                assert torch.allclose(
                    getattr(merged_moe.experts, down_proj)[dom_expert],
                    getattr(merged_moe.experts, down_proj)[expert],
                ), f"Experts {expert_indices} are not merged correctly."
    else:
        up_proj = model_attr["up_proj"]
        for cluster_id in cluster_label.unique():
            expert_indices = torch.where(cluster_label == cluster_id)[0]
            dom_expert = expert_indices[0]
            for expert in expert_indices[1:]:
                assert (
                    getattr(merged_moe.experts[dom_expert], up_proj).weight
                    == getattr(merged_moe.experts[expert], up_proj).weight
                ).all(), f"Experts {expert_indices} are not merged correctly."
                assert (
                    getattr(merged_moe.experts[dom_expert], down_proj).weight
                    == getattr(merged_moe.experts[expert], down_proj).weight
                ).all(), f"Experts {expert_indices} are not merged correctly."
                assert (
                    getattr(merged_moe.experts[dom_expert], gate_proj).weight
                    == getattr(merged_moe.experts[expert], gate_proj).weight
                ).all(), f"Experts {expert_indices} are not merged correctly."


def patched_model_map(model: str):
    patched = False
    model_name = model

    if model == "deepseek-ai/DeepSeek-V2-Lite-Chat":
        patched = True
        model_name = "artifacts/models/DeepSeek-V2-Lite-Chat"

    # until hf version lands
    if model == "baidu/ERNIE-4.5-21B-A3B-PT":
        patched = True
        model_name = "artifacts/models/ERNIE-4.5-21B-A3B-PT"

    if model == "Qwen/NonUniformQwen3-30B-A3B":
        patched = True
        model_name = "artifacts/models/NonUniformQwen3-30B-A3B"

    if model == "zai-org/GLM-4.5-Air":
        patched = True
        model_name = "artifacts/models/GLM-4.5-Air"

    if model == "zai-org/GLM-4.5-Air-FP8":
        patched = True
        model_name = "artifacts/models/GLM-4.5-Air-FP8"

    if model == "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8":
        patched = True
        model_name = "artifacts/models/Qwen3-Coder-480B-A35B-Instruct-FP8"

    if model in ("zai-org/GLM-5.1-FP8", "GLM-5.1-FP8"):
        patched = True
        model_name = "artifacts/models/GLM-5.1-FP8"

    if model in ("zai-org/GLM-5.1", "GLM-5.1"):
        patched = True
        model_name = "artifacts/models/GLM-5.1"

    if patched:
        logger.info(f"Using patched model for {model} from: {model_name}")
    return model_name


def assert_tied_weights(model, clusters_labels):
    model_attrs = MODEL_ATTRS.get(model.__class__.__name__)
    for layer_idx in clusters_labels:
        clusters = clusters_labels[layer_idx]
        moe = get_moe(model, layer_idx)
        experts = getattr(moe, model_attrs["experts"])
        for cluster_idx in torch.unique(clusters):
            experts_in_cluster = torch.where(clusters == cluster_idx)[0].tolist()
            dom_expert = experts[experts_in_cluster[0]]
            for attr in ["up_proj", "down_proj", "gate_proj"]:
                for expert_idx in experts_in_cluster:
                    if expert_idx == dom_expert:
                        continue
                    expert = experts[expert_idx]
                    proj = getattr(expert, attr)
                    weight = proj.weight
                    dom_proj = getattr(dom_expert, attr)
                    dom_weight = dom_proj.weight
                    if not torch.allclose(weight, dom_weight):
                        print(
                            f"Weights for expert {expert_idx} in cluster {cluster_idx} for layer {layer_idx} and attr {attr} are not tied!"
                        )
                        print(f"Max diff: {torch.abs(weight - dom_weight).max()}")
                    # check adapters
                    for lora_adapter in ["lora_A", "lora_B"]:
                        if hasattr(proj, lora_adapter):
                            lora_weight = getattr(proj, lora_adapter).default.weight
                            dom_lora_weight = getattr(
                                dom_proj, lora_adapter
                            ).default.weight
                            if not torch.allclose(lora_weight, dom_lora_weight):
                                print(
                                    f"LoRA Weights for expert {expert_idx} in cluster {cluster_idx} for layer {layer_idx} and adapter {lora_adapter} are not tied!"
                                )
                                print(
                                    f"Max diff: {torch.abs(lora_weight - dom_lora_weight).max()}"
                                )

def get_super_expert_indices(observer_data, include_last_layers: bool = False):
    logger.info("Identifying super experts to preserve...")
    quantile = 99.5
    times = 10
    all_max_activations = [layer['max_activations'] for layer in observer_data.values()]
    num_layers = len(all_max_activations)
    all_max_activations = torch.cat(all_max_activations).flatten()
    percentile_threshold = torch.quantile(all_max_activations, quantile / 100.0).item()
    abs_threshold = all_max_activations.max().item() / times
    final_threshold = max(percentile_threshold, abs_threshold)
    # reshape back into per layer data
    all_max_activations = all_max_activations.reshape(num_layers, -1)
    super_experts_mask = all_max_activations > final_threshold
    if not include_last_layers:
        # only consider first 75% of layers for super experts
        logger.info(
            "Only considering first 75% of layers for super expert "
            "identification since perserve_outliers is False"
        )
        num_layers = int(num_layers * 0.75)
        super_experts_mask[num_layers:, :] = False
    super_expert_idx = torch.argwhere(super_experts_mask)
    logger.info(f"Identified {super_experts_mask.sum().item()} super experts with threshold: {final_threshold:.4f}")
    return super_expert_idx

def register_llama_with_vllm():
    from vllm.model_executor.models import ModelRegistry
    print("Registering Llama4ForCausalLM with vLLM")
    ModelRegistry.register_model("Llama4ForCausalLM", "vllm.model_executor.models.llama4:Llama4ForCausalLM")