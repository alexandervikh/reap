"""GPU load / dispatch helpers for large fused MoE checkpoints."""

import pytest
import torch
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

from reap.model_util import (
    build_auto_max_memory,
    dispatch_model_to_auto,
    get_model_device,
    model_weights_all_on_cpu,
)


def _tiny_qwen():
    model = Qwen3MoeForCausalLM(
        Qwen3MoeConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            moe_intermediate_size=8,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            num_experts=3,
            num_experts_per_tok=1,
            norm_topk_prob=False,
        )
    )
    model.eval()
    return model


def test_build_auto_max_memory_includes_cpu():
    mem = build_auto_max_memory(gpu_memory_fraction=0.5)
    assert "cpu" in mem
    if torch.cuda.is_available():
        assert 0 in mem


def test_get_model_device_from_parameters():
    model = _tiny_qwen()
    assert get_model_device(model) == next(model.parameters()).device


def test_model_weights_all_on_cpu():
    model = _tiny_qwen()
    assert model_weights_all_on_cpu(model)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_dispatch_model_to_auto_roundtrip():
    model = _tiny_qwen()
    assert model_weights_all_on_cpu(model)
    model = dispatch_model_to_auto(model)
    assert hasattr(model, "hf_device_map")
    assert not model_weights_all_on_cpu(model)
    assert get_model_device(model).type == "cuda"
