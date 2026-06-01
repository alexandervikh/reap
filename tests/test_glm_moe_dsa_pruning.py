from types import SimpleNamespace

import torch
from safetensors.torch import load_file
from transformers import GlmMoeDsaConfig, GlmMoeDsaForCausalLM

from reap.args import PruneArgs
from reap.model_util import get_moe
from reap.prune import prune


def _make_glm_moe_dsa_model():
    model = GlmMoeDsaForCausalLM(
        GlmMoeDsaConfig(
            vocab_size=32,
            hidden_size=64,
            intermediate_size=128,
            moe_intermediate_size=32,
            num_hidden_layers=3,
            mlp_layer_types=["dense", "dense", "sparse"],
            num_attention_heads=2,
            num_key_value_heads=2,
            n_routed_experts=4,
            num_experts_per_tok=1,
            n_shared_experts=1,
            q_lora_rank=16,
            kv_lora_rank=16,
            qk_nope_head_dim=12,
            qk_rope_head_dim=4,
            v_head_dim=16,
            index_n_heads=2,
            index_head_dim=8,
            index_topk=4,
            indexer_types=["full", "full", "full"],
            norm_topk_prob=False,
        )
    )
    model.eval()
    return model


def _run_prune(observer_data, model, tmp_path, subdir_name):
    pruned_model_dir = tmp_path / subdir_name
    prune_args = PruneArgs(prune_method="frequency", n_experts_to_prune=1)
    prune(
        observer_data=observer_data,
        model=model,
        prune_args=prune_args,
        n_experts_to_prune=prune_args.n_experts_to_prune,
        pruned_model_dir=pruned_model_dir,
    )
    return pruned_model_dir, prune_args.n_experts_to_prune


def _assert_glm_moe_dsa_fused_expert_counts(model, layer_idx: int, expected: int) -> None:
    assert model.config.n_routed_experts == expected
    moe = get_moe(model, layer_idx)
    assert moe.n_routed_experts == expected
    assert moe.experts.num_experts == expected
    assert moe.gate.n_routed_experts == expected
    assert not hasattr(moe, "num_experts")


def test_glm_moe_dsa_fused_expert_count_attrs(tmp_path):
    torch.manual_seed(0)
    model = _make_glm_moe_dsa_model()
    observer_data = {
        2: {
            "expert_frequency": torch.tensor([1.0, 2.0, 3.0, 4.0]),
            "total_tokens": torch.tensor(10.0),
            "reap": torch.tensor([4.0, 3.0, 2.0, 1.0]),
        }
    }
    pruned_dir, n_experts_to_prune = _run_prune(
        observer_data=observer_data,
        model=model,
        tmp_path=tmp_path,
        subdir_name="glm_moe_dsa_fused_counts",
    )
    assert pruned_dir.exists()
    _assert_glm_moe_dsa_fused_expert_counts(model, 2, 4 - n_experts_to_prune)


def test_glm_moe_dsa_offloaded_fused_experts_save_with_native_keys(tmp_path):
    torch.manual_seed(0)
    model = _make_glm_moe_dsa_model()
    moe = get_moe(model, 2)
    experts = moe.experts
    gate_up_proj = experts.gate_up_proj.detach().clone()
    down_proj = experts.down_proj.detach().clone()
    gate_weight = moe.gate.weight.detach().clone()
    experts.gate_up_proj = torch.nn.Parameter(
        torch.empty_like(gate_up_proj, device="meta")
    )
    experts.down_proj = torch.nn.Parameter(torch.empty_like(down_proj, device="meta"))
    moe.gate.weight = torch.nn.Parameter(torch.empty_like(gate_weight, device="meta"))
    experts._hf_hook = SimpleNamespace(
        weights_map={
            "gate_up_proj": gate_up_proj,
            "down_proj": down_proj,
        }
    )
    moe.gate._hf_hook = SimpleNamespace(weights_map={"weight": gate_weight})
    model.hf_device_map = {
        "model.layers.2.mlp.experts": "cpu",
        "model.layers.2.mlp.gate": "cpu",
        "lm_head": 0,
    }

    observer_data = {
        2: {
            "expert_frequency": torch.tensor([1.0, 2.0, 3.0, 4.0]),
            "total_tokens": torch.tensor(10.0),
        }
    }
    pruned_dir, n_experts_to_prune = _run_prune(
        observer_data=observer_data,
        model=model,
        tmp_path=tmp_path,
        subdir_name="glm_moe_dsa_offloaded_fused_save",
    )
    expected_count = 4 - n_experts_to_prune

    _assert_glm_moe_dsa_fused_expert_counts(model, 2, expected_count)

    saved_tensors = load_file(pruned_dir / "model.safetensors")
    assert saved_tensors["model.layers.2.mlp.experts.gate_up_proj"].shape[0] == expected_count
    assert saved_tensors["model.layers.2.mlp.experts.down_proj"].shape[0] == expected_count
    assert saved_tensors["model.layers.2.mlp.gate.weight"].shape[0] == expected_count
    assert torch.equal(
        saved_tensors["model.layers.2.mlp.experts.gate_up_proj"],
        gate_up_proj[1:],
    )
    assert torch.equal(saved_tensors["model.layers.2.mlp.experts.down_proj"], down_proj[1:])
    assert torch.equal(saved_tensors["model.layers.2.mlp.gate.weight"], gate_weight[1:])
    assert not any(".experts.0." in key for key in saved_tensors)
