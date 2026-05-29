import torch
from transformers import GlmMoeDsaConfig, GlmMoeDsaForCausalLM

from reap.observer import glm_moe_dsa_route_tokens
from reap.pruning_metrics import (
    initialize_pruning_state,
    scatter_topk_routing_weights,
    update_pruning_state_from_selected_experts,
)


def _make_sparse_glm_model(*, norm_topk_prob: bool = False):
    model = GlmMoeDsaForCausalLM(
        GlmMoeDsaConfig(
            vocab_size=32,
            hidden_size=64,
            intermediate_size=128,
            moe_intermediate_size=32,
            num_hidden_layers=1,
            mlp_layer_types=["sparse"],
            num_attention_heads=2,
            num_key_value_heads=2,
            n_routed_experts=8,
            num_experts_per_tok=2,
            n_shared_experts=1,
            n_group=2,
            topk_group=1,
            q_lora_rank=16,
            kv_lora_rank=16,
            qk_nope_head_dim=12,
            qk_rope_head_dim=4,
            v_head_dim=16,
            index_n_heads=2,
            index_head_dim=8,
            index_topk=4,
            indexer_types=["full"],
            norm_topk_prob=norm_topk_prob,
            routed_scaling_factor=2.5,
        )
    )
    model.eval()
    return model


def test_glm_route_differs_from_naive_topk_with_bias():
    torch.manual_seed(0)
    model = _make_sparse_glm_model()
    moe = model.model.layers[0].mlp
    moe.gate.e_score_correction_bias.copy_(
        torch.tensor([0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    )

    router_logits = moe.gate(torch.randn(4, model.config.hidden_size))
    naive_topk = torch.topk(router_logits, moe.top_k, dim=-1).indices
    glm_topk, _ = glm_moe_dsa_route_tokens(moe, router_logits)

    assert not torch.equal(naive_topk, glm_topk)


def test_glm_pruning_metrics_use_sigmoid_weights_not_softmax():
    torch.manual_seed(1)
    model = _make_sparse_glm_model()
    moe = model.model.layers[0].mlp
    router_logits = moe.gate(torch.randn(3, model.config.hidden_size))
    selected_experts, topk_weights = glm_moe_dsa_route_tokens(moe, router_logits)
    routing_weights = scatter_topk_routing_weights(
        selected_experts, topk_weights, moe.n_routed_experts
    )

    expert_activations = {}
    for expert_idx in torch.unique(selected_experts).tolist():
        mask = (selected_experts == expert_idx).any(dim=-1)
        expert_activations[int(expert_idx)] = torch.randn(
            int(mask.sum()), 8, dtype=torch.float32
        )

    glm_state = initialize_pruning_state(moe.n_routed_experts)
    update_pruning_state_from_selected_experts(
        glm_state,
        expert_activations=expert_activations,
        selected_experts=selected_experts,
        router_logits=router_logits,
        num_experts=moe.n_routed_experts,
        routing_weights=routing_weights,
        renormalize_router_weights=False,
    )

    softmax_state = initialize_pruning_state(moe.n_routed_experts)
    update_pruning_state_from_selected_experts(
        softmax_state,
        expert_activations=expert_activations,
        selected_experts=selected_experts,
        router_logits=router_logits,
        num_experts=moe.n_routed_experts,
        renormalize_router_weights=False,
    )

    assert not torch.allclose(
        glm_state["weighted_ean_sum"],
        softmax_state["weighted_ean_sum"],
    )
    assert glm_state["weighted_expert_frequency_sum"].sum() > 0
