#!/usr/bin/env python3
"""
Run the same generation check as reap.main.smoke_test on a tiny in-memory Qwen3-MoE.

Usage (from repo root):
    python scripts/run_smoke_test.py
"""
from __future__ import annotations

import logging

import torch
from torch import nn
from transformers import AutoTokenizer, Qwen3MoeConfig, Qwen3MoeForCausalLM

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)


def make_tiny_qwen_moe() -> Qwen3MoeForCausalLM:
    model = Qwen3MoeForCausalLM(
        Qwen3MoeConfig(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            moe_intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_experts=4,
            num_experts_per_tok=2,
            norm_topk_prob=False,
        )
    )
    model.eval()
    return model


@torch.no_grad()
def smoke_test(model: nn.Module, tokenizer: AutoTokenizer | None = None) -> None:
    """Mirror of reap.main.smoke_test (kept here to avoid heavy reap.main imports)."""
    if tokenizer is not None:
        prompt = "What is your name?"
        test_input = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            test_input,
            return_tensors="pt",
            add_generation_prompt=True,
            tokenize=True,
        ).to(model.device)
        decode = lambda out: tokenizer.batch_decode(out, skip_special_tokens=False)
    else:
        # Tiny random-init models use a small vocab; avoid tokenizer/HF download.
        inputs = torch.tensor([[1, 2, 3, 4, 5]], device=model.device)
        decode = lambda out: out.tolist()

    outputs = model.generate(inputs, max_new_tokens=50, do_sample=True)
    logger.info("Smoke test response: %s", decode(outputs)[0])


def main() -> None:
    import os

    device = os.environ.get("REAP_SMOKE_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    model = make_tiny_qwen_moe().to(device)

    logger.info("Device=%s num_experts=%s", device, model.config.num_experts)
    smoke_test(model)
    logger.info("Smoke test completed.")


if __name__ == "__main__":
    main()
