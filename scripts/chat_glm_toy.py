#!/usr/bin/env python3
"""Interactive chat with a local GLM-5.1-toy (or pruned) checkpoint.

The toy weights are random-init — expect nonsense; use this to verify load/generate/chat template.

Examples:
  .venv/bin/python scripts/chat_glm_toy.py
  .venv/bin/python scripts/chat_glm_toy.py --model artifacts/GLM-5.1-toy/evol-codealpaca-v1/pruned_models/reap-renorm_true-seed_42-0.25
"""

from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _model_inputs(batch, device: torch.device) -> dict[str, torch.Tensor]:
    if hasattr(batch, "items"):
        return {k: v.to(device) for k, v in batch.items()}
    return {"input_ids": batch.to(device)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with GLM-5.1-toy (or pruned) checkpoint")
    parser.add_argument(
        "--model",
        default="artifacts/GLM-5.1-toy/evol-codealpaca-v1/pruned_models/reap-renorm_true-seed_42-0.25",
        help="HF model directory (base or pruned)",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--greedy", action="store_true", help="Disable sampling")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Loading {args.model} on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, device_map=str(device)
    )
    model.eval()

    history: list[dict[str, str]] = []
    print("GLM toy chat (random weights → gibberish is normal). Empty line or Ctrl-D to quit.\n")

    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not user:
            break

        history.append({"role": "user", "content": user})
        batch = tokenizer.apply_chat_template(
            history,
            return_tensors="pt",
            add_generation_prompt=True,
            tokenize=True,
            enable_thinking=False,
        )
        inputs = _model_inputs(batch, device)
        input_len = inputs["input_ids"].shape[-1]

        gen_kwargs: dict = {"max_new_tokens": args.max_new_tokens}
        if args.greedy:
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs.update(do_sample=True, temperature=args.temperature, top_p=0.8)

        with torch.no_grad():
            out = model.generate(**inputs, **gen_kwargs)

        new_tokens = out[0, input_len:]
        reply = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        history.append({"role": "assistant", "content": reply})
        print(f"Assistant: {reply}\n")


if __name__ == "__main__":
    main()
