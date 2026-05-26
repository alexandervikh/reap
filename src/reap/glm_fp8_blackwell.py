"""Blackwell (B300) fallback for GLM-5.1-FP8 finegrained FP8 Triton kernels.

The ``kernels-community/finegrained-fp8`` build does not yet target ``sm_103a``.
REAP loads FP8 checkpoints as-is and uses BF16 ``F.linear`` at runtime instead of
Triton kernels (including batched/grouped MoE paths).
"""

from __future__ import annotations

import functools
import logging
from typing import Callable

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_PATCHED = False


def is_blackwell_gpu() -> bool:
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability(0)[0] >= 10


def dequantize_fp8_tensor(
    quantized: torch.Tensor,
    scales: torch.Tensor,
    block_size: tuple[int, int] | None,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize via ``transformers`` Fp8Dequantize (handles block + MXFP4 layouts)."""
    if quantized.element_size() > 1:
        return quantized.to(out_dtype)

    from transformers.integrations.finegrained_fp8 import Fp8Dequantize

    try:
        dequant = Fp8Dequantize(hf_quantizer=None)._dequantize_one(quantized, scales)
        return dequant.to(out_dtype)
    except ValueError as exc:
        logger.warning(
            "FP8 block dequant failed for shape %s (scale %s): %s; using cast fallback",
            tuple(quantized.shape),
            tuple(scales.shape),
            exc,
        )
        return quantized.to(out_dtype)


def _bf16_w8a8_fp8_matmul(A, B, As, Bs, block_size, output_dtype):
    del As  # dynamic activation: use A in compute dtype
    weight = dequantize_fp8_tensor(B, Bs, block_size, output_dtype)
    return F.linear(A.to(output_dtype), weight, None)


def _bf16_fp8_act_quant(input: torch.Tensor, block_size: int):
    """Passthrough activation quant: use BF16 activations with unit scale."""
    del block_size
    scale = torch.ones((), device=input.device, dtype=torch.float32)
    return input, scale


def _bf16_batched_fp8_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    *,
    block_size: tuple[int, int] | None,
    expert_ids: torch.Tensor,
) -> torch.Tensor:
    out_dim = B.shape[1]
    out = torch.empty(A.shape[0], out_dim, dtype=A.dtype, device=A.device)
    for expert_idx in expert_ids.unique():
        mask = expert_ids == expert_idx
        if not mask.any():
            continue
        eid = int(expert_idx.item())
        weight = dequantize_fp8_tensor(B[eid], Bs[eid], block_size, A.dtype)
        out[mask] = F.linear(A[mask], weight)
    return out


def _bf16_grouped_fp8_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    *,
    tokens_per_expert: torch.Tensor,
    block_size: tuple[int, int] | None,
    offsets: torch.Tensor,
) -> torch.Tensor:
    out_dim = B.shape[1]
    out = torch.empty(A.shape[0], out_dim, dtype=A.dtype, device=A.device)
    start = 0
    for expert_idx, count in enumerate(tokens_per_expert.tolist()):
        count = int(count)
        if count == 0:
            continue
        end = start + count
        weight = dequantize_fp8_tensor(B[expert_idx], Bs[expert_idx], block_size, A.dtype)
        out[start:end] = F.linear(A[start:end], weight)
        start = end
    return out


@functools.cache
def _blackwell_finegrained_fp8_kernel():
    from transformers.integrations.finegrained_fp8 import FineGrainedFP8

    return FineGrainedFP8(
        fp8_matmul=_bf16_w8a8_fp8_matmul,
        fp8_act_quant=_bf16_fp8_act_quant,
        batched_fp8_matmul=_bf16_batched_fp8_matmul,
        grouped_fp8_matmul=_bf16_grouped_fp8_matmul,
    )


def apply_glm_fp8_blackwell_fallback() -> None:
    """Monkeypatch transformers FP8 paths to BF16 matmul on Blackwell GPUs."""
    global _PATCHED
    if _PATCHED or not is_blackwell_gpu():
        return

    import transformers.integrations.finegrained_fp8 as fg

    _orig_load_kernel: Callable[[], object] | None = getattr(
        fg, "_load_finegrained_fp8_kernel", None
    )

    def _bf16_fp8_linear_forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.weight.element_size() > 1:
            return F.linear(input, self.weight, self.bias)
        weight = self.weight
        scale_inv = self.weight_scale_inv
        if hasattr(weight, "to_local"):
            weight = weight.to_local()
            scale_inv = scale_inv.to_local()
        weight_bf16 = dequantize_fp8_tensor(
            weight, scale_inv, self.block_size, input.dtype
        )
        return F.linear(input, weight_bf16, self.bias)

    def _bf16_expert_linear(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        weight_scale_inv: torch.Tensor,
        activation_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del activation_scale
        if weight.element_size() > 1:
            return F.linear(input, weight, None)
        weight_bf16 = dequantize_fp8_tensor(
            weight, weight_scale_inv, self.block_size, input.dtype
        )
        return F.linear(input, weight_bf16, None)

    def _load_finegrained_fp8_kernel_blackwell():
        return _blackwell_finegrained_fp8_kernel()

    fg.w8a8_fp8_matmul = _bf16_w8a8_fp8_matmul
    fg.FP8Linear.forward = _bf16_fp8_linear_forward
    if _orig_load_kernel is not None:
        fg._load_finegrained_fp8_kernel = _load_finegrained_fp8_kernel_blackwell
        if hasattr(_orig_load_kernel, "cache_clear"):
            _orig_load_kernel.cache_clear()

    if hasattr(fg, "FP8Experts"):
        fg.FP8Experts.linear = _bf16_expert_linear

    _PATCHED = True
    logger.info(
        "Applied GLM FP8 Blackwell fallback (BF16 matmul; finegrained-fp8 kernels skipped)"
    )
