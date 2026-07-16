from typing import Literal

import torch.nn.functional as F
from jaxtyping import BFloat16, Float, Float16
from torch import Tensor

from popcorn import Range, kernel, register_kernel


@register_kernel(
    test_shapes={"batch": Range(1, 8), "seq": Range(2, 512), "heads": {4}, "kv_heads": {2}, "head_dim": {64}},
    test_args={"softmax_scale": [None, 0.25]},
)
def attn(
    q: Float[Tensor, "batch seq heads head_dim"],
    k: Float[Tensor, "batch seq kv_heads head_dim"],
    v: Float[Tensor, "batch seq kv_heads head_dim"],
    causal: bool = False,
    softmax_scale: float | None = None,
) -> Float[Tensor, "batch seq heads head_dim"]:
    """Multi-head attention with grouped kv heads (heads must divide evenly)."""
    return F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        scale=softmax_scale,
        is_causal=causal,
        enable_gqa=True,
    ).transpose(1, 2)


@attn.register("fa3", source="flash_attn_interface.flash_attn_func")
def attn_fa3(
    q: Float16[Tensor, "batch seq heads head_dim"] | BFloat16[Tensor, "batch seq heads head_dim"],
    k,
    v,
    causal,
    softmax_scale,
):
    return kernel(q, k, v, softmax_scale=softmax_scale, causal=causal)


# 16-bit only: the triton kernel computes fp32 matmuls with tf32 cores, which
# misses the float32 tolerance.
@attn.register("fla", source="fla.ops.attn.parallel_attn")
def attn_fla(
    q: Float16[Tensor, "batch seq heads head_dim"] | BFloat16[Tensor, "batch seq heads head_dim"],
    k,
    v,
    causal: Literal[True],
    softmax_scale,
):
    return kernel(q, k, v, scale=softmax_scale)
