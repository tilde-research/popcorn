from typing import Literal

import torch.nn.functional as F
from jaxtyping import BFloat16, Float, Float16
from torch import Tensor

from popcorn import Tag, kernel, register_kernel


@register_kernel(
    test_args={"softmax_scale": [None, 0.25]},
    tags={Tag.SEQUENCE_MIXER, Tag.ATTENTION},
)
def attn(
    q: Float[Tensor, "batch seq q_heads head_dim"],
    k: Float[Tensor, "batch seq kv_heads head_dim"],
    v: Float[Tensor, "batch seq kv_heads head_dim"],
    causal: bool = False,
    softmax_scale: float | None = None,
) -> Float[Tensor, "batch seq q_heads head_dim"]:
    r"""Multi-head attention with grouped kv heads (q_heads must divide evenly).

    $$y = \operatorname{softmax}\!\left(\frac{q k^\top}{\sqrt{d}} + M\right) v$$

    [Neural Machine Translation by Jointly Learning to Align and Translate (Bahdanau et al., 2014)](https://arxiv.org/abs/1409.0473),
    [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762),
    [FlashAttention (Dao et al., 2022)](https://arxiv.org/abs/2205.14135),
    [GQA (Ainslie et al., 2023)](https://arxiv.org/abs/2305.13245)
    """
    return F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        scale=softmax_scale,
        is_causal=causal,
        enable_gqa=True,
    ).transpose(1, 2)


@attn.register("cudnn", source="cudnn.experimental.ops.scaled_dot_product_attention")
def attn_cudnn(
    q: Float16[Tensor, "batch seq q_heads head_dim"] | BFloat16[Tensor, "batch seq q_heads head_dim"],
    k: Float16[Tensor, "batch seq kv_heads head_dim"] | BFloat16[Tensor, "batch seq kv_heads head_dim"],
    v: Float16[Tensor, "batch seq kv_heads head_dim"] | BFloat16[Tensor, "batch seq kv_heads head_dim"],
    causal,
    softmax_scale,
):
    return kernel(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        scale=softmax_scale,
        is_causal=causal,
        enable_gqa=True,
    ).transpose(1, 2)


@attn.register("fa3", source="flash_attn_interface.flash_attn_func")
def attn_fa3(
    q: Float16[Tensor, "batch seq q_heads head_dim"] | BFloat16[Tensor, "batch seq q_heads head_dim"],
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
    q: Float16[Tensor, "batch seq q_heads head_dim"] | BFloat16[Tensor, "batch seq q_heads head_dim"],
    k,
    v,
    causal: Literal[True],
    softmax_scale,
):
    return kernel(q, k, v, scale=softmax_scale)
