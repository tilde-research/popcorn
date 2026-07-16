from itertools import pairwise
from typing import Literal

import torch
import torch.nn.functional as F
from jaxtyping import BFloat16, Float, Float16, Int32
from torch import Tensor

from popcorn import Range, kernel, register_kernel
from popcorn.kernels._varlen import cuts


@register_kernel(
    test_shapes={"total": Range(8, 2048), "boundaries": Range(2, 9), "heads": {4}, "kv_heads": {2}, "head_dim": {64}},
    test_args={"softmax_scale": [None, 0.25]},
    test_inputs={"cu_seqlens": cuts},
)
def attn_varlen(
    q: Float[Tensor, "total heads head_dim"],
    k: Float[Tensor, "total kv_heads head_dim"],
    v: Float[Tensor, "total kv_heads head_dim"],
    cu_seqlens: Int32[Tensor, "boundaries"],
    causal: bool = False,
    softmax_scale: float | None = None,
) -> Float[Tensor, "total heads head_dim"]:
    """Attention over sequences packed along `total`, delimited by `cu_seqlens`
    offsets (flash-attention varlen layout). Grouped kv heads as in `attn`."""
    return torch.cat(
        [
            F.scaled_dot_product_attention(
                q[lo:hi].transpose(0, 1),
                k[lo:hi].transpose(0, 1),
                v[lo:hi].transpose(0, 1),
                scale=softmax_scale,
                is_causal=causal,
                enable_gqa=True,
            ).transpose(0, 1)
            for lo, hi in pairwise(cu_seqlens.tolist())
        ]
    )


@attn_varlen.register("fa3", source="flash_attn_interface.flash_attn_varlen_func")
def attn_varlen_fa3(
    q: Float16[Tensor, "total heads head_dim"] | BFloat16[Tensor, "total heads head_dim"],
    k,
    v,
    cu_seqlens,
    causal,
    softmax_scale,
):
    max_seqlen = int(cu_seqlens.diff().max())
    return kernel(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        softmax_scale=softmax_scale,
        causal=causal,
    )


# 16-bit only, causal only: same triton kernel as `attn`, batch folded to 1.
@attn_varlen.register("fla", source="fla.ops.attn.parallel_attn")
def attn_varlen_fla(
    q: Float16[Tensor, "total heads head_dim"] | BFloat16[Tensor, "total heads head_dim"],
    k,
    v,
    cu_seqlens,
    causal: Literal[True],
    softmax_scale,
):
    return kernel(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), scale=softmax_scale, cu_seqlens=cu_seqlens).squeeze(0)
