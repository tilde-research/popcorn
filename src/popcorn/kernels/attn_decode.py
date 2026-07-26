from itertools import pairwise

import torch
from jaxtyping import BFloat16, Float, Int32
from torch import Tensor

from popcorn import Tag, kernel, register_kernel
from popcorn.kernels._utils import default_scale, upcast
from popcorn.kernels._varlen import cuts


@register_kernel(
    test_args={"softmax_scale": [None, 0.25]},
    test_inputs={"cu_seqlens": cuts},
    tags={Tag.SEQUENCE_MIXER, Tag.ATTENTION, Tag.VARLEN},
)
def attn_decode(
    q: Float[Tensor, "batch q_heads key_dim"],
    k: Float[Tensor, "total kv_heads key_dim"],
    v: Float[Tensor, "total kv_heads value_dim"],
    cu_seqlens: Int32[Tensor, "batch+1"],
    softmax_scale: float | None = None,
) -> Float[Tensor, "batch q_heads value_dim"]:
    r"""One-query attention over each sequence in a packed key-value cache.

    $$y_s = \operatorname{softmax}\!\left(c \, q_s K_s^\top\right) V_s$$

    [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
    """
    scale = default_scale(softmax_scale, q.shape[-1])
    groups = q.shape[1] // k.shape[1]
    k32, v32 = (upcast(t).repeat_interleave(groups, 1) for t in (k, v))
    outputs = []
    for query, (lo, hi) in zip(upcast(q), pairwise(cu_seqlens.tolist())):
        scores = torch.einsum("hd,lhd->hl", query * scale, k32[lo:hi])
        outputs.append(torch.einsum("hl,lhv->hv", scores.softmax(-1), v32[lo:hi]))
    return torch.stack(outputs).to(q.dtype)


# bfloat16 only: float16 output error exceeds the harness budget across 25 of
# 56 tested cases on H100, while bfloat16 passes the full grid.
@attn_decode.register(
    "fla",
    source="fla.ops.attn.decoding.attn_decoding_one_step",
    supports={"key_dim": {64}, "value_dim": {64}, "q_heads": {4}, "kv_heads": {2}},
    forward_only=True,
)
def attn_decode_fla(
    q: BFloat16[Tensor, "batch q_heads key_dim"],
    k,
    v,
    cu_seqlens,
    softmax_scale,
):
    return kernel(
        q.unsqueeze(0),
        k.unsqueeze(0),
        v.unsqueeze(0),
        scale=softmax_scale,
        cu_seqlens=cu_seqlens,
    ).squeeze(0)
