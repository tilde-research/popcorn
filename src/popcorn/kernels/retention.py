import torch
from jaxtyping import Float
from torch import Tensor

from popcorn import Tag, kernel, register_kernel
from popcorn.kernels._utils import default_scale, upcast


@register_kernel(
    test_args={"softmax_scale": [None, 0.25]},
    tags={Tag.SEQUENCE_MIXER, Tag.LINEAR_ATTENTION},
)
def retention(
    q: Float[Tensor, "batch seq heads key_dim"],
    k: Float[Tensor, "batch seq heads key_dim"],
    v: Float[Tensor, "batch seq heads value_dim"],
    softmax_scale: float | None = None,
) -> Float[Tensor, "batch seq heads value_dim"]:
    r"""Multi-scale retention: causal linear attention with a fixed per-head exponential decay.

    $$S_t = \gamma_h S_{t-1} + k_t v_t^\top, \qquad o_t = c \, q_t^\top S_t,
    \qquad \gamma_h = 1 - 2^{-5-h}$$

    [Retentive Network (Sun et al., 2023)](https://arxiv.org/abs/2307.08621)
    """
    dtype = q.dtype
    q, k, v = map(upcast, (q, k, v))
    heads, seq = q.shape[2], q.shape[1]
    scale = default_scale(softmax_scale, q.shape[-1])
    gamma = 1 - torch.exp2(-5.0 - torch.arange(heads, device=q.device, dtype=q.dtype))
    n = torch.arange(seq, device=q.device, dtype=q.dtype)
    steps = n[:, None] - n[None, :]
    decay = torch.exp2(steps * gamma.log2()[:, None, None]) * (steps >= 0)
    scores = torch.einsum("bqhk,bjhk->bhqj", q * scale, k)
    return torch.einsum("bhqj,bjhv->bqhv", scores * decay, v).to(dtype)


@retention.register("fla", source="fla.ops.retention.chunk_retention")
def retention_fla(q, k, v, softmax_scale):
    return kernel(q, k, v, scale=softmax_scale)[0]


@retention.register("fla:recurrent", source="fla.ops.retention.fused_recurrent_retention", forward_only=True)
def retention_fla_recurrent(q, k, v, softmax_scale):
    return kernel(q, k, v, scale=softmax_scale)[0]
