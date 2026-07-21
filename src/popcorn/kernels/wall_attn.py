import functools
import importlib.util

import torch
import torch.nn.functional as F
from jaxtyping import Float, Float32
from torch import Tensor

from popcorn import Range, Tag, kernel, register_kernel
from popcorn.kernels._utils import default_scale, upcast

RCP_LN2 = 1.4426950408889634


@functools.cache
def _wall_deps(**_):
    return all(importlib.util.find_spec(module) for module in ("fla", "einops", "triton"))


# `g`/`g_scalar` are log-decay gates: the kernel's per-block reference frame
# assumes the prefix `P = cumsum(g)` is monotone non-increasing (as FoX/forgetting
# attention), so the gates are seeded in the log-sigmoid (<= 0) domain. A mixed-sign
# `g` drives the intra-block `exp2` far past the kernel's headroom and the kernel
# diverges from the reference by ~10-40% (its own upstream tests only cover this
# decay domain); `sink_bias` is a small per-head sink logit.
@register_kernel(
    test_shapes={"batch": Range(1, 8), "seq": Range(2, 512), "heads": {4}, "kv_heads": {2}, "head_dim": {64}},
    test_args={"softmax_scale": [None, 0.25], "window_size": [None, 128]},
    test_inputs={"g": F.logsigmoid, "g_scalar": F.logsigmoid, "sink_bias": lambda t: t * 0.1},
    tags={Tag.SEQUENCE_MIXER, Tag.ATTENTION},
)
def wall_attn(
    q: Float[Tensor, "batch seq heads head_dim"],
    k: Float[Tensor, "batch seq kv_heads head_dim"],
    v: Float[Tensor, "batch seq kv_heads head_dim"],
    g: Float[Tensor, "batch seq heads head_dim"],
    g_scalar: Float[Tensor, "batch seq heads"] | None = None,
    sink_bias: Float[Tensor, "heads"] | None = None,
    softmax_scale: float | None = None,
    window_size: int | None = None,
) -> Float[Tensor, "batch seq heads head_dim"]:
    r"""Causal attention with per-channel decay on every logit, optional scalar gate and sink.

    $$s_{ij} = c \sum_{d} q_{id} \, k_{jd} \, 2^{P_{id} - P_{jd}}, \qquad P = \mathrm{cumsum}(g)$$

    [Wall Attention](https://github.com/tilde-research/wall-attention-release),
    [Forgetting Transformer (Lin et al., 2025)](https://arxiv.org/abs/2503.02130)
    """
    scale = default_scale(softmax_scale, q.shape[-1]) * RCP_LN2
    q32, k32, v32, g32 = map(upcast, (q, k, v, g))
    groups = q.shape[2] // k.shape[2]
    prefix = g32.cumsum(1) * RCP_LN2
    q_til = q32 * torch.exp2(prefix)
    k_til = k32.repeat_interleave(groups, 2) * torch.exp2(-prefix)
    scores = torch.einsum("bihc,bjhc->bhij", q_til, k_til) * scale

    seq = q.shape[1]
    idx = torch.arange(seq, device=q.device)
    valid = idx[:, None] >= idx[None, :]
    if window_size is not None:
        valid = valid & (idx[:, None] - idx[None, :] < window_size)
    scores = scores.masked_fill(~valid, float("-inf"))
    if g_scalar is not None:
        c = upcast(g_scalar).cumsum(1).permute(0, 2, 1) * RCP_LN2
        scores = scores + c[..., :, None] - c[..., None, :]

    m = scores.amax(-1, keepdim=True)
    m = torch.where(torch.isfinite(m), m, 0.0)
    p = torch.exp2(scores - m)
    denom = p.sum(-1)
    if sink_bias is not None:
        denom = denom + torch.exp2(upcast(sink_bias)[:, None] * RCP_LN2 - m.squeeze(-1))
    weights = p / denom[..., None]
    out = torch.einsum("bhij,bjhc->bihc", weights, v32.repeat_interleave(groups, 2))
    return out.to(q.dtype)


# fp32 only: in 16-bit the structurally-derived gate gradients (`g`, `g_scalar`)
# land ~1.1-1.7x past the dtype floor on short sequences, while q/k/v/output stay
# clean (see ISSUES.md). fp32 is exact here because popcorn pins
# `TRITON_F32_DEFAULT=ieee`, so the kernel's fp32 matmuls avoid tf32.
@wall_attn.register("popcorn", source="popcorn.impls.wall_attn_tl.wall_attn", predicate=_wall_deps)
def wall_attn_popcorn(
    q: Float32[Tensor, "batch seq heads head_dim"],
    k,
    v,
    g,
    g_scalar,
    sink_bias,
    softmax_scale,
    window_size,
):
    return kernel(q, k, v, g, g_scalar, sink_bias, softmax_scale, window_size)
