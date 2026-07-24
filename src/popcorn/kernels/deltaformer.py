import importlib.util

import torch
from jaxtyping import Float
from torch import Tensor

from popcorn import Tag, kernel, register_kernel
from popcorn.kernels._utils import upcast


@register_kernel(
    test_inputs={"beta": torch.sigmoid},
    tags={Tag.SEQUENCE_MIXER, Tag.ATTENTION},
)
def deltaformer(
    q: Float[Tensor, "batch seq heads head_dim"],
    k: Float[Tensor, "batch seq heads head_dim"],
    v: Float[Tensor, "batch seq heads head_dim"],
    beta: Float[Tensor, "batch seq heads"] | None = None,
) -> Float[Tensor, "batch seq heads head_dim"]:
    r"""Values corrected by a strictly causal delta rule in attention space, then attended normally.

    $$u_t = v_t - \beta_t \sum_{s < t} \tilde{p}_{ts} u_s, \qquad y = \operatorname{softmax}(c\, q k^\top) \, u$$

    [DeltaFormer (Zhong et al., 2025)](https://arxiv.org/abs/2505.19488)
    """
    scale = q.shape[-1] ** -0.5
    q32, k32, v32 = (upcast(t).transpose(1, 2) for t in (q, k, v))
    scores = q32 @ k32.transpose(-1, -2) * scale
    seq = q.shape[1]
    lower = torch.ones(seq, seq, dtype=torch.bool, device=q.device).tril()
    strict = scores.masked_fill(~lower.tril(-1), -torch.inf).softmax(-1)
    strict = strict.nan_to_num()  # the first row has no visible keys
    beta32 = torch.ones_like(q32[..., 0]) if beta is None else upcast(beta).transpose(1, 2)
    u = torch.zeros_like(v32)
    for t in range(seq):
        correction = torch.einsum("bhj,bhjd->bhd", strict[:, :, t, :t], u[:, :, :t]) if t else 0
        u[:, :, t] = v32[:, :, t] - beta32[:, :, t, None] * correction
    return (scores.masked_fill(~lower, -torch.inf).softmax(-1) @ u).transpose(1, 2).to(q.dtype)


def _has_flash_attn(**_):
    """fla's deltaformer runs its second stage through flash-attn 2."""
    return importlib.util.find_spec("flash_attn") is not None


@deltaformer.register("fla", source="fla.ops.deltaformer.deltaformer_attn", predicate=_has_flash_attn)
def deltaformer_fla(q, k, v, beta):
    return kernel(q, k, v, beta)
