import importlib.util

import torch
from jaxtyping import Float
from torch import Tensor

from popcorn import Range, kernel, register_kernel
from popcorn.kernels._utils import upcast


@register_kernel(
    test_shapes={"batch": Range(1, 8), "seq": Range(2, 128), "heads": {4}, "head_dim": {64}},
    test_inputs={"beta": torch.sigmoid},
)
def deltaformer(
    q: Float[Tensor, "batch seq heads head_dim"],
    k: Float[Tensor, "batch seq heads head_dim"],
    v: Float[Tensor, "batch seq heads head_dim"],
    beta: Float[Tensor, "batch seq heads"] | None = None,
) -> Float[Tensor, "batch seq heads head_dim"]:
    """DeltaFormer (arXiv:2505.19488): values are first corrected by a strictly
    causal delta rule in softmax-attention space, then attended normally."""
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
