import torch
import torch.nn.functional as F
from jaxtyping import BFloat16, Float, Float32
from torch import Tensor

from popcorn import Tag, kernel, register_kernel
from popcorn.kernels._utils import default_scale, upcast


def _fla_supported(q, k, v, block_size, softmax_scale):
    del v, softmax_scale
    if block_size < 16 or block_size & (block_size - 1):
        return False
    groups = q.shape[2] // k.shape[2]
    power_of_two = groups > 0 and groups & (groups - 1) == 0
    blocks = -(-q.shape[1] // block_size)
    return q.shape[2] % k.shape[2] == 0 and groups >= 16 and power_of_two and k.shape[1] == blocks


# A compressed block becomes visible at the token that completes it. Rows
# before the first complete block return zero output and zero log-sum-exp.
@register_kernel(
    test_args={"block_size": [32], "softmax_scale": [None, 0.25]},
    test_inputs={"k": lambda t: F.normalize(t, dim=-1)},
    tags={Tag.SEQUENCE_MIXER, Tag.ATTENTION},
)
def nsa_compression(
    q: Float[Tensor, "batch seq q_heads key_dim"],
    k: Float[Tensor, "batch blocks kv_heads key_dim"],
    v: Float[Tensor, "batch blocks kv_heads value_dim"],
    block_size: int = 64,
    softmax_scale: float | None = None,
) -> tuple[
    Float[Tensor, "batch seq q_heads value_dim"],
    Float32[Tensor, "batch seq q_heads"],
]:
    r"""Causal NSA attention over pre-compressed kv blocks; returns output and log-sum-exp.

    $$y_t = \sum_{j \,:\, (j+1) B \,\le\, t+1} p_{tj} \, v_j,
    \qquad p_t = \operatorname{softmax}\big(c \, q_t k^\top\big)$$

    [Native Sparse Attention (Yuan et al., 2025)](https://arxiv.org/abs/2502.11089)
    """
    scale = default_scale(softmax_scale, k.shape[-1])
    q32, k32, v32 = map(upcast, (q, k, v))
    groups = q.shape[2] // k.shape[2]
    k32 = k32.repeat_interleave(groups, 2)
    v32 = v32.repeat_interleave(groups, 2)

    token = torch.arange(q.shape[1], device=q.device)[:, None]
    block = torch.arange(k.shape[1], device=q.device)[None, :]
    visible = block < (token + 1) // block_size
    scores = torch.einsum("bthk,bjhk->bhtj", q32 * scale, k32).masked_fill(~visible, -torch.inf)
    peak = scores.amax(-1, keepdim=True)
    peak = torch.where(torch.isfinite(peak), peak, 0.0)
    weights = torch.exp(scores - peak)
    denom = weights.sum(-1, keepdim=True)
    out = torch.einsum("bhtj,bjhv->bthv", weights / denom.clamp_min(1e-30), v32)

    live = visible.any(-1)
    lse = peak.squeeze(-1) + denom.clamp_min(1e-30).log().squeeze(-1)
    lse = torch.where(live[None, None], lse, 0.0).transpose(1, 2).float()
    # FLA exposes lse but deliberately ignores its cotangent in backward.
    anchor = sum(t.reshape(-1)[0].float() for t in (q, k, v)) * 0
    return out.to(v.dtype).to(q.dtype), lse.detach() + anchor


@nsa_compression.register("fla", source="fla.ops.nsa.compression.parallel_nsa_compression", predicate=_fla_supported)
def nsa_compression_fla(
    q: Float32[Tensor, "batch seq q_heads key_dim"] | BFloat16[Tensor, "batch seq q_heads key_dim"],
    k,
    v,
    block_size,
    softmax_scale,
):
    return kernel(q, k, v, q.shape[1], block_size=block_size, scale=softmax_scale)
