import math

import torch
import torch.nn.functional as F
from jaxtyping import BFloat16, Float, Float32
from torch import Tensor

from popcorn import Tag, kernel, register_kernel
from popcorn.kernels._utils import upcast


# levels must equal ceil(log2(seq)) + 1; singleton test pools keep the grid
# consistent.
@register_kernel(
    test_inputs={"g": F.logsigmoid, "level_scales": torch.sigmoid}, tags={Tag.SEQUENCE_MIXER, Tag.LINEAR_ATTENTION}
)
def log_linear_attn(
    q: Float[Tensor, "batch seq heads key_dim"],
    k: Float[Tensor, "batch seq heads key_dim"],
    v: Float[Tensor, "batch seq heads value_dim"],
    g: Float[Tensor, "batch seq heads"],
    level_scales: Float[Tensor, "batch seq heads levels"],
) -> Float[Tensor, "batch seq heads value_dim"]:
    r"""Gated linear attention through a hierarchical mask, one learned scale per log2 level.

    $$o_t = \sum_{s \le t} \lambda_{t,\ell(t,s)} \, e^{D_t - D_s} \big(q_t^\top k_s\big) v_s,
    \qquad D = \mathrm{cumsum}(g)$$

    [Log-Linear Attention (Guo et al., 2025)](https://arxiv.org/abs/2506.04761)
    """
    seq = q.shape[1]
    lower = torch.ones(seq, seq, dtype=torch.bool, device=q.device).tril()
    decay = upcast(g).transpose(1, 2).cumsum(-1)
    A = (decay[..., :, None] - decay[..., None, :]).masked_fill(~lower, -torch.inf).exp()
    scales = upcast(level_scales).permute(0, 2, 3, 1)  # b h level t
    i = torch.arange(seq, device=q.device)[:, None]
    j = torch.arange(seq, device=q.device)[None, :]
    H = torch.diag_embed(scales[..., 0, :])
    for level in range(1, math.ceil(math.log2(seq)) + 1):
        half = 1 << (level - 1)
        anchor = i - i % half
        picked = (i % (2 * half) >= half) & (j + half >= anchor) & (j < anchor)
        H = H + torch.where(picked, scales[..., level, :, None], 0.0)
    M = torch.einsum("bhqj,bqhk,bjhk->bhqj", A * H, upcast(q), upcast(k))
    return torch.einsum("bhqj,bjhv->bqhv", M, upcast(v)).to(q.dtype)


# single head only: with more the kernel diverges from fla's own naive
# reference and the backward returns misshapen gradients; see ISSUES.md.
# float16 value gradients land just past tolerance; fp32 and bf16 hold.
@log_linear_attn.register("fla", source="fla.ops.log_linear_attn.chunk_log_linear_attn")
def log_linear_attn_fla(
    q: Float32[Tensor, "batch seq heads key_dim"] | BFloat16[Tensor, "batch seq heads key_dim"], k, v, g, level_scales
):
    return kernel(q, k, v, g, level_scales)[0]
