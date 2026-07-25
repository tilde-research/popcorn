import torch
import torch.nn.functional as F
from jaxtyping import BFloat16, Float
from torch import Tensor

from popcorn import Tag, kernel, register_kernel
from popcorn.kernels._utils import default_scale, upcast


@register_kernel(
    test_args={"softmax_scale": [None, 0.25]},
    test_inputs={"g": F.logsigmoid},
    tags={Tag.SEQUENCE_MIXER, Tag.ATTENTION},
)
def forgetting_attn(
    q: Float[Tensor, "batch seq q_heads head_dim"],
    k: Float[Tensor, "batch seq kv_heads head_dim"],
    v: Float[Tensor, "batch seq kv_heads head_dim"],
    g: Float[Tensor, "batch seq q_heads"],
    softmax_scale: float | None = None,
) -> Float[Tensor, "batch seq q_heads head_dim"]:
    r"""Causal softmax attention with a learned per-head log forget gate on the scores.

    $$s_{ij} = c \, q_i k_j^\top + D_i - D_j, \qquad D = \mathrm{cumsum}(g)$$

    [Forgetting Transformer (Lin et al., 2025)](https://arxiv.org/abs/2503.02130)
    """
    seq, scale = q.shape[1], default_scale(softmax_scale, q.shape[-1])
    groups = q.shape[2] // k.shape[2]
    k, v = (upcast(t).repeat_interleave(groups, 2) for t in (k, v))
    decay = upcast(g).cumsum(1).transpose(1, 2)
    scores = torch.einsum("bqhd,bkhd->bhqk", upcast(q) * scale, k)
    scores = scores + decay[..., :, None] - decay[..., None, :]
    mask = torch.ones(seq, seq, dtype=torch.bool, device=q.device).tril()
    scores = scores.masked_fill(~mask, -torch.inf)
    return torch.einsum("bhqk,bkhd->bqhd", scores.softmax(-1), v).to(q.dtype)


# bfloat16 only: fp32 runs the matmuls on tf32 cores and fp16 gate gradients
# land just past tolerance.
@forgetting_attn.register("fla", source="fla.ops.forgetting_attn.parallel_forgetting_attn")
def forgetting_attn_fla(q: BFloat16[Tensor, "batch seq q_heads head_dim"], k, v, g, softmax_scale):
    return kernel(q, k, v, g, scale=softmax_scale)
