import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from popcorn import Tag, kernel, register_kernel
from popcorn.kernels._utils import default_scale, upcast


@register_kernel(
    test_args={"softmax_scale": [None, 0.25]},
    test_inputs={"g": F.logsigmoid},
    tags={Tag.SEQUENCE_MIXER, Tag.LINEAR_ATTENTION},
)
def simple_gla(
    q: Float[Tensor, "batch seq heads key_dim"],
    k: Float[Tensor, "batch seq heads key_dim"],
    v: Float[Tensor, "batch seq heads value_dim"],
    g: Float[Tensor, "batch seq heads"],
    softmax_scale: float | None = None,
) -> Float[Tensor, "batch seq heads value_dim"]:
    r"""Gated linear attention with a scalar per-head log forget gate.

    $$S_t = e^{g_t} S_{t-1} + k_t v_t^\top, \qquad o_t = c \, q_t^\top S_t$$

    [Gated Linear Attention (Yang et al., 2023)](https://arxiv.org/abs/2312.06635)
    """
    scale = default_scale(softmax_scale, q.shape[-1])
    decay = upcast(g).cumsum(1).transpose(1, 2)
    decay = (decay[..., :, None] - decay[..., None, :]).tril().exp().tril()
    scores = torch.einsum("bqhk,bjhk->bhqj", upcast(q) * scale, upcast(k))
    return torch.einsum("bhqj,bjhv->bqhv", scores * decay, upcast(v)).to(q.dtype)


@simple_gla.register("fla", source="fla.ops.simple_gla.chunk_simple_gla")
def simple_gla_fla(q, k, v, g, softmax_scale):
    return kernel(q, k, v, g, scale=softmax_scale)[0]
