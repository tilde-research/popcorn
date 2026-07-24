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
def gla(
    q: Float[Tensor, "batch seq heads key_dim"],
    k: Float[Tensor, "batch seq heads key_dim"],
    v: Float[Tensor, "batch seq heads value_dim"],
    g: Float[Tensor, "batch seq heads key_dim"],
    softmax_scale: float | None = None,
) -> Float[Tensor, "batch seq heads value_dim"]:
    r"""Gated linear attention with a per-key-dimension log forget gate.

    $$S_t = \mathrm{diag}\!\big(e^{g_t}\big) \, S_{t-1} + k_t v_t^\top, \qquad o_t = c \, q_t^\top S_t$$

    [Gated Linear Attention (Yang et al., 2023)](https://arxiv.org/abs/2312.06635)
    """
    scale = default_scale(softmax_scale, q.shape[-1])
    q32, k32, v32, gate = map(upcast, (q, k, v, g))
    state = q32.new_zeros(q.shape[0], q.shape[2], q.shape[3], v.shape[3])
    outs = []
    for t in range(q.shape[1]):
        state = state * gate[:, t].exp()[..., None] + k32[:, t, :, :, None] * v32[:, t, :, None, :]
        outs.append(torch.einsum("bhkv,bhk->bhv", state, q32[:, t] * scale))
    return torch.stack(outs, 1).to(q.dtype)


@gla.register("fla", source="fla.ops.gla.chunk_gla")
def gla_fla(q, k, v, g, softmax_scale):
    return kernel(q, k, v, g, scale=softmax_scale)[0]
