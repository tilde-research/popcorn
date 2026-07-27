import torch
import torch.nn.functional as F
from jaxtyping import BFloat16, Float, Float16
from torch import Tensor

from popcorn import Tag, kernel, register_kernel
from popcorn.kernels._utils import default_scale, upcast


@register_kernel(
    test_args={"softmax_scale": [None, 0.25]},
    test_inputs={"k": lambda t: F.normalize(t, dim=-1), "beta": torch.sigmoid, "g": F.logsigmoid},
    tags={Tag.SEQUENCE_MIXER, Tag.LINEAR_ATTENTION},
)
def kda(
    q: Float[Tensor, "batch seq heads key_dim"],
    k: Float[Tensor, "batch seq heads key_dim"],
    v: Float[Tensor, "batch seq heads value_dim"],
    g: Float[Tensor, "batch seq heads key_dim"],
    beta: Float[Tensor, "batch seq heads"],
    softmax_scale: float | None = None,
) -> Float[Tensor, "batch seq heads value_dim"]:
    r"""Kimi Delta Attention: the delta rule with a per-key-dimension log forget gate.

    $$S_t = \mathrm{diag}\!\big(e^{g_t}\big) S_{t-1}
    + \beta_t k_t \big(v_t - S_{t-1}^\top k_t\big)^\top, \qquad o_t = c \, q_t^\top S_t$$

    [Kimi Linear (Kimi Team, 2025)](https://arxiv.org/abs/2510.26692)
    """
    scale = default_scale(softmax_scale, q.shape[-1])
    q32, k32, v32, g32, beta32 = map(upcast, (q, k, v, g, beta))
    state = q32.new_zeros(q.shape[0], q.shape[2], q.shape[3], v.shape[3])
    outs = []
    for t in range(q.shape[1]):
        k_t = k32[:, t]
        state = state * g32[:, t].exp()[..., None]
        error = v32[:, t] - torch.einsum("bhkv,bhk->bhv", state, k_t)
        update = beta32[:, t, :, None] * k_t
        state = state + update[..., None] * error[..., None, :]
        outs.append(torch.einsum("bhkv,bhk->bhv", state, q32[:, t] * scale))
    return torch.stack(outs, 1).to(q.dtype)


# 16-bit only: the chunked fp32 accumulation lands just past the harness
# floor even with ieee matmuls.
@kda.register("fla", source="fla.ops.kda.chunk_kda")
def kda_fla(
    q: Float16[Tensor, "batch seq heads key_dim"] | BFloat16[Tensor, "batch seq heads key_dim"],
    k,
    v,
    g,
    beta,
    softmax_scale,
):
    return kernel(q, k, v, g, beta, scale=softmax_scale)[0]


@kda.register("fla:recurrent", source="fla.ops.kda.fused_recurrent_kda", forward_only=True)
def kda_fla_recurrent(q, k, v, g, beta, softmax_scale):
    return kernel(q, k, v, g, beta, scale=softmax_scale)[0]
