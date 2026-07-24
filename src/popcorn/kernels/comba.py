import torch
import torch.nn.functional as F
from jaxtyping import BFloat16, Float, Float32
from torch import Tensor

from popcorn import Tag, kernel, register_kernel
from popcorn.kernels._utils import default_scale, upcast


# The correction reads the pre-decay state (fla's chunk kernel; their naive
# recurrence reads post-decay and disagrees, see ISSUES.md).
@register_kernel(
    test_args={"softmax_scale": [None, 0.25]},
    test_inputs={
        "k": lambda t: F.normalize(t, dim=-1),
        "p": lambda t: F.normalize(t, dim=-1),
        "beta": torch.sigmoid,
        "g": F.logsigmoid,
    },
    tags={Tag.SEQUENCE_MIXER, Tag.LINEAR_ATTENTION},
)
def comba(
    q: Float[Tensor, "batch seq heads key_dim"],
    k: Float[Tensor, "batch seq heads key_dim"],
    v: Float[Tensor, "batch seq heads value_dim"],
    p: Float[Tensor, "batch seq heads key_dim"],
    g: Float[Tensor, "batch seq heads"],
    beta: Float[Tensor, "batch seq heads"],
    softmax_scale: float | None = None,
) -> Float[Tensor, "batch seq heads value_dim"]:
    r"""A gated delta rule reading corrections through an auxiliary key p while writing through k.

    $$S_t = e^{g_t} S_{t-1} + \beta_t k_t \big(v_t - S_{t-1}^\top p_t\big)^\top,
    \qquad o_t = c \, q_t^\top S_t$$

    [Comba (Hu et al., 2025)](https://arxiv.org/abs/2506.02475)
    """
    scale = default_scale(softmax_scale, q.shape[-1])
    q32, k32, v32, p32, g32, beta32 = map(upcast, (q, k, v, p, g, beta))
    state = q32.new_zeros(q.shape[0], q.shape[2], q.shape[3], v.shape[3])
    outs = []
    for t in range(q.shape[1]):
        error = v32[:, t] - torch.einsum("bhkv,bhk->bhv", state, p32[:, t])
        state = state * g32[:, t].exp()[..., None, None]
        update = beta32[:, t, :, None] * error
        state = state + k32[:, t][..., None] * update[..., None, :]
        outs.append(torch.einsum("bhkv,bhk->bhv", state, q32[:, t] * scale))
    return torch.stack(outs, 1).to(q.dtype)


# fp16 gradients land just past tolerance (see ISSUES.md).
@comba.register("fla", source="fla.ops.comba.chunk_comba")
def comba_fla(
    q: Float32[Tensor, "batch seq heads key_dim"] | BFloat16[Tensor, "batch seq heads key_dim"],
    k,
    v,
    p,
    g,
    beta,
    softmax_scale,
):
    return kernel(q, k, v, p, g, beta=beta, scale=softmax_scale)[0]
