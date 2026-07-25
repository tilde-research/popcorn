import torch
import torch.nn.functional as F
from jaxtyping import BFloat16, Float
from torch import Tensor

from popcorn import Tag, kernel, register_kernel
from popcorn.kernels._utils import default_scale, upcast


# k, v, and beta hold num_householder entries per step (fla's contract).
@register_kernel(
    test_args={"num_householder": [2], "softmax_scale": [None, 0.25]},
    test_inputs={"k": lambda t: F.normalize(t, dim=-1), "beta": torch.sigmoid, "g": F.logsigmoid},
    tags={Tag.SEQUENCE_MIXER, Tag.LINEAR_ATTENTION},
)
def gated_delta_product(
    q: Float[Tensor, "batch seq heads key_dim"],
    k: Float[Tensor, "batch seq*num_householder heads key_dim"],
    v: Float[Tensor, "batch seq*num_householder heads value_dim"],
    g: Float[Tensor, "batch seq heads"],
    beta: Float[Tensor, "batch seq*num_householder heads"],
    num_householder: int = 1,
    softmax_scale: float | None = None,
) -> Float[Tensor, "batch seq heads value_dim"]:
    r"""Gated DeltaProduct: several delta-rule writes per step under one per-head forget gate.

    $$S_t = e^{g_t} S_{t-1} \prod_{j=1}^{H} \big(I - \beta_{t,j} k_{t,j} k_{t,j}^\top\big)
    + \text{writes}, \qquad o_t = c \, q_t^\top S_t$$

    [DeltaProduct (Siems et al., 2025)](https://arxiv.org/abs/2502.10297)
    """
    scale = default_scale(softmax_scale, q.shape[-1])
    q32, k32, v32, g32, beta32 = map(upcast, (q, k, v, g, beta))
    state = q32.new_zeros(q.shape[0], q.shape[2], q.shape[3], v.shape[3])
    outs = []
    for t in range(q.shape[1]):
        state = state * g32[:, t].exp()[..., None, None]
        for j in range(num_householder):
            i = t * num_householder + j
            error = v32[:, i] - torch.einsum("bhkv,bhk->bhv", state, k32[:, i])
            update = beta32[:, i, :, None] * error
            state = state + k32[:, i][..., None] * update[..., None, :]
        outs.append(torch.einsum("bhkv,bhk->bhv", state, q32[:, t] * scale))
    return torch.stack(outs, 1).to(q.dtype)


# bfloat16 only: the kernel asserts against float32 and fp16 outputs land just
# past tolerance (see ISSUES.md).
@gated_delta_product.register("fla", source="fla.ops.gated_delta_product.chunk_gated_delta_product")
def gated_delta_product_fla(
    q: BFloat16[Tensor, "batch seq heads key_dim"],
    k,
    v,
    g,
    beta,
    num_householder,
    softmax_scale,
):
    return kernel(q, k, v, g, beta, num_householder=num_householder, scale=softmax_scale)[0]
