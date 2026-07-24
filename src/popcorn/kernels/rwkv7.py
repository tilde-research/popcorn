import torch
import torch.nn.functional as F
from jaxtyping import BFloat16, Float, Float32
from torch import Tensor

from popcorn import Tag, kernel, register_kernel
from popcorn.kernels._utils import upcast


# `a` is the in-context erase direction (unit norm, negative) and `b` its
# replacement.
@register_kernel(
    test_args={"softmax_scale": [1.0, 0.25]},
    test_inputs={
        "w": F.logsigmoid,
        "a": lambda t: -F.normalize(t, dim=-1),
        "b": lambda t: 0.5 * F.normalize(t, dim=-1),
    },
    tags={Tag.SEQUENCE_MIXER, Tag.LINEAR_ATTENTION},
)
def rwkv7(
    r: Float[Tensor, "batch seq heads key_dim"],
    w: Float[Tensor, "batch seq heads key_dim"],
    k: Float[Tensor, "batch seq heads key_dim"],
    v: Float[Tensor, "batch seq heads value_dim"],
    a: Float[Tensor, "batch seq heads key_dim"],
    b: Float[Tensor, "batch seq heads key_dim"],
    softmax_scale: float = 1.0,
) -> Float[Tensor, "batch seq heads value_dim"]:
    r"""RWKV-7: a diagonal-plus-low-rank state transition with in-context erase and replace.

    $$S_t = \mathrm{diag}\!\big(e^{w_t}\big) S_{t-1} + b_t \big(S_{t-1}^\top a_t\big)^\top + k_t v_t^\top,
    \qquad o_t = c \, r_t^\top S_t$$

    [RWKV-7 "Goose" (Peng et al., 2025)](https://arxiv.org/abs/2503.14456)
    """
    r32, w32, k32, v32, a32, b32 = map(upcast, (r, w, k, v, a, b))
    state = r32.new_zeros(r.shape[0], r.shape[2], r.shape[3], v.shape[3])
    outs = []
    for t in range(r.shape[1]):
        removed = torch.einsum("bhkv,bhk->bhv", state, a32[:, t])
        state = state * w32[:, t].exp()[..., None] + b32[:, t, :, :, None] * removed[:, :, None, :]
        state = state + k32[:, t, :, :, None] * v32[:, t, :, None, :]
        outs.append(torch.einsum("bhkv,bhk->bhv", state, r32[:, t] * softmax_scale))
    return torch.stack(outs, 1).to(r.dtype)


# float16 value gradients land just past tolerance; fp32 and bf16 hold.
@rwkv7.register("fla", source="fla.ops.rwkv7.chunk_rwkv7")
def rwkv7_fla(
    r: Float32[Tensor, "batch seq heads key_dim"] | BFloat16[Tensor, "batch seq heads key_dim"],
    w,
    k,
    v,
    a,
    b,
    softmax_scale,
):
    return kernel(r, w, k, v, a, b, scale=softmax_scale)[0]
