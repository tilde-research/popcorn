import torch
import torch.nn.functional as F
from jaxtyping import BFloat16, Float, Float32
from torch import Tensor

from popcorn import Range, kernel, register_kernel
from popcorn.kernels._utils import upcast


@register_kernel(
    test_shapes={"batch": Range(1, 8), "seq": Range(2, 128), "heads": {4}, "key_dim": {64}, "value_dim": {64}},
    test_args={"softmax_scale": [1.0, 0.25]},
    test_inputs={
        "w": F.logsigmoid,
        "a": lambda t: -F.normalize(t, dim=-1),
        "b": lambda t: 0.5 * F.normalize(t, dim=-1),
    },
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
    """RWKV-7 (arXiv:2503.14456): S_t = S_{t-1} diag(exp(w_t)) + S_{t-1} a_t b_t^T + k_t v_t^T,
    the diagonal-plus-low-rank transition; `a` is the in-context erase
    direction (unit norm, negative) and `b` its replacement."""
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
