import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from popcorn import Range, kernel, register_kernel
from popcorn.kernels._utils import default_scale, upcast


@register_kernel(
    test_shapes={"batch": Range(1, 8), "seq": Range(2, 128), "heads": {4}, "key_dim": {64}, "value_dim": {64}},
    test_args={"softmax_scale": [None, 0.25]},
    test_inputs={"w": F.logsigmoid},
)
def rwkv6(
    r: Float[Tensor, "batch seq heads key_dim"],
    k: Float[Tensor, "batch seq heads key_dim"],
    v: Float[Tensor, "batch seq heads value_dim"],
    w: Float[Tensor, "batch seq heads key_dim"],
    u: Float[Tensor, "heads key_dim"],
    softmax_scale: float | None = None,
) -> Float[Tensor, "batch seq heads value_dim"]:
    """RWKV-6 (arXiv:2404.05892): linear attention with data-dependent per-key
    log decay `w` and a bonus weight `u` for the current token."""
    scale = default_scale(softmax_scale, r.shape[-1])
    r32, k32, v32, w32, u32 = map(upcast, (r, k, v, w, u))
    state = r32.new_zeros(r.shape[0], r.shape[2], r.shape[3], v.shape[3])
    outs = []
    for t in range(r.shape[1]):
        kv = k32[:, t, :, :, None] * v32[:, t, :, None, :]
        boosted = state + u32[None, :, :, None] * kv
        outs.append(torch.einsum("bhkv,bhk->bhv", boosted, r32[:, t] * scale))
        state = state * w32[:, t].exp()[..., None] + kv
    return torch.stack(outs, 1).to(r.dtype)


@rwkv6.register("fla", source="fla.ops.rwkv6.chunk_rwkv6")
def rwkv6_fla(r, k, v, w, u, softmax_scale):
    return kernel(r, k, v, w, u, scale=softmax_scale)[0]
