from typing import Literal

import torch
from jaxtyping import BFloat16, Float, Float32
from torch import Tensor

from popcorn import Range, kernel, register_kernel
from popcorn.kernels._utils import default_scale, upcast


@register_kernel(
    test_shapes={"batch": Range(1, 8), "seq": Range(2, 128), "heads": {4}, "key_dim": {64}, "value_dim": {64}, "slots": {16}},
    test_args={"softmax_scale": [None, 0.25]},
)
def abc(
    q: Float[Tensor, "batch seq heads key_dim"],
    k: Float[Tensor, "batch seq heads key_dim"],
    v: Float[Tensor, "batch seq heads value_dim"],
    s: Float[Tensor, "batch seq heads slots"],
    softmax_scale: float | None = None,
) -> Float[Tensor, "batch seq heads value_dim"]:
    """Attention with bounded-memory control (arXiv:2110.02488): slot scores `s`
    are normalized online into forget gates, then read through a softmax."""
    scale = default_scale(softmax_scale, q.shape[-1])
    q32, k32, v32, s32 = map(upcast, (q, k, v, s))
    z = s32.logcumsumexp(1)
    gate = (torch.cat((z[:, :1], z[:, :-1]), 1) - z).exp()
    s32 = (s32 - z).exp()

    hk = q32.new_zeros(q.shape[0], q.shape[2], q.shape[3], s.shape[3])
    ok = []
    for t in range(q.shape[1]):
        hk = hk * gate[:, t, :, None, :] + k32[:, t, :, :, None] * s32[:, t, :, None, :]
        ok.append(torch.einsum("bhkm,bhk->bhm", hk, q32[:, t] * scale))
    qv = torch.stack(ok, 1).softmax(-1)

    hv = q32.new_zeros(q.shape[0], q.shape[2], s.shape[3], v.shape[3])
    ov = []
    for t in range(q.shape[1]):
        hv = hv * gate[:, t, :, :, None] + s32[:, t, :, :, None] * v32[:, t, :, None, :]
        ov.append(torch.einsum("bhmv,bhm->bhv", hv, qv[:, t]))
    return torch.stack(ov, 1).to(q.dtype)


# float16 slot gradients land just past tolerance; bf16 and fp32 hold.
@abc.register("fla", source="fla.ops.abc.chunk_abc")
def abc_fla(
    q: Float32[Tensor, "batch seq heads key_dim"] | BFloat16[Tensor, "batch seq heads key_dim"],
    k,
    v,
    s,
    softmax_scale: Literal[None],
):
    return kernel(q, k, v, s)[0]
