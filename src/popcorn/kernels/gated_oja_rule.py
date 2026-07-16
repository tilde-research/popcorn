import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from popcorn import Range, kernel, register_kernel
from popcorn.kernels._utils import default_scale, upcast


@register_kernel(
    test_shapes={"batch": Range(1, 8), "seq": Range(2, 128), "heads": {4}, "key_dim": {64}, "value_dim": {64}},
    test_args={"softmax_scale": [None, 0.25]},
    test_inputs={
        # v feeds back into its own write (Oja's normalizing term assumes bounded
        # data); unnormalized draws blow the state up within ~64 steps.
        "k": lambda t: F.normalize(t, dim=-1),
        "v": lambda t: F.normalize(t, dim=-1),
        "beta": torch.sigmoid,
        "gv": F.logsigmoid,
    },
)
def gated_oja_rule(
    q: Float[Tensor, "batch seq heads key_dim"],
    k: Float[Tensor, "batch seq heads key_dim"],
    v: Float[Tensor, "batch seq heads value_dim"],
    gv: Float[Tensor, "batch seq heads value_dim"],
    beta: Float[Tensor, "batch seq heads"],
    softmax_scale: float | None = None,
) -> Float[Tensor, "batch seq heads value_dim"]:
    """Gated Oja's rule: the state decays per value channel through `gv`, and each
    step subtracts the state's readout of `v` from `k` before writing (Oja's
    normalizing term), keeping the memory bounded."""
    scale = default_scale(softmax_scale, q.shape[-1])
    q32, k32, v32, gv32, beta32 = map(upcast, (q, k, v, gv, beta))
    state = q32.new_zeros(q.shape[0], q.shape[2], q.shape[3], v.shape[3])
    outs = []
    for t in range(q.shape[1]):
        state = state * gv32[:, t].exp()[:, :, None, :]
        write = beta32[:, t, :, None] * (k32[:, t] - torch.einsum("bhkv,bhv->bhk", state, v32[:, t]))
        state = state + write[..., None] * v32[:, t][..., None, :]
        outs.append(torch.einsum("bhkv,bhk->bhv", state, q32[:, t] * scale))
    return torch.stack(outs, 1).to(q.dtype)


@gated_oja_rule.register("fla", source="fla.ops.gated_oja_rule.chunk_gated_oja_rule")
def gated_oja_rule_fla(q, k, v, gv, beta, softmax_scale):
    return kernel(q, k, v, gv, beta, scale=softmax_scale)[0]
