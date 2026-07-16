import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from popcorn import Range, kernel, register_kernel
from popcorn.kernels._utils import default_scale, upcast


@register_kernel(
    test_shapes={"batch": Range(1, 8), "seq": Range(2, 128), "heads": {4}, "key_dim": {64}, "value_dim": {64}},
    test_args={"softmax_scale": [None, 0.25]},
    test_inputs={"k": lambda t: F.normalize(t, dim=-1), "beta": torch.sigmoid, "g": F.logsigmoid},
)
def gated_delta_rule(
    q: Float[Tensor, "batch seq heads key_dim"],
    k: Float[Tensor, "batch seq heads key_dim"],
    v: Float[Tensor, "batch seq heads value_dim"],
    g: Float[Tensor, "batch seq heads"],
    beta: Float[Tensor, "batch seq heads"],
    softmax_scale: float | None = None,
) -> Float[Tensor, "batch seq heads value_dim"]:
    """Gated DeltaNet (arXiv:2412.06464): the delta rule with a per-head log
    forget gate `g` applied to the state each step."""
    scale = default_scale(softmax_scale, q.shape[-1])
    q32, k32, v32, g32, beta32 = map(upcast, (q, k, v, g, beta))
    state = q32.new_zeros(q.shape[0], q.shape[2], q.shape[3], v.shape[3])
    outs = []
    for t in range(q.shape[1]):
        k_t = k32[:, t]
        state = state * g32[:, t].exp()[..., None, None]
        error = v32[:, t] - torch.einsum("bhkv,bhk->bhv", state, k_t)
        update = beta32[:, t, :, None] * error
        state = state + k_t[..., None] * update[..., None, :]
        outs.append(torch.einsum("bhkv,bhk->bhv", state, q32[:, t] * scale))
    return torch.stack(outs, 1).to(q.dtype)


@gated_delta_rule.register("fla", source="fla.ops.gated_delta_rule.chunk_gated_delta_rule")
def gated_delta_rule_fla(q, k, v, g, beta, softmax_scale):
    return kernel(q, k, v, g, beta, scale=softmax_scale)[0]
