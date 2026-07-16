import torch
import torch.nn.functional as F
from jaxtyping import BFloat16, Float
from torch import Tensor

from popcorn import Range, kernel, register_kernel
from popcorn.kernels._utils import default_scale, upcast


@register_kernel(
    test_shapes={"batch": Range(1, 8), "seq": Range(2, 128), "heads": {4}, "key_dim": {64}, "value_dim": {64}},
    test_args={"softmax_scale": [None, 0.25]},
    test_inputs={"k": lambda t: F.normalize(t, dim=-1), "beta": torch.sigmoid},
)
def delta_rule(
    q: Float[Tensor, "batch seq heads key_dim"],
    k: Float[Tensor, "batch seq heads key_dim"],
    v: Float[Tensor, "batch seq heads value_dim"],
    beta: Float[Tensor, "batch seq heads"],
    softmax_scale: float | None = None,
) -> Float[Tensor, "batch seq heads value_dim"]:
    """DeltaNet (arXiv:2406.06484): S_t = S_{t-1}(I - beta_t k_t k_t^T) + beta_t k_t v_t^T.
    Keys should be unit-norm for stability."""
    scale = default_scale(softmax_scale, q.shape[-1])
    q32, k32, v32, beta32 = map(upcast, (q, k, v, beta))
    state = q32.new_zeros(q.shape[0], q.shape[2], q.shape[3], v.shape[3])
    outs = []
    for t in range(q.shape[1]):
        k_t = k32[:, t]
        error = v32[:, t] - torch.einsum("bhkv,bhk->bhv", state, k_t)
        update = beta32[:, t, :, None] * error
        state = state + k_t[..., None] * update[..., None, :]
        outs.append(torch.einsum("bhkv,bhk->bhv", state, q32[:, t] * scale))
    return torch.stack(outs, 1).to(q.dtype)


# bfloat16 only: fla asserts against float32 inputs ("Please use bfloat16")
# and float16 key gradients land just past tolerance.
@delta_rule.register("fla", source="fla.ops.delta_rule.chunk_delta_rule")
def delta_rule_fla(q: BFloat16[Tensor, "batch seq heads key_dim"], k, v, beta, softmax_scale):
    return kernel(q, k, v, beta, scale=softmax_scale)[0]
