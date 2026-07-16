import torch
import torch.nn.functional as F
from jaxtyping import BFloat16, Float
from torch import Tensor

from popcorn import Range, kernel, register_kernel
from popcorn.kernels._utils import default_scale, upcast


# seq_h must equal seq * num_householder (fla's contract); singleton test pools
# keep the grid consistent, like grpo's resp/resp_plus.
@register_kernel(
    test_shapes={"batch": Range(1, 8), "seq": {32}, "seq_h": {64}, "heads": {4}, "key_dim": {64}, "value_dim": {64}},
    test_args={"num_householder": [2], "softmax_scale": [None, 0.25]},
    test_inputs={"k": lambda t: F.normalize(t, dim=-1), "beta": torch.sigmoid, "g": F.logsigmoid},
)
def gated_delta_product(
    q: Float[Tensor, "batch seq heads key_dim"],
    k: Float[Tensor, "batch seq_h heads key_dim"],
    v: Float[Tensor, "batch seq_h heads value_dim"],
    g: Float[Tensor, "batch seq heads"],
    beta: Float[Tensor, "batch seq_h heads"],
    num_householder: int = 1,
    softmax_scale: float | None = None,
) -> Float[Tensor, "batch seq heads value_dim"]:
    """Gated DeltaProduct (arXiv:2502.10297): `num_householder` delta-rule writes
    per step (k/v/beta carry `seq * num_householder` positions), with one per-head
    log forget gate applied per step."""
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
