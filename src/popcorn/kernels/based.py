import torch
from jaxtyping import BFloat16, Float, Float32
from torch import Tensor

from popcorn import Range, kernel, register_kernel
from popcorn.kernels._utils import default_scale, upcast


@register_kernel(
    test_shapes={"batch": Range(1, 8), "seq": Range(2, 256), "heads": {4}, "key_dim": {16}, "value_dim": {64}},
    test_args={"softmax_scale": [None, 0.25]},
)
def based(
    q: Float[Tensor, "batch seq heads key_dim"],
    k: Float[Tensor, "batch seq heads key_dim"],
    v: Float[Tensor, "batch seq heads value_dim"],
    use_norm: bool = True,
    softmax_scale: float | None = None,
) -> Float[Tensor, "batch seq heads value_dim"]:
    """Based linear attention (arXiv:2402.18668): a second-order Taylor
    approximation of softmax, 1 + qk + (qk)^2 / 2."""
    scale = default_scale(softmax_scale, q.shape[-1])
    scores = torch.einsum("bqhk,bjhk->bhqj", upcast(q) * scale, upcast(k))
    attn = (1 + scores + 0.5 * scores.square()).tril()
    o = torch.einsum("bhqj,bjhv->bqhv", attn, upcast(v))
    if use_norm:
        o = o / (attn.sum(-1).transpose(1, 2)[..., None] + 1e-6)
    return o.to(q.dtype)


# float16 normalizer gradients land just past tolerance; bf16 and fp32 hold.
@based.register("fla", source="fla.ops.based.parallel_based")
def based_fla(
    q: Float32[Tensor, "batch seq heads key_dim"] | BFloat16[Tensor, "batch seq heads key_dim"],
    k,
    v,
    use_norm,
    softmax_scale,
):
    return kernel(q, k, v, scale=softmax_scale, use_norm=use_norm)
