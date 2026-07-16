import torch
from jaxtyping import Float, Float32
from torch import Tensor

from popcorn import Range, kernel, register_kernel
from popcorn.kernels._utils import upcast


@register_kernel(test_shapes={"batch": Range(1, 8), "seq": Range(2, 256), "heads": {4}, "key_dim": {16}, "value_dim": {64}})
def rebased(
    q: Float[Tensor, "batch seq heads key_dim"],
    k: Float[Tensor, "batch seq heads key_dim"],
    v: Float[Tensor, "batch seq heads value_dim"],
    eps: float = 1e-5,
    use_scale: bool = True,
    use_normalize: bool = True,
) -> Float[Tensor, "batch seq heads value_dim"]:
    """ReBased linear attention (arXiv:2402.10644): a learned-free quadratic
    feature map, (qk)^2."""
    scale = q.shape[-1] ** -0.5 if use_scale else 1.0
    scores = torch.einsum("bqhk,bjhk->bhqj", upcast(q) * scale, upcast(k))
    attn = scores.square().tril()
    o = torch.einsum("bhqj,bjhv->bqhv", attn, upcast(v))
    if use_normalize:
        o = o / (attn.sum(-1).transpose(1, 2)[..., None] + eps)
    return o.to(q.dtype)


# float32 only: 16-bit gradients through the quadratic normalizer overflow to
# inf (fp16) or miss tolerance (bf16); see ISSUES.md.
@rebased.register("fla", source="fla.ops.rebased.parallel_rebased")
def rebased_fla(q: Float32[Tensor, "batch seq heads key_dim"], k, v, eps, use_scale, use_normalize):
    return kernel(q, k, v, eps=eps, use_scale=use_scale, use_normalize=use_normalize)
