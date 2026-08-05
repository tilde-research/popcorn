import torch
from jaxtyping import BFloat16, Float, Float32
from torch import Tensor

from popcorn import Tag, kernel, register_kernel
from popcorn.kernels._utils import default_scale, upcast


@register_kernel(
    test_args={"softmax_scale": [None, 0.25]},
    tags={Tag.SEQUENCE_MIXER, Tag.LINEAR_ATTENTION},
)
def based(
    q: Float[Tensor, "batch seq heads key_dim"],
    k: Float[Tensor, "batch seq heads key_dim"],
    v: Float[Tensor, "batch seq heads value_dim"],
    use_norm: bool = True,
    softmax_scale: float | None = None,
) -> Float[Tensor, "batch seq heads value_dim"]:
    r"""Based linear attention: a second-order Taylor approximation of the softmax kernel.

    $$a_{ts} = 1 + q_t^\top k_s + \tfrac{1}{2} \big(q_t^\top k_s\big)^2, \qquad
    o_t = \frac{\sum_{s \le t} a_{ts} v_s}{\sum_{s \le t} a_{ts}}$$

    [Based (Arora et al., 2024)](https://arxiv.org/abs/2402.18668)
    """
    scale = default_scale(softmax_scale, q.shape[-1])
    scores = torch.einsum("bqhk,bjhk->bhqj", upcast(q) * scale, upcast(k))
    attn = (1 + scores + 0.5 * scores.square()).tril()
    o = torch.einsum("bhqj,bjhv->bqhv", attn, upcast(v))
    if use_norm:
        o = o / (attn.sum(-1).transpose(1, 2)[..., None] + 1e-6)
    return o.to(q.dtype)


# float16 normalizer gradients land just past tolerance; bf16 and fp32 hold.
def _fla_ready(**arguments):
    return arguments["q"].shape[-1] <= 128


@based.register("fla", source="fla.ops.based.parallel_based", predicate=_fla_ready)
def based_fla(
    q: Float32[Tensor, "batch seq heads key_dim"] | BFloat16[Tensor, "batch seq heads key_dim"],
    k,
    v,
    use_norm,
    softmax_scale,
):
    return kernel(q, k, v, scale=softmax_scale, use_norm=use_norm)
