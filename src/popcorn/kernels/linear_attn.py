import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from popcorn import Range, Tag, kernel, register_kernel
from popcorn.kernels._utils import default_scale, upcast


# q and k pass through the paper's positive feature map (elu + 1) so the
# normalizer never crosses zero; feature-mapped q and k are the caller's job.
@register_kernel(
    test_shapes={"seq": Range(2, 256)},
    test_args={"softmax_scale": [None, 0.25]},
    test_inputs={"q": lambda t: F.elu(t) + 1, "k": lambda t: F.elu(t) + 1},
    tags={Tag.SEQUENCE_MIXER, Tag.LINEAR_ATTENTION},
)
def linear_attn(
    q: Float[Tensor, "batch seq heads key_dim"],
    k: Float[Tensor, "batch seq heads key_dim"],
    v: Float[Tensor, "batch seq heads value_dim"],
    normalize: bool = True,
    softmax_scale: float | None = None,
) -> Float[Tensor, "batch seq heads value_dim"]:
    r"""Causal linear attention, optionally normalized by the running key sum.

    $$S_t = S_{t-1} + k_t v_t^\top, \qquad o_t = c \, q_t^\top S_t$$

    [Transformers are RNNs (Katharopoulos et al., 2020)](https://arxiv.org/abs/2006.16236)
    """
    scale = default_scale(softmax_scale, q.shape[-1])
    q32, k32 = upcast(q) * scale, upcast(k)
    scores = torch.einsum("bqhk,bjhk->bhqj", q32, k32).tril()
    o = torch.einsum("bhqj,bjhv->bqhv", scores, upcast(v))
    if normalize:
        z = torch.einsum("bqhk,bqhk->bqh", q32, k32.cumsum(1))
        o = o / (z[..., None] + 1e-10)
    return o.to(q.dtype)


def _supported(**arguments):
    """fla refuses seq < heads, guessing the layout is head-first (see ISSUES.md),
    and float16 normalizer gradients land just past tolerance."""
    q = arguments["q"]
    fp16_norm = q.dtype == torch.float16 and arguments["normalize"]
    return q.shape[1] >= q.shape[2] and not fp16_norm


@linear_attn.register("fla", source="fla.ops.linear_attn.chunk_linear_attn", predicate=_supported)
def linear_attn_fla(q, k, v, normalize, softmax_scale):
    return kernel(q, k, v, scale=softmax_scale, normalize=normalize)[0]
