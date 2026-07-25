import torch
from jaxtyping import Float
from torch import Tensor

from popcorn import Tag, kernel, register_kernel


def _rotate_half(x):
    x1, x2 = x.chunk(2, -1)
    return torch.cat((-x2, x1), -1)


def _table(f):
    return lambda t: f(t).chunk(2, -1)[0].repeat(1, 1, 2)


# The frequency tables may have one batch or match `q`, and duplicate their
# first half.
@register_kernel(
    test_inputs={"cos": _table(torch.cos), "sin": _table(torch.sin)},
    tags={Tag.POSITIONAL},
)
def rope(
    q: Float[Tensor, "batch q_heads seq head_dim"],
    k: Float[Tensor, "batch kv_heads seq head_dim"],
    cos: Float[Tensor, "cos_batch seq head_dim"],
    sin: Float[Tensor, "cos_batch seq head_dim"],
) -> tuple[Float[Tensor, "batch q_heads seq head_dim"], Float[Tensor, "batch kv_heads seq head_dim"]]:
    r"""Paired Llama-style rotary embedding of queries and keys.

    $$y = x \odot \cos_t + \mathrm{rot}_{1/2}(x) \odot \sin_t, \qquad x \in \{q, k\}$$

    [RoFormer (Su et al., 2021)](https://arxiv.org/abs/2104.09864)
    """
    cos, sin = cos.detach()[:, None], sin.detach()[:, None]
    return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


rope.register("liger", source="liger_kernel.transformers.functional.liger_rope")


# unsloth rotates each of q, k with the first half of the (duplicated) tables,
# matching the reference; cos/sin are non-differentiable in both.
@rope.register("unsloth", source="unsloth.kernels.rope_embedding.fast_rope_embedding")
def rope_unsloth(q, k, cos, sin):
    return kernel(q, k, cos, sin)
