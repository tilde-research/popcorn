import torch
from jaxtyping import Float
from torch import Tensor

from popcorn import Range, register_kernel


def _rotate_half(x):
    x1, x2 = x.chunk(2, -1)
    return torch.cat((-x2, x1), -1)


def _table(f):
    return lambda t: f(t).chunk(2, -1)[0].repeat(1, 1, 2)


@register_kernel(
    test_shapes={
        "batch": Range(1, 8),
        "seq": Range(2, 512),
        "heads": {4},
        "kv_heads": {2},
        "cos_batch": {1},
        "head_dim": {32},
    },
    test_inputs={"cos": _table(torch.cos), "sin": _table(torch.sin)},
)
def rope(
    q: Float[Tensor, "batch heads seq head_dim"],
    k: Float[Tensor, "batch kv_heads seq head_dim"],
    cos: Float[Tensor, "cos_batch seq head_dim"],
    sin: Float[Tensor, "cos_batch seq head_dim"],
) -> tuple[Float[Tensor, "batch heads seq head_dim"], Float[Tensor, "batch kv_heads seq head_dim"]]:
    """Paired Llama-style rotary embedding. The frequency tables may have one
    batch or match `q` and duplicate their first half."""
    cos, sin = cos.detach()[:, None], sin.detach()[:, None]
    return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


rope.register("liger", source="liger_kernel.transformers.functional.liger_rope")
