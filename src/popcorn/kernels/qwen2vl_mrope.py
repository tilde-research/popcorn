import torch
from jaxtyping import Float
from torch import Tensor

from popcorn import Range, Tag, register_kernel


def _rotate_half(x):
    x1, x2 = x.chunk(2, -1)
    return torch.cat((-x2, x1), -1)


def _table(f):
    """RoPE tables are bounded and duplicate their halves; liger relies on it."""
    return lambda t: f(t).chunk(2, -1)[0].repeat(1, 1, 1, 2)


# mrope_section must sum to head_dim / 2; singleton test pools keep the grid
# consistent. cos/sin are constants, matching liger.
@register_kernel(
    test_shapes={"batch": Range(1, 8), "seq": Range(2, 512), "heads": {4}, "kv_heads": {2}, "head_dim": {32}},
    test_args={"mrope_section": [[8, 4, 4]]},
    test_inputs={"cos": _table(torch.cos), "sin": _table(torch.sin)},
    tags={Tag.POSITIONAL},
)
def qwen2vl_mrope(
    q: Float[Tensor, "batch heads seq head_dim"],
    k: Float[Tensor, "batch kv_heads seq head_dim"],
    cos: Float[Tensor, "3 batch seq head_dim"],
    sin: Float[Tensor, "3 batch seq head_dim"],
    mrope_section: list,
) -> tuple[Float[Tensor, "batch heads seq head_dim"], Float[Tensor, "batch kv_heads seq head_dim"]]:
    r"""Multimodal rotary embedding: temporal, height, and width tables interleaved per section.

    $$y = x \odot \tilde{\cos}_t + \mathrm{rot}_{1/2}(x) \odot \tilde{\sin}_t$$

    [RoFormer (Su et al., 2021)](https://arxiv.org/abs/2104.09864),
    [Qwen2-VL (Wang et al., 2024)](https://arxiv.org/abs/2409.12191)
    """
    doubled = [*mrope_section, *mrope_section]
    cos = torch.cat([m[i % 3] for i, m in enumerate(cos.detach().split(doubled, -1))], -1)[:, None]
    sin = torch.cat([m[i % 3] for i, m in enumerate(sin.detach().split(doubled, -1))], -1)[:, None]
    return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


qwen2vl_mrope.register("liger", source="liger_kernel.transformers.functional.liger_qwen2vl_mrope")
