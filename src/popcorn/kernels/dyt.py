import torch
from jaxtyping import Float
from torch import Tensor

from popcorn import Tag, register_kernel


@register_kernel(tags={Tag.NORMALIZATION, Tag.ACTIVATION})
def dyt(
    x: Float[Tensor, "... hidden"],
    alpha: Float[Tensor, "1"],
    gamma: Float[Tensor, "hidden"],
    beta: Float[Tensor, "hidden"] | None = None,
) -> Float[Tensor, "... hidden"]:
    r"""Dynamic Tanh, an elementwise normalization replacement.

    $$y = \gamma \odot \tanh(\alpha x) + \beta$$

    [Transformers without Normalization (Zhu et al., 2025)](https://arxiv.org/abs/2503.10622)
    """
    out = gamma * torch.tanh(alpha * x)
    return out if beta is None else out + beta


dyt.register("liger", source="liger_kernel.transformers.functional.liger_dyt")
