import torch
from jaxtyping import Float
from torch import Tensor

from popcorn import Tag, register_kernel


@register_kernel(tags={Tag.ACTIVATION})
def sqrelu(x: Float[Tensor, "... hidden"]) -> Float[Tensor, "... hidden"]:
    r"""Squared ReLU.

    $$y = \max(x, 0)^2$$

    [Primer (So et al., 2021)](https://arxiv.org/abs/2109.08668)
    """
    return torch.relu(x).square()


sqrelu.register("fla", source="fla.modules.activations.sqrelu")
