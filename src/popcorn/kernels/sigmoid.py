import torch
from jaxtyping import Float
from torch import Tensor

from popcorn import Tag, register_kernel


@register_kernel(tags={Tag.ACTIVATION})
def sigmoid(x: Float[Tensor, "... hidden"]) -> Float[Tensor, "... hidden"]:
    r"""Logistic sigmoid.

    $$y = \frac{1}{1 + e^{-x}}$$
    """
    return torch.sigmoid(x)


sigmoid.register("fla", source="fla.modules.activations.sigmoid")
