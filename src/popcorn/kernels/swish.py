import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from popcorn import Tag, register_kernel


@register_kernel(tags={Tag.ACTIVATION})
def swish(x: Float[Tensor, "... hidden"]) -> Float[Tensor, "... hidden"]:
    r"""Swish / SiLU.

    $$y = x \, \sigma(x)$$

    [Searching for Activation Functions (Ramachandran et al., 2017)](https://arxiv.org/abs/1710.05941)
    """
    return F.silu(x)


swish.register("fla", source="fla.modules.activations.swish")
