import torch
from jaxtyping import Float
from torch import Tensor

from popcorn import register_kernel


@register_kernel
def sqrelu(x: Float[Tensor, "... hidden"]) -> Float[Tensor, "... hidden"]:
    """Squared ReLU (arXiv:2109.08668): `relu(x)^2`."""
    return torch.relu(x).square()


sqrelu.register("fla", source="fla.modules.activations.sqrelu")
