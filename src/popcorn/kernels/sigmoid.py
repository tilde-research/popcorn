import torch
from jaxtyping import Float
from torch import Tensor

from popcorn import register_kernel


@register_kernel
def sigmoid(x: Float[Tensor, "... hidden"]) -> Float[Tensor, "... hidden"]:
    """Logistic sigmoid: `1 / (1 + exp(-x))`."""
    return torch.sigmoid(x)


sigmoid.register("fla", source="fla.modules.activations.sigmoid")
