import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from popcorn import register_kernel


@register_kernel
def swish(x: Float[Tensor, "... hidden"]) -> Float[Tensor, "... hidden"]:
    """Swish / SiLU (arXiv:1710.05941): `x * sigmoid(x)`."""
    return F.silu(x)


swish.register("fla", source="fla.modules.activations.swish")
