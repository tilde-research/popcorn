import torch
from jaxtyping import Float
from torch import Tensor

from popcorn import register_kernel


@register_kernel(test_args={"eps": [1e-6, 1e-5]})
def l2_norm(x: Float[Tensor, "... hidden"], eps: float = 1e-6) -> Float[Tensor, "... hidden"]:
    """L2 normalization along the last dim: `x * rsqrt(sum(x^2) + eps)`."""
    return x * torch.rsqrt(x.pow(2).sum(-1, keepdim=True) + eps)


l2_norm.register("fla", source="fla.modules.l2norm.l2_norm")
