import torch
from jaxtyping import Float
from torch import Tensor

from popcorn import Tag, register_kernel


@register_kernel(test_args={"eps": [1e-6, 1e-5]}, tags={Tag.NORMALIZATION})
def l2_norm(x: Float[Tensor, "... hidden"], eps: float = 1e-6) -> Float[Tensor, "... hidden"]:
    r"""L2 normalization along the last dim.

    $$y = \frac{x}{\sqrt{\sum_i x_i^2 + \varepsilon}}$$
    """
    return x * torch.rsqrt(x.pow(2).sum(-1, keepdim=True) + eps)


l2_norm.register("fla", source="fla.modules.l2norm.l2_norm")
