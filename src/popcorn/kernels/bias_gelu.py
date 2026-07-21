import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from popcorn import Tag, register_kernel


@register_kernel(tags={Tag.ACTIVATION, Tag.FUSED})
def bias_gelu(x: Float[Tensor, "... hidden"], bias: Float[Tensor, "hidden"]) -> Float[Tensor, "... hidden"]:
    r"""The classic fused bias activation: tanh-approximate GELU of a biased input.

    $$y = \mathrm{gelu}(x + b)$$

    [GELU (Hendrycks & Gimpel, 2016)](https://arxiv.org/abs/1606.08415)
    """
    return F.gelu(x + bias, approximate="tanh")


# forward_only: fla's GeLUFunction.backward returns the (grad_input, grad_bias)
# tuple twice instead of unpacking it; see ISSUES.md.
bias_gelu.register("fla", source="fla.modules.activations.bias_gelu_impl", forward_only=True)
