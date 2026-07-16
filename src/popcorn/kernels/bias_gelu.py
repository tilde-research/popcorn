import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from popcorn import register_kernel


@register_kernel
def bias_gelu(x: Float[Tensor, "... hidden"], bias: Float[Tensor, "hidden"]) -> Float[Tensor, "... hidden"]:
    """Tanh-approximate GELU applied to `x + bias`, the classic fused bias activation."""
    return F.gelu(x + bias, approximate="tanh")


# forward_only: fla's GeLUFunction.backward returns the (grad_input, grad_bias)
# tuple twice instead of unpacking it; see ISSUES.md.
bias_gelu.register("fla", source="fla.modules.activations.bias_gelu_impl", forward_only=True)
