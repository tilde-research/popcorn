import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from popcorn import Range, kernel, register_kernel


@register_kernel(test_args={"eps": [1e-6, 1e-5]})
def layer_norm_linear(
    x: Float[Tensor, "... hidden"],
    norm_weight: Float[Tensor, "hidden"],
    norm_bias: Float[Tensor, "hidden"] | None,
    linear_weight: Float[Tensor, "out_features hidden"],
    linear_bias: Float[Tensor, "out_features"] | None = None,
    eps: float = 1e-6,
) -> Float[Tensor, "... out_features"]:
    """Layer norm followed by a linear projection, fused by optimized backends."""
    h = F.layer_norm(x, norm_weight.shape, norm_weight, norm_bias, eps)
    return F.linear(h, linear_weight, linear_bias)


# rows of one element are degenerate (grad x is exactly zero); fla's backward
# returns junk there.
@layer_norm_linear.register("fla", source="fla.modules.layernorm.layer_norm_linear", supports={"hidden": Range(2, 1 << 20)})
def layer_norm_linear_fla(x, norm_weight, norm_bias, linear_weight, linear_bias, eps):
    return kernel(x, norm_weight, norm_bias, linear_weight, linear_bias, eps=eps)
