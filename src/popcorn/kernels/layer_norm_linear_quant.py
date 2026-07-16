import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from popcorn import Range, kernel, register_kernel
from popcorn.kernels._quant import activation_quant, weight_quant


@register_kernel(test_args={"eps": [1e-6, 1e-5]})
def layer_norm_linear_quant(
    x: Float[Tensor, "... hidden"],
    norm_weight: Float[Tensor, "hidden"],
    norm_bias: Float[Tensor, "hidden"] | None,
    linear_weight: Float[Tensor, "out_features hidden"],
    linear_bias: Float[Tensor, "out_features"] | None = None,
    eps: float = 1e-6,
) -> Float[Tensor, "... out_features"]:
    """BitNet-style linear (arXiv:2310.11453) on a layer-normed input: int8 activation and ternary weight fake quant around the matmul."""
    h = activation_quant(F.layer_norm(x, norm_weight.shape, norm_weight, norm_bias, eps))
    return F.linear(h, weight_quant(linear_weight), linear_bias)


# rows of one element are degenerate (grad x is exactly zero); fla's backward
# returns junk there.
@layer_norm_linear_quant.register(
    "fla",
    source="fla.modules.fused_bitlinear.layer_norm_linear_quant_fn",
    supports={"hidden": Range(2, 1 << 20)},
)
def layer_norm_linear_quant_fla(x, norm_weight, norm_bias, linear_weight, linear_bias, eps):
    return kernel(x, norm_weight, norm_bias, linear_weight, linear_bias, eps=eps)
