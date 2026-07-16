import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from popcorn import kernel, register_kernel
from popcorn.kernels._quant import activation_quant, weight_quant
from popcorn.kernels._utils import rms


@register_kernel(test_args={"eps": [1e-6, 1e-5]})
def rms_norm_linear_quant(
    x: Float[Tensor, "... hidden"],
    norm_weight: Float[Tensor, "hidden"],
    norm_bias: Float[Tensor, "hidden"] | None,
    linear_weight: Float[Tensor, "out_features hidden"],
    linear_bias: Float[Tensor, "out_features"] | None = None,
    eps: float = 1e-6,
) -> Float[Tensor, "... out_features"]:
    """BitNet-style linear (arXiv:2310.11453) on an RMS-normed input: int8 activation and ternary weight fake quant around the matmul."""
    h = rms(x, eps)
    h = h * norm_weight if norm_bias is None else h * norm_weight + norm_bias
    return F.linear(activation_quant(h), weight_quant(linear_weight), linear_bias)


@rms_norm_linear_quant.register("fla", source="fla.modules.fused_bitlinear.rms_norm_linear_quant")
def rms_norm_linear_quant_fla(x, norm_weight, norm_bias, linear_weight, linear_bias, eps):
    return kernel(x, norm_weight, norm_bias, linear_weight, linear_bias, eps=eps)
