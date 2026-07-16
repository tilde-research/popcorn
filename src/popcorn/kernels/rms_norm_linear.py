import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from popcorn import kernel, register_kernel
from popcorn.kernels._utils import rms


@register_kernel(test_args={"eps": [1e-6, 1e-5]})
def rms_norm_linear(
    x: Float[Tensor, "... hidden"],
    norm_weight: Float[Tensor, "hidden"],
    norm_bias: Float[Tensor, "hidden"] | None,
    linear_weight: Float[Tensor, "out_features hidden"],
    linear_bias: Float[Tensor, "out_features"] | None = None,
    eps: float = 1e-6,
) -> Float[Tensor, "... out_features"]:
    """RMS norm followed by a linear projection, fused by optimized backends."""
    h = rms(x, eps)
    h = h * norm_weight if norm_bias is None else h * norm_weight + norm_bias
    return F.linear(h, linear_weight, linear_bias)


@rms_norm_linear.register("fla", source="fla.modules.layernorm.rms_norm_linear")
def rms_norm_linear_fla(x, norm_weight, norm_bias, linear_weight, linear_bias, eps):
    return kernel(x, norm_weight, norm_bias, linear_weight, linear_bias, eps=eps)
