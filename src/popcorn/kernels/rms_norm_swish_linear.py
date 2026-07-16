import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from popcorn import kernel, register_kernel
from popcorn.kernels._utils import rms


@register_kernel(test_args={"eps": [1e-6, 1e-5]})
def rms_norm_swish_linear(
    x: Float[Tensor, "... hidden"],
    g: Float[Tensor, "... hidden"],
    norm_weight: Float[Tensor, "hidden"],
    norm_bias: Float[Tensor, "hidden"] | None,
    linear_weight: Float[Tensor, "out_features hidden"],
    linear_bias: Float[Tensor, "out_features"] | None = None,
    eps: float = 1e-6,
) -> Float[Tensor, "... out_features"]:
    """RMS norm, swish gate, then linear: `linear(RMSNorm(x) * silu(g))`."""
    h = rms(x, eps)
    h = h * norm_weight if norm_bias is None else h * norm_weight + norm_bias
    return F.linear(h * F.silu(g), linear_weight, linear_bias)


# forward_only: fla drops the silu(g) factor in dlinear_weight; see ISSUES.md.
@rms_norm_swish_linear.register("fla", source="fla.modules.fused_norm_gate.rms_norm_swish_gate_linear", forward_only=True)
def rms_norm_swish_linear_fla(x, g, norm_weight, norm_bias, linear_weight, linear_bias, eps):
    return kernel(x, g, norm_weight, norm_bias, linear_weight, linear_bias, eps=eps)
