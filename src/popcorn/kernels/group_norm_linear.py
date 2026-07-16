import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from popcorn import Div, kernel, register_kernel


@register_kernel(test_shapes={"channels": Div(4)}, test_args={"num_groups": [1, 4], "eps": [1e-6, 1e-5]})
def group_norm_linear(
    x: Float[Tensor, "tokens channels"],
    norm_weight: Float[Tensor, "channels"],
    norm_bias: Float[Tensor, "channels"] | None,
    linear_weight: Float[Tensor, "out_features channels"],
    linear_bias: Float[Tensor, "out_features"] | None = None,
    num_groups: int = 1,
    eps: float = 1e-6,
) -> Float[Tensor, "tokens out_features"]:
    """Group norm followed by a linear projection, fused by optimized backends."""
    h = F.group_norm(x, num_groups, norm_weight, norm_bias, eps)
    return F.linear(h, linear_weight, linear_bias)


def _wide_groups(**arguments):
    """fla's grad x drifts a few 1e-3 relative on groups narrower than 8 channels."""
    return arguments["x"].shape[-1] // arguments["num_groups"] >= 8


@group_norm_linear.register("fla", source="fla.modules.layernorm.group_norm_linear", predicate=_wide_groups)
def group_norm_linear_fla(x, norm_weight, norm_bias, linear_weight, linear_bias, num_groups, eps):
    return kernel(x, norm_weight, norm_bias, linear_weight, linear_bias, eps=eps, num_groups=num_groups)
