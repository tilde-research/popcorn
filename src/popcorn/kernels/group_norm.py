import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from popcorn import Tag, kernel, register_kernel


@register_kernel(test_args={"num_groups": [1, 4], "eps": [1e-6, 1e-5]}, tags={Tag.NORMALIZATION})
def group_norm(
    x: Float[Tensor, "tokens channels"],
    weight: Float[Tensor, "channels"],
    bias: Float[Tensor, "channels"] | None = None,
    num_groups: int = 1,
    eps: float = 1e-6,
) -> Float[Tensor, "tokens channels"]:
    r"""Group normalization: standardize within each of num_groups channel groups, then scale and shift.

    $$y_g = \frac{x_g - \overline{x_g}}{\sqrt{\operatorname{Var}(x_g) + \varepsilon}} \odot w_g + b_g$$

    [Group Normalization (Wu & He, 2018)](https://arxiv.org/abs/1803.08494)
    """
    return F.group_norm(x, num_groups, weight, bias, eps)


def _wide_groups(**arguments):
    """Both kernels miss tolerance on groups narrower than 8 channels."""
    return arguments["x"].shape[-1] // arguments["num_groups"] >= 8


@group_norm.register("fla", source="fla.modules.layernorm.group_norm", predicate=_wide_groups)
def group_norm_fla(x, weight, bias, num_groups, eps):
    return kernel(x, weight, bias, eps=eps, num_groups=num_groups)


# forward_only: liger's backward assumes a 3-D [batch, channels, length]
# input and crashes reshaping 2-D inputs; see ISSUES.md.
@group_norm.register(
    "liger",
    source="liger_kernel.transformers.functional.liger_group_norm",
    forward_only=True,
    predicate=_wide_groups,
)
def group_norm_liger(x, weight, bias: Float[Tensor, "channels"], num_groups, eps):
    return kernel(x, weight, bias, weight.shape[0], num_groups, eps)
