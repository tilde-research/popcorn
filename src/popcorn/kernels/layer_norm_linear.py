import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from popcorn import Tag, kernel, register_kernel


@register_kernel(test_args={"eps": [1e-6, 1e-5]}, tags={Tag.NORMALIZATION, Tag.LINEAR, Tag.FUSED, Tag.FEATURE_MIXER})
def layer_norm_linear(
    x: Float[Tensor, "... hidden"],
    norm_weight: Float[Tensor, "hidden"],
    norm_bias: Float[Tensor, "hidden"] | None,
    linear_weight: Float[Tensor, "out_features hidden"],
    linear_bias: Float[Tensor, "out_features"] | None = None,
    eps: float = 1e-6,
) -> Float[Tensor, "... out_features"]:
    r"""Layer norm fused with a linear projection.

    $$y = \mathrm{LN}_{w_n, b_n}(x) \, w_l^\top + b_l$$

    [Layer Normalization (Ba et al., 2016)](https://arxiv.org/abs/1607.06450)
    """
    h = F.layer_norm(x, norm_weight.shape, norm_weight, norm_bias, eps)
    return F.linear(h, linear_weight, linear_bias)


# rows of one element are degenerate (grad x is exactly zero); fla's backward
# returns junk there.
@layer_norm_linear.register("fla", source="fla.modules.layernorm.layer_norm_linear")
def layer_norm_linear_fla(x, norm_weight, norm_bias, linear_weight, linear_bias, eps):
    return kernel(x, norm_weight, norm_bias, linear_weight, linear_bias, eps=eps)
