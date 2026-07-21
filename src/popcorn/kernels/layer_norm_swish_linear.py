import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from popcorn import Tag, kernel, register_kernel


@register_kernel(
    test_args={"eps": [1e-6, 1e-5]}, tags={Tag.NORMALIZATION, Tag.ACTIVATION, Tag.LINEAR, Tag.FUSED, Tag.FEATURE_MIXER}
)
def layer_norm_swish_linear(
    x: Float[Tensor, "... hidden"],
    g: Float[Tensor, "... hidden"],
    norm_weight: Float[Tensor, "hidden"],
    norm_bias: Float[Tensor, "hidden"] | None,
    linear_weight: Float[Tensor, "out_features hidden"],
    linear_bias: Float[Tensor, "out_features"] | None = None,
    eps: float = 1e-6,
) -> Float[Tensor, "... out_features"]:
    r"""Layer norm, swish gate, then a linear projection, in one fused op.

    $$y = \big(\mathrm{LN}_{w_n, b_n}(x) \odot \mathrm{silu}(g)\big) \, w_l^\top + b_l$$

    [Layer Normalization (Ba et al., 2016)](https://arxiv.org/abs/1607.06450)
    """
    h = F.layer_norm(x, norm_weight.shape, norm_weight, norm_bias, eps) * F.silu(g)
    return F.linear(h, linear_weight, linear_bias)


# forward_only: fla drops the silu(g) factor in dlinear_weight; see ISSUES.md.
@layer_norm_swish_linear.register("fla", source="fla.modules.fused_norm_gate.layer_norm_swish_gate_linear", forward_only=True)
def layer_norm_swish_linear_fla(x, g, norm_weight, norm_bias, linear_weight, linear_bias, eps):
    return kernel(x, g, norm_weight, norm_bias, linear_weight, linear_bias, eps=eps)
