import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from popcorn import Range, Tag, kernel, register_kernel


@register_kernel(test_args={"eps": [1e-6, 1e-5]}, tags={Tag.NORMALIZATION})
def layer_norm(
    x: Float[Tensor, "... normalized_shape"],
    weight: Float[Tensor, "normalized_shape"],
    bias: Float[Tensor, "normalized_shape"] | None = None,
    eps: float = 1e-6,
) -> Float[Tensor, "... normalized_shape"]:
    r"""Layer normalization: standardize the last dim, then scale and shift.

    $$y = \frac{x - \overline{x}}{\sqrt{\operatorname{Var}(x) + \varepsilon}} \odot w + b$$

    [Layer Normalization (Ba et al., 2016)](https://arxiv.org/abs/1607.06450)
    """
    return F.layer_norm(x, weight.shape, weight, bias, eps)


# rows of one element are degenerate (grad x is exactly zero); fla's backward
# returns junk there.
@layer_norm.register("fla", source="fla.modules.layernorm.layer_norm", supports={"normalized_shape": Range(2, 1 << 20)})
def layer_norm_fla(x, weight, bias, eps):
    return kernel(x, weight, bias, eps=eps)


# tiny rows lose precision in liger's backward; see ISSUES.md.
@layer_norm.register(
    "liger",
    source="liger_kernel.transformers.functional.liger_layer_norm",
    supports={"normalized_shape": Range(8, 1 << 20)},
)
def layer_norm_liger(x, weight, bias: Float[Tensor, "normalized_shape"], eps):
    return kernel(x, weight, bias, eps)


# unsloth's backward returns only the input gradient (weight and bias are
# assumed frozen), so it is forward-only here; bias must be present.
@layer_norm.register("unsloth", source="unsloth.kernels.layernorm.Fast_Layernorm.apply", forward_only=True)
def layer_norm_unsloth(x, weight, bias: Float[Tensor, "normalized_shape"], eps):
    return kernel(x, weight, bias, eps)
