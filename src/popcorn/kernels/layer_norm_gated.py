from typing import Literal

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from popcorn import Range, Tag, kernel, register_kernel


@register_kernel(test_args={"eps": [1e-6, 1e-5]}, tags={Tag.NORMALIZATION, Tag.ACTIVATION, Tag.FUSED})
def layer_norm_gated(
    x: Float[Tensor, "... hidden"],
    g: Float[Tensor, "... hidden"],
    weight: Float[Tensor, "hidden"],
    bias: Float[Tensor, "hidden"] | None = None,
    activation: Literal["swish", "sigmoid"] = "swish",
    eps: float = 1e-6,
) -> Float[Tensor, "... hidden"]:
    r"""Layer norm scaled by a swish or sigmoid gate.

    $$y = \mathrm{LN}_{w,b}(x) \odot \mathrm{act}(g)$$

    [Layer Normalization (Ba et al., 2016)](https://arxiv.org/abs/1607.06450)
    """
    out = F.layer_norm(x, weight.shape, weight, bias, eps)
    return out * (F.silu(g) if activation == "swish" else torch.sigmoid(g))


# rows of one element are degenerate (grad x is exactly zero); fla's backward
# returns junk there.
@layer_norm_gated.register(
    "fla", source="fla.modules.fused_norm_gate.layer_norm_gated", supports={"hidden": Range(2, 1 << 20)}
)
def layer_norm_gated_fla(x, g, weight, bias, activation, eps):
    return kernel(x, g, weight, bias, activation=activation, eps=eps)
