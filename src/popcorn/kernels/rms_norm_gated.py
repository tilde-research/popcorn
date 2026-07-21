from typing import Literal

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from popcorn import Tag, kernel, register_kernel
from popcorn.kernels._utils import rms


@register_kernel(test_args={"eps": [1e-6, 1e-5]}, tags={Tag.NORMALIZATION, Tag.ACTIVATION, Tag.FUSED})
def rms_norm_gated(
    x: Float[Tensor, "... hidden"],
    g: Float[Tensor, "... hidden"],
    weight: Float[Tensor, "hidden"],
    bias: Float[Tensor, "hidden"] | None = None,
    activation: Literal["swish", "sigmoid"] = "swish",
    eps: float = 1e-6,
) -> Float[Tensor, "... hidden"]:
    r"""RMS norm scaled by a swish or sigmoid gate.

    $$y = \left(\frac{x}{\sqrt{\overline{x^2} + \varepsilon}} \odot w + b\right) \odot \mathrm{act}(g)$$

    [RMSNorm (Zhang & Sennrich, 2019)](https://arxiv.org/abs/1910.07467)
    """
    out = rms(x, eps) * weight
    out = out if bias is None else out + bias
    return out * (F.silu(g) if activation == "swish" else torch.sigmoid(g))


@rms_norm_gated.register("fla", source="fla.modules.fused_norm_gate.rms_norm_gated")
def rms_norm_gated_fla(x, g, weight, bias, activation, eps):
    return kernel(x, g, weight, bias, activation=activation, eps=eps)
