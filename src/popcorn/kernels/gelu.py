from typing import Literal

import torch.nn.functional as F
from jaxtyping import BFloat16, Float, Float32
from torch import Tensor

from popcorn import Tag, kernel, register_kernel


@register_kernel(tags={Tag.ACTIVATION})
def gelu(
    x: Float[Tensor, "... hidden"],
    approximate: Literal["none", "tanh"] = "none",
) -> Float[Tensor, "... hidden"]:
    r"""Gaussian Error Linear Unit, exact or tanh-approximate.

    $$y = x \, \Phi(x)$$

    [GELU (Hendrycks & Gimpel, 2016)](https://arxiv.org/abs/1606.08415)
    """
    return F.gelu(x, approximate=approximate)


# float16 tanh gradients land just past tolerance; fp32 and bf16 hold.
@gelu.register("fla", source="fla.modules.activations.fast_gelu_impl")
def gelu_fla(x: Float32[Tensor, "... hidden"] | BFloat16[Tensor, "... hidden"], approximate: Literal["tanh"]):
    return kernel(x)
