from typing import Literal

import torch.nn.functional as F
from jaxtyping import BFloat16, Float, Float32
from torch import Tensor

from popcorn import kernel, register_kernel


@register_kernel
def gelu(
    x: Float[Tensor, "... hidden"],
    approximate: Literal["none", "tanh"] = "none",
) -> Float[Tensor, "... hidden"]:
    """Gaussian Error Linear Unit (arXiv:1606.08415): `x * Phi(x)`, exact or tanh-approximate."""
    return F.gelu(x, approximate=approximate)


# float16 tanh gradients land just past tolerance; fp32 and bf16 hold.
@gelu.register("fla", source="fla.modules.activations.fast_gelu_impl")
def gelu_fla(x: Float32[Tensor, "... hidden"] | BFloat16[Tensor, "... hidden"], approximate: Literal["tanh"]):
    return kernel(x)
