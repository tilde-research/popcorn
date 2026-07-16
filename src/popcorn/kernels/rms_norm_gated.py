from typing import Literal

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from popcorn import kernel, register_kernel
from popcorn.kernels._utils import rms


@register_kernel(test_args={"eps": [1e-6, 1e-5]})
def rms_norm_gated(
    x: Float[Tensor, "... hidden"],
    g: Float[Tensor, "... hidden"],
    weight: Float[Tensor, "hidden"],
    bias: Float[Tensor, "hidden"] | None = None,
    activation: Literal["swish", "sigmoid"] = "swish",
    eps: float = 1e-6,
) -> Float[Tensor, "... hidden"]:
    """RMS norm times a swish or sigmoid gate: `RMSNorm(x) * act(g)`."""
    out = rms(x, eps) * weight
    out = out if bias is None else out + bias
    return out * (F.silu(g) if activation == "swish" else torch.sigmoid(g))


@rms_norm_gated.register("fla", source="fla.modules.fused_norm_gate.rms_norm_gated")
def rms_norm_gated_fla(x, g, weight, bias, activation, eps):
    return kernel(x, g, weight, bias, activation=activation, eps=eps)
