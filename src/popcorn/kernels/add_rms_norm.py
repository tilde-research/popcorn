from jaxtyping import Float
from torch import Tensor

from popcorn import kernel, register_kernel
from popcorn.kernels._utils import rms


@register_kernel(test_args={"eps": [1e-6, 1e-5], "offset": [0.0, 1.0]})
def add_rms_norm(
    x: Float[Tensor, "... hidden"],
    residual: Float[Tensor, "... hidden"],
    weight: Float[Tensor, "hidden"],
    eps: float = 1e-6,
    offset: float = 0.0,
) -> tuple[Tensor, Tensor]:
    """Residual add followed by RMS norm with a weight offset (offset=1 gives the
    Gemma convention). Returns (normalized, x + residual); the sum feeds the next
    residual stream."""
    s = x + residual
    return rms(s, eps) * (weight + offset), s


@add_rms_norm.register("liger", source="liger_kernel.transformers.functional.liger_fused_add_rms_norm")
def add_rms_norm_liger(x, residual, weight, eps, offset):
    return kernel(x, residual, weight, eps, offset, casting_mode="llama", in_place=False)
