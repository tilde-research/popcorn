from typing import Literal

import torch
from jaxtyping import Float
from torch import Tensor

from popcorn import kernel, register_kernel
from popcorn.kernels._utils import rms


@register_kernel(test_args={"eps": [1e-6, 1e-5]})
def rms_norm(
    x: Float[Tensor, "... normalized_shape"],
    weight: Float[Tensor, "normalized_shape"],
    bias: Float[Tensor, "normalized_shape"] | None = None,
    eps: float = 1e-6,
) -> Float[Tensor, "... normalized_shape"]:
    """Root-mean-square normalization (arXiv:1910.07467): `x / rms(x) * weight (+ bias)`."""
    out = rms(x, eps) * weight
    return out if bias is None else out + bias


@rms_norm.register("fla", source="fla.modules.layernorm.rms_norm")
def rms_norm_fla(x, weight, bias, eps):
    return kernel(x, weight, bias, eps=eps)


@rms_norm.register("liger", source="liger_kernel.transformers.functional.liger_rms_norm")
def rms_norm_liger(x, weight, bias: Literal[None], eps):
    return kernel(x, weight, eps, in_place=False)


def _aligned(**arguments):
    """quack's cute kernels miscompile odd row widths for 16-bit dtypes."""
    x = arguments["x"]
    return x.dtype == torch.float32 or x.shape[-1] % 8 == 0


@rms_norm.register("quack", source="quack.rmsnorm.rmsnorm", predicate=_aligned)
def rms_norm_quack(x, weight, bias, eps):
    return kernel(x, weight, bias, eps=eps)
