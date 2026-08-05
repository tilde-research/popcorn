from typing import Literal

import torch

# liger's rms_norm checks `isinstance(X, torch.distributed.tensor.DTensor)` without
# importing the submodule, which only resolves when something else imported it first
# (see ISSUES.md); loading it here keeps the check working in minimal environments.
import torch.distributed.tensor  # noqa: F401
from jaxtyping import Float
from torch import Tensor

from popcorn import Tag, kernel, register_kernel
from popcorn.kernels._utils import rms


@register_kernel(test_args={"eps": [1e-6, 1e-5]}, tags={Tag.NORMALIZATION})
def rms_norm(
    x: Float[Tensor, "... normalized_shape"],
    weight: Float[Tensor, "normalized_shape"],
    bias: Float[Tensor, "normalized_shape"] | None = None,
    eps: float = 1e-6,
) -> Float[Tensor, "... normalized_shape"]:
    r"""Root-mean-square normalization.

    $$y = \frac{x}{\sqrt{\overline{x^2} + \varepsilon}} \odot w + b$$

    [RMSNorm (Zhang & Sennrich, 2019)](https://arxiv.org/abs/1910.07467)
    """
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


@rms_norm.register("quack", source="quack.rmsnorm", predicate=_aligned)
def rms_norm_quack(x, weight, bias, eps):
    return kernel(x, weight, bias, eps=eps)


# unsloth's backward returns only the input gradient (norm weights are assumed
# frozen), so it is forward-only here; the kernel has no bias term. The final
# False selects the plain variant over gemma's (1 + w) weighting.
@rms_norm.register("unsloth", source="unsloth.kernels.rms_layernorm.Fast_RMS_Layernorm.apply", forward_only=True)
def rms_norm_unsloth(x, weight, bias: Literal[None], eps):
    return kernel(x, weight, eps, False)
