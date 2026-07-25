import torch
from jaxtyping import BFloat16, Float, Float32
from torch import Tensor

from popcorn import Tag, kernel, register_kernel
from popcorn.kernels._utils import upcast


@register_kernel(test_args={"eps": [1e-6, 1e-5]}, tags={Tag.NORMALIZATION, Tag.ACTIVATION})
def poly_norm(
    x: Float[Tensor, "... hidden"], weight: Float[Tensor, "3"], bias: Float[Tensor, ""], eps: float = 1e-6
) -> Float[Tensor, "... hidden"]:
    r"""Polynomial composition of RMS-normalized powers of the input.

    $$y = w_0\,\mathrm{n}(x^3) + w_1\,\mathrm{n}(x^2) + w_2\,\mathrm{n}(x) + b,
    \qquad \mathrm{n}(u) = \frac{u}{\sqrt{\overline{u^2} + \varepsilon}}$$

    [PolyNorm (Zhuo et al., 2024)](https://arxiv.org/abs/2411.03884)
    """
    h = upcast(x)
    powers = [h**3, h**2, h]
    normed = [u * torch.rsqrt(u.pow(2).mean(-1, keepdim=True) + eps) for u in powers]
    out = sum(w * u for w, u in zip(upcast(weight), normed)) + upcast(bias)
    return out.to(x.dtype)


# fp16 overflows: the kernel accumulates x^6 for the cubic term's rms in half
# precision, and bf16 gradients on degenerate 1-2 element rows miss tolerance
# (see ISSUES.md).
@poly_norm.register("liger", source="liger_kernel.transformers.functional.liger_poly_norm")
def poly_norm_liger(x: Float32[Tensor, "... hidden"] | BFloat16[Tensor, "... hidden"], weight, bias, eps):
    return kernel(x, weight, bias, eps, in_place=False)
