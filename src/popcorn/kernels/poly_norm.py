import torch
from jaxtyping import BFloat16, Float, Float32
from torch import Tensor

from popcorn import Range, kernel, register_kernel
from popcorn.kernels._utils import upcast


@register_kernel(test_args={"eps": [1e-6, 1e-5]})
def poly_norm(
    x: Float[Tensor, "... hidden"],
    weight: Float[Tensor, "3"],
    bias: Float[Tensor, ""],
    eps: float = 1e-6,
) -> Float[Tensor, "... hidden"]:
    """PolyNorm (arXiv:2411.03884): y = w0*norm(x^3) + w1*norm(x^2) + w2*norm(x) + b,
    where norm(u) = u * rsqrt(mean(u^2) + eps) over the last dimension."""
    h = upcast(x)
    powers = [h**3, h**2, h]
    normed = [u * torch.rsqrt(u.pow(2).mean(-1, keepdim=True) + eps) for u in powers]
    out = sum(w * u for w, u in zip(upcast(weight), normed)) + upcast(bias)
    return out.to(x.dtype)


# fp16 overflows: the kernel accumulates x^6 for the cubic term's rms in half
# precision, and bf16 gradients on degenerate 1-2 element rows miss tolerance
# (see ISSUES.md).
@poly_norm.register(
    "liger",
    source="liger_kernel.transformers.functional.liger_poly_norm",
    supports={"hidden": Range(3, 1 << 20)},
)
def poly_norm_liger(
    x: Float32[Tensor, "... hidden"] | BFloat16[Tensor, "... hidden"],
    weight,
    bias,
    eps,
):
    return kernel(x, weight, bias, eps, in_place=False)
