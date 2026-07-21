import torch
from jaxtyping import Float, Float16, Float32
from torch import Tensor

from popcorn import Tag, kernel, register_kernel


@register_kernel(tags={Tag.ACTIVATION, Tag.REDUCTION})
def sparsemax(x: Float[Tensor, "... hidden"], dim: int = -1) -> Float[Tensor, "... hidden"]:
    r"""Sparse softmax: the Euclidean projection onto the probability simplex.

    $$y = \operatorname*{arg\,min}_{p \in \Delta}\; \lVert p - x \rVert^2 = \max(x - \tau(x),\, 0)$$

    [Sparsemax (Martins & Astudillo, 2016)](https://arxiv.org/abs/1602.02068)
    """
    sorted_x, _ = torch.sort(x, dim=dim, descending=True)
    cumulative = sorted_x.cumsum(dim) - 1
    shape = [1] * x.dim()
    shape[dim] = -1
    k = torch.arange(1, x.size(dim) + 1, device=x.device, dtype=x.dtype).view(shape)
    support = ((k * sorted_x) > cumulative).sum(dim=dim, keepdim=True).clamp(min=1)
    tau = cumulative.gather(dim, support - 1) / support.to(x.dtype)
    return torch.clamp(x - tau, min=0)


# bfloat16 ties break the support-set selection differently than torch,
# blowing up the (piecewise) gradients.
@sparsemax.register("liger", source="liger_kernel.transformers.functional.liger_sparsemax")
def sparsemax_liger(x: Float32[Tensor, "... hidden"] | Float16[Tensor, "... hidden"], dim):
    return kernel(x, dim)
