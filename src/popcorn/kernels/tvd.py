from typing import Literal

from jaxtyping import Float, Float32
from torch import Tensor

from popcorn import Tag, kernel, register_kernel
from popcorn.kernels._utils import upcast


# q is the target and treated as a constant, matching liger.
@register_kernel(
    test_inputs={"p": lambda t: t.softmax(-1), "q": lambda t: t.softmax(-1)},
    tags={Tag.LOSS},
)
def tvd(
    p: Float[Tensor, "tokens vocab"],
    q: Float[Tensor, "tokens vocab"],
    reduction: Literal["mean", "sum", "batchmean", "none"] = "mean",
) -> Float[Tensor, ""]:
    r"""Total variation distance to a constant target distribution.

    $$\mathcal{L} = \tfrac{1}{2} \lVert p - q \rVert_1$$
    """
    distance = 0.5 * (p - q.detach()).abs()
    if reduction == "none":
        return distance
    total = upcast(distance).sum()
    if reduction == "mean":
        return total / p.numel()
    return total / p.shape[0] if reduction == "batchmean" else total


# liger always returns the loss in fp32, so only float32 inputs round-trip.
@tvd.register("liger", source="liger_kernel.transformers.functional.liger_tvd")
def tvd_liger(p: Float32[Tensor, "tokens vocab"], q, reduction):
    return kernel(p, q, reduction=reduction)
