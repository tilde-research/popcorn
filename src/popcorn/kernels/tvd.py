from typing import Literal

from jaxtyping import Float, Float32
from torch import Tensor

from popcorn import Range, kernel, register_kernel


@register_kernel(
    test_shapes={"vocab": Range(2, 4096)},
    test_inputs={"p": lambda t: t.softmax(-1), "q": lambda t: t.softmax(-1)},
)
def tvd(
    p: Float[Tensor, "tokens vocab"],
    q: Float[Tensor, "tokens vocab"],
    reduction: Literal["mean", "sum", "batchmean", "none"] = "mean",
) -> Float[Tensor, ""]:
    """Total variation distance between two distributions; `q` is the target
    and treated as a constant, matching the liger kernel."""
    distance = 0.5 * (p - q.detach()).abs()
    if reduction == "none":
        return distance
    total = distance.sum()
    if reduction == "mean":
        return total / p.numel()
    return total / p.shape[0] if reduction == "batchmean" else total


# liger always returns the loss in fp32, so only float32 inputs round-trip.
@tvd.register("liger", source="liger_kernel.transformers.functional.liger_tvd")
def tvd_liger(p: Float32[Tensor, "tokens vocab"], q, reduction):
    return kernel(p, q, reduction=reduction)
