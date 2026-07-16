from typing import Literal

from jaxtyping import Float, Float32
from torch import Tensor

from popcorn import Range, kernel, register_kernel


@register_kernel(
    test_shapes={"vocab": Range(2, 4096)},
    test_inputs={"x": lambda t: t.log_softmax(-1), "target": lambda t: t.softmax(-1)},
)
def kl_div(
    x: Float[Tensor, "tokens vocab"],
    target: Float[Tensor, "tokens vocab"],
    reduction: Literal["mean", "sum", "batchmean", "none"] = "mean",
    log_target: bool = False,
    eps: float = 1e-10,
) -> Float[Tensor, "..."]:
    """KL(target || x) with `x` in log-space and `target` in probability space
    (log space if `log_target`); `target` is a constant, matching liger."""
    target = target.detach()
    loss = target.exp() * (target - x) if log_target else target * (target.clamp(min=eps).log() - x)
    if reduction == "none":
        return loss
    if reduction == "batchmean":
        return loss.sum() / x.shape[0]
    return loss.mean() if reduction == "mean" else loss.sum()


# liger always returns the loss in fp32, so only float32 inputs round-trip.
@kl_div.register("liger", source="liger_kernel.transformers.functional.liger_kl_div")
def kl_div_liger(x: Float32[Tensor, "tokens vocab"], target, reduction, log_target, eps):
    return kernel(x, target, reduction=reduction, log_target=log_target, eps=eps)
