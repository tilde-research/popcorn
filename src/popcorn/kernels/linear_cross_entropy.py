from typing import Literal

import torch.nn.functional as F
from jaxtyping import Float, Float32, Int
from torch import Tensor

from popcorn import Tag, kernel, register_kernel
from popcorn.kernels._utils import upcast


@register_kernel(test_args={"label_smoothing": [0.0, 0.1]}, tags={Tag.LOSS, Tag.LINEAR, Tag.FUSED})
def linear_cross_entropy(
    x: Float[Tensor, "tokens hidden"],
    weight: Float[Tensor, "vocab hidden"],
    labels: Int[Tensor, "tokens"],
    bias: Float[Tensor, "vocab"] | None = None,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    reduction: Literal["mean", "sum"] = "mean",
) -> Float[Tensor, ""]:
    r"""Cross entropy fused with the lm-head projection, never materializing the logits.

    $$\mathcal{L}_t = -\log \operatorname{softmax}(x_t w^\top + b)_{y_t}$$

    [Cut Cross-Entropy (Wijmans et al., 2024)](https://arxiv.org/abs/2411.09009)
    """
    logits = F.linear(upcast(x), upcast(weight), None if bias is None else upcast(bias))
    return F.cross_entropy(
        logits,
        labels,
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
        reduction=reduction,
    )


# fla always returns the loss in fp32, so only float32 inputs round-trip.
@linear_cross_entropy.register("fla", source="fla.modules.fused_linear_cross_entropy.fused_linear_cross_entropy_loss")
def linear_cross_entropy_fla(
    x: Float32[Tensor, "tokens hidden"], weight, labels, bias, ignore_index, label_smoothing, reduction
):
    return kernel(
        x,
        labels,
        weight,
        bias,
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
        reduction=reduction,
    )


# liger always returns the loss in fp32, so only float32 inputs round-trip.
@linear_cross_entropy.register("liger", source="liger_kernel.transformers.functional.liger_fused_linear_cross_entropy")
def linear_cross_entropy_liger(
    x: Float32[Tensor, "tokens hidden"], weight, labels, bias, ignore_index, label_smoothing, reduction
):
    return kernel(
        x,
        weight,
        labels,
        bias=bias,
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
        reduction=reduction,
    )


# quack 0.5.0 cannot serve this op: its gemm accepts only fp16/fp8 activations while the
# loss always returns fp32, so no input dtype round-trips (see ISSUES.md). Earlier quack
# releases took fp32 and were adapted here; restore an adapter if a release aligns again.
