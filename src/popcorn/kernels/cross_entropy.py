from typing import Literal

import torch.nn.functional as F
from jaxtyping import Float, Float32, Int
from torch import Tensor

from popcorn import Tag, kernel, register_kernel
from popcorn.kernels._utils import upcast


@register_kernel(test_args={"label_smoothing": [0.0, 0.1]}, tags={Tag.LOSS})
def cross_entropy(
    logits: Float[Tensor, "tokens vocab"],
    labels: Int[Tensor, "tokens"],
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    reduction: Literal["mean", "sum", "none"] = "mean",
) -> Float[Tensor, "..."]:
    r"""Softmax cross-entropy with ignore-index and label smoothing, matching torch semantics.

    $$\mathcal{L}_t = -\log \operatorname{softmax}(x_t)_{y_t}$$
    """
    return F.cross_entropy(
        upcast(logits),
        labels,
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
        reduction=reduction,
    )


# fla returns unreduced (losses, z_losses) in fp32; z_losses are zero unless
# z-loss is on, so float32 inputs and reduction="none" are the exact subset.
@cross_entropy.register("fla", source="fla.modules.fused_cross_entropy.cross_entropy_loss")
def cross_entropy_fla(
    logits: Float32[Tensor, "tokens vocab"], labels, ignore_index, label_smoothing, reduction: Literal["none"]
):
    return kernel(logits, labels, label_smoothing=label_smoothing, ignore_index=ignore_index)[0]


@cross_entropy.register("liger", source="liger_kernel.transformers.functional.liger_cross_entropy")
def cross_entropy_liger(logits, labels, ignore_index, label_smoothing, reduction):
    return kernel(logits, labels, ignore_index=ignore_index, label_smoothing=label_smoothing, reduction=reduction).float()


# quack supports unsmoothed loss only and returns fp32, so only float32 inputs round-trip.
@cross_entropy.register("quack", source="quack.cross_entropy")
def cross_entropy_quack(
    logits: Float32[Tensor, "tokens vocab"], labels, ignore_index, label_smoothing: Literal[0.0], reduction
):
    return kernel(logits, labels, ignore_index=ignore_index, reduction=reduction)


# unsloth hardcodes ignore_index=-100, no smoothing, and returns unreduced fp32
# losses (the backward writes the logit gradient into the logits buffer), so
# float32 inputs and reduction="none" are the exact subset.
@cross_entropy.register("unsloth", source="unsloth.kernels.cross_entropy_loss.Fast_CrossEntropyLoss.apply")
def cross_entropy_unsloth(
    logits: Float32[Tensor, "tokens vocab"],
    labels,
    ignore_index: Literal[-100],
    label_smoothing: Literal[0.0],
    reduction: Literal["none"],
):
    return kernel(logits, labels)
