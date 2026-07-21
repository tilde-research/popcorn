from typing import Literal

import torch.nn.functional as F
from jaxtyping import Float, Float32, Int
from torch import Tensor

from popcorn import Range, Tag, kernel, register_kernel


@register_kernel(
    test_shapes={"vocab": Range(2, 4096)}, test_args={"label_smoothing": [0.0, 0.1]}, tags={Tag.LOSS, Tag.LINEAR, Tag.FUSED}
)
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
    """
    return F.cross_entropy(
        F.linear(x, weight, bias),
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


def _aligned(**arguments):
    """The chunked kernel asserts tokens % 8 == 0."""
    return arguments["x"].shape[0] % 8 == 0


# forward_only: quack returns the loss in fp32 (so only float32 inputs
# round-trip) while its backward gemm accepts only fp16/fp8 activations.
@linear_cross_entropy.register(
    "quack",
    source="quack.linear_cross_entropy.chunked_linear_cross_entropy",
    predicate=_aligned,
    forward_only=True,
)
def linear_cross_entropy_quack(
    x: Float32[Tensor, "tokens hidden"],
    weight,
    labels,
    bias: Literal[None],
    ignore_index,
    label_smoothing: Literal[0.0],
    reduction,
):
    return kernel(x, weight, labels, ignore_index=ignore_index, reduction=reduction)
