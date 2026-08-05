from jaxtyping import BFloat16, Float, Float16
from torch import Tensor

from popcorn import Tag, kernel, register_kernel
from popcorn.kernels._utils import upcast

# quack also ships a softmax, but it is strictly 2-D and its backward
# miscompiles on plain contiguous inputs; see ISSUES.md.


@register_kernel(tags={Tag.ACTIVATION, Tag.REDUCTION})
def softmax(x: Float[Tensor, "... hidden"]) -> Float[Tensor, "... hidden"]:
    r"""Softmax over the last dimension.

    $$y_i = \frac{e^{x_i}}{\sum_j e^{x_j}}$$
    """
    return upcast(x).softmax(-1).to(x.dtype)


@softmax.register("fla", source="fla.ops.utils.softmax_fwd", forward_only=True)
def softmax_fla(x):
    return kernel(x, dtype=x.dtype)


# fp16 backward on tiny rows lands just past tolerance.
@softmax.register("liger", source="liger_kernel.transformers.functional.liger_softmax")
def softmax_liger(x):
    return kernel(x)


def _transformer_engine_ready(**arguments):
    """The fused kernel requires aligned keys and 32-row launch tiles."""
    x = arguments["x"]
    rows, width = x.numel() // x.shape[-1], x.shape[-1]
    return rows % 32 == 0 and 16 < width < 16_384 and width % 8 == 0


@softmax.register(
    "transformer_engine",
    source="transformer_engine.pytorch.attention.dot_product_attention.softmax.ScaledSoftmax.apply",
    predicate=_transformer_engine_ready,
)
def softmax_transformer_engine(x: Float16[Tensor, "... hidden"] | BFloat16[Tensor, "... hidden"]):
    return kernel(x.reshape(4, 1, -1, x.shape[-1]), 1.0).reshape_as(x)
