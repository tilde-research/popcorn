from jaxtyping import Float
from torch import Tensor

from popcorn import Range, kernel, register_kernel
from popcorn.kernels._utils import upcast

# quack also ships a softmax, but it is strictly 2-D and its backward
# miscompiles on plain contiguous inputs; see ISSUES.md.


@register_kernel
def softmax(x: Float[Tensor, "... hidden"]) -> Float[Tensor, "... hidden"]:
    """Softmax over the last dimension."""
    return upcast(x).softmax(-1).to(x.dtype)


@softmax.register("fla", source="fla.ops.utils.softmax_fwd", forward_only=True)
def softmax_fla(x):
    return kernel(x, dtype=x.dtype)


# fp16 backward on tiny rows lands just past tolerance.
@softmax.register(
    "liger", source="liger_kernel.transformers.functional.liger_softmax", supports={"hidden": Range(16, 1 << 20)}
)
def softmax_liger(x):
    return kernel(x)
