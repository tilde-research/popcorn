import torch
from jaxtyping import Float
from torch import Tensor

from popcorn import Tag, kernel, register_kernel
from popcorn.kernels._utils import upcast


@register_kernel(
    test_shapes={"batch": {2}, "rows": {3}, "hidden": {16, 33}},
    test_args={"scale": [None, 0.5]},
    tags={Tag.REDUCTION},
)
def logsumexp(
    x: Float[Tensor, "batch rows hidden"],
    scale: float | None = None,
) -> Float[Tensor, "batch rows"]:
    r"""Log-sum-exp over the last dimension, optionally after scaling.

    $$y = \log \sum_i e^{s \, x_i}$$
    """
    dtype = x.dtype
    x = upcast(x)
    if scale is not None:
        x = x * scale
    return torch.logsumexp(x, -1).to(dtype)


@logsumexp.register("fla", source="fla.ops.utils.logsumexp_fwd", forward_only=True)
def logsumexp_fla(x, scale):
    return kernel(x, scale=scale, dtype=x.dtype)
