import torch
from jaxtyping import Float
from torch import Tensor

from popcorn import Tag, kernel, register_kernel
from popcorn.kernels._utils import upcast


@register_kernel(
    test_args={"scale": [None, 0.5]},
    tags={Tag.REDUCTION},
)
def chunk_global_cumsum(
    s: Float[Tensor, "batch first second width"],
    reverse: bool = False,
    scale: float | None = None,
    head_first: bool = False,
) -> Float[Tensor, "batch first second width"]:
    r"""Inclusive sequence-wide cumsum in either FLA layout, optionally reversed and scaled.

    $$y_t = c \sum_{u \le t} s_u$$
    """
    axis = 2 if head_first else 1
    s32 = upcast(s)
    if reverse:
        out = torch.flip(torch.cumsum(torch.flip(s32, (axis,)), axis), (axis,))
    else:
        out = torch.cumsum(s32, axis)
    if scale is not None:
        out = out * scale
    return out.to(s.dtype)


@chunk_global_cumsum.register(
    "fla",
    source="fla.ops.utils.chunk_global_cumsum",
    forward_only=True,
)
def chunk_global_cumsum_fla(s, reverse, scale, head_first):
    return kernel(
        s,
        reverse=reverse,
        scale=scale,
        head_first=head_first,
        output_dtype=s.dtype,
    )
