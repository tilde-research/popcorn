import torch
from jaxtyping import Float
from torch import Tensor

from popcorn import Tag, kernel, register_kernel
from popcorn.kernels._utils import upcast


@register_kernel(
    test_shapes={"batch": {1}, "first": {9}, "second": {9}, "width": {16}},
    test_args={"chunk_size": [4], "scale": [None, 0.5]},
    tags={Tag.REDUCTION},
)
def chunk_local_cumsum(
    g: Float[Tensor, "batch first second width"],
    chunk_size: int,
    reverse: bool = False,
    scale: float | None = None,
    head_first: bool = False,
) -> Float[Tensor, "batch first second width"]:
    r"""Inclusive cumsum reset at chunk boundaries, in either FLA layout.

    $$y_t = c \sum_{u = \lfloor t/C \rfloor C}^{t} g_u$$
    """
    axis = 2 if head_first else 1
    chunks = []
    for chunk in upcast(g).split(chunk_size, axis):
        if reverse:
            chunk = torch.flip(torch.cumsum(torch.flip(chunk, (axis,)), axis), (axis,))
        else:
            chunk = torch.cumsum(chunk, axis)
        chunks.append(chunk)
    out = torch.cat(chunks, axis)
    if scale is not None:
        out = out * scale
    return out.to(g.dtype)


@chunk_local_cumsum.register(
    "fla",
    source="fla.ops.utils.chunk_local_cumsum",
    forward_only=True,
)
def chunk_local_cumsum_fla(g, chunk_size, reverse, scale, head_first):
    return kernel(
        g,
        chunk_size,
        reverse=reverse,
        scale=scale,
        head_first=head_first,
        output_dtype=g.dtype,
    )
