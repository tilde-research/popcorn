import torch
from jaxtyping import Float
from torch import Tensor

from popcorn import register_kernel
from popcorn.kernels._utils import upcast


@register_kernel(
    test_shapes={"batch": {1}, "seq": {7, 8}, "heads": {2}, "width": {16}},
    test_args={"chunk_size": [4]},
)
def mean_pooling(
    x: Float[Tensor, "batch seq heads width"],
    chunk_size: int,
) -> Float[Tensor, "batch chunks heads width"]:
    """Mean-pool consecutive, non-overlapping sequence chunks."""
    return torch.stack([upcast(chunk).mean(1) for chunk in x.split(chunk_size, 1)], 1).to(x.dtype)


mean_pooling.register("fla", source="fla.ops.utils.mean_pooling")
