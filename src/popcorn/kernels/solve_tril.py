import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from popcorn import kernel, register_kernel
from popcorn.kernels._utils import upcast


def _strict_lower(A, dims, _generator):
    rows = torch.arange(dims["seq"]).remainder(dims["block"])
    cols = torch.arange(dims["block"])
    mask = cols[None, :] < rows[:, None]
    return A * mask[None, :, None, :] * 0.05


@register_kernel(
    test_shapes={"batch": {1}, "seq": {16, 23}, "heads": {2}, "block": {16}},
    test_inputs={"A": _strict_lower},
)
def solve_tril(A: Float[Tensor, "batch seq heads block"]) -> Float[Tensor, "batch seq heads block"]:
    """Invert each block of a chunked unit-lower-triangular matrix."""
    block_size = A.shape[-1]
    chunks = []
    for chunk in upcast(A).split(block_size, 1):
        length = chunk.shape[1]
        matrix = chunk[..., :length].permute(0, 2, 1, 3).tril(-1)
        identity = torch.eye(length, dtype=matrix.dtype, device=A.device)
        inverse = torch.linalg.inv(identity + matrix)
        chunks.append(F.pad(inverse, (0, block_size - length)).permute(0, 2, 1, 3))
    return torch.cat(chunks, 1).to(A.dtype)


@solve_tril.register(
    "fla",
    source="fla.ops.utils.solve_tril",
    supports={"block": {16, 32, 64}},
    forward_only=True,
)
def solve_tril_fla(A):
    return kernel(A, output_dtype=A.dtype)
