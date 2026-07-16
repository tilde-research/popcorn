import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from popcorn import kernel, register_kernel
from popcorn.kernels._utils import upcast


@register_kernel(
    test_shapes={"batch": {1}, "seq": {9}, "heads": {2}, "width": {16}},
    test_args={"chunk_size": [4], "scale": [None, 0.5], "lower_bound": [None, -5.0]},
    test_inputs={"A_log": lambda x: x * 0.1, "dt_bias": lambda x: x * 0.1},
)
def kda_gate_cumsum(
    g: Float[Tensor, "batch seq heads width"],
    A_log: Float[Tensor, "heads"],
    chunk_size: int,
    scale: float | None = None,
    dt_bias: Float[Tensor, "heads width"] | None = None,
    lower_bound: float | None = None,
) -> Float[Tensor, "batch seq heads width"]:
    """Apply the KDA gate transform, then a chunk-local inclusive cumsum."""
    g32 = upcast(g)
    if dt_bias is not None:
        g32 = g32 + upcast(dt_bias)
    rate = upcast(A_log).exp()[None, None, :, None]
    if lower_bound is None:
        gate = -rate * F.softplus(g32)
    else:
        gate = lower_bound * torch.sigmoid(rate * g32)
    out = torch.cat([chunk.cumsum(1) for chunk in gate.split(chunk_size, 1)], 1)
    if scale is not None:
        out = out * scale
    return out.to(g.dtype)


@kda_gate_cumsum.register(
    "fla",
    source="fla.ops.kda.gate.kda_gate_chunk_cumsum",
    forward_only=True,
)
def kda_gate_cumsum_fla(g, A_log, chunk_size, scale, dt_bias, lower_bound):
    return kernel(
        g,
        A_log,
        chunk_size,
        scale=scale,
        dt_bias=None if dt_bias is None else dt_bias.flatten(),
        output_dtype=g.dtype,
        lower_bound=lower_bound,
    )
