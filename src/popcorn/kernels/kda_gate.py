import torch
import torch.nn.functional as F
from jaxtyping import Float, Float32
from torch import Tensor

from popcorn import Range, kernel, register_kernel
from popcorn.kernels._utils import upcast


def _fla_supported(g, A_log, dt_bias, lower_bound):
    del A_log, lower_bound
    return dt_bias is None or dt_bias.numel() == g.shape[-2] * g.shape[-1]


# gate_dim must equal heads * key_dim; singleton pools keep the grid valid.
@register_kernel(
    test_shapes={
        "batch": Range(1, 4),
        "seq": {32},
        "heads": {4},
        "key_dim": {32},
        "gate_dim": {128},
    },
    test_args={"lower_bound": [None, -5.0]},
    test_inputs={
        "g": lambda t: 2 * t,
        "A_log": lambda t: (F.softplus(t) + 1).log(),
        "dt_bias": lambda t: 0.1 * t,
    },
)
def kda_gate(
    g: Float[Tensor, "batch seq heads key_dim"],
    A_log: Float[Tensor, "heads"],
    dt_bias: Float[Tensor, "gate_dim"] | None = None,
    lower_bound: float | None = None,
) -> Float32[Tensor, "batch seq heads key_dim"]:
    """KDA's per-dimension log forget gate.

    The default branch is ``-exp(A_log) * softplus(g + dt_bias)``. With a
    lower bound, it is ``lower_bound * sigmoid(exp(A_log) * (g + dt_bias))``.
    """
    heads, key_dim = g.shape[-2:]
    gate = upcast(g)
    if dt_bias is not None:
        gate = gate + upcast(dt_bias).view(heads, key_dim)
    rate = upcast(A_log).view(heads, 1).exp()
    if lower_bound is None:
        out = -rate * F.softplus(gate)
    else:
        out = lower_bound * torch.sigmoid(rate * gate)
    return out.float()


@kda_gate.register(
    "fla",
    source="fla.ops.kda.gate.fused_kda_gate",
    supports={"key_dim": Range(1, 128)},
    predicate=_fla_supported,
)
def kda_gate_fla(g, A_log, dt_bias, lower_bound):
    return kernel(
        g,
        A_log,
        dt_bias,
        lower_bound=lower_bound,
        output_dtype=torch.float32,
    )
