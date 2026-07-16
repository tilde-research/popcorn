import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from popcorn import Range, kernel, register_kernel
from popcorn.kernels._utils import upcast


def _rank_one_state(t):
    vector = F.normalize(t[..., 0], dim=-1)
    return vector[..., None] * vector[..., None, :]


def _log_decay(t):
    return (0.95 + 0.04 * torch.sigmoid(t)).log()


@register_kernel(
    test_shapes={
        "batch": Range(1, 4),
        "heads": {2},
        "key_dim": {16},
        "value_dim": {16},
    },
    test_args={"max_cg_iterations": [1, 3]},
    test_inputs={
        "q": lambda t: 0.1 * t,
        "k": lambda t: F.normalize(t, dim=-1),
        "v": lambda t: 0.1 * t,
        "g": _log_decay,
        "lamb": lambda t: 0.25 + 0.75 * torch.sigmoid(t),
        "beta": torch.sigmoid,
        "prev_h_kk": _rank_one_state,
        "prev_h_kv": lambda t: 0.1 * t,
    },
)
def mesa_net_decode(
    q: Float[Tensor, "batch heads key_dim"],
    k: Float[Tensor, "batch heads key_dim"],
    v: Float[Tensor, "batch heads value_dim"],
    g: Float[Tensor, "batch heads"],
    lamb: Float[Tensor, "heads key_dim"],
    beta: Float[Tensor, "batch heads"],
    prev_h_kk: Float[Tensor, "batch heads key_dim key_dim"],
    prev_h_kv: Float[Tensor, "batch heads key_dim value_dim"],
    max_cg_iterations: int = 30,
) -> tuple[
    Float[Tensor, "batch heads value_dim"],
    Float[Tensor, "batch heads key_dim key_dim"],
    Float[Tensor, "batch heads key_dim value_dim"],
]:
    """One MesaNet decode step with explicit recurrent statistics and CG solve."""
    q32, k32, v32, g32, lamb32, beta32, h_kk, h_kv = map(
        upcast,
        (q, k, v, g, lamb, beta, prev_h_kk, prev_h_kv),
    )
    weighted_k = k32 * beta32[..., None]
    decay = g32.exp()[..., None, None]
    h_kk = h_kk * decay + weighted_k[..., None] * k32[..., None, :]
    h_kv = h_kv * decay + weighted_k[..., None] * v32[..., None, :]
    regularizer = lamb32[None]

    def apply_system(x):
        return torch.einsum("bhij,bhi->bhj", h_kk, x) + regularizer * x

    x = q32 / (h_kk.diagonal(dim1=-2, dim2=-1) + regularizer + 1e-5)
    residual = q32 - apply_system(x)
    direction = residual
    squared = residual.square().sum(-1)
    for _ in range(max_cg_iterations):
        applied = apply_system(direction)
        alpha = squared / ((direction * applied).sum(-1) + 1e-5)
        x = x + alpha[..., None] * direction
        residual = residual - alpha[..., None] * applied
        next_squared = residual.square().sum(-1)
        direction = residual + (next_squared / (squared + 1e-5))[..., None] * direction
        squared = next_squared

    out = torch.einsum("bhkv,bhk->bhv", h_kv, x)
    return out.to(q.dtype), h_kk.to(prev_h_kk.dtype), h_kv.to(prev_h_kv.dtype)


@mesa_net_decode.register(
    "fla",
    source="fla.ops.mesa_net.mesa_net_decoding_one_step",
    supports={"key_dim": Range(1, 128), "value_dim": Range(1, 128)},
    forward_only=True,
)
def mesa_net_decode_fla(
    q,
    k,
    v,
    g,
    lamb,
    beta,
    prev_h_kk,
    prev_h_kv,
    max_cg_iterations,
):
    return kernel(
        q,
        k,
        v,
        g,
        lamb,
        beta,
        prev_h_kk,
        prev_h_kv,
        max_CG_iteration=max_cg_iterations,
    )
