import torch
import torch.nn.functional as F
from jaxtyping import Float, Float32
from torch import Tensor

from popcorn import Range, Tag, kernel, register_kernel
from popcorn.kernels._utils import default_scale, upcast


def _unit(t):
    return F.normalize(torch.ones_like(t), dim=-1)


@register_kernel(
    test_shapes={
        "batch": Range(1, 4),
        "seq": Range(2, 64),
        "heads": {2},
        "key_dim": {32},
        "value_dim": {32},
    },
    test_args={"softmax_scale": [None, 0.25]},
    test_inputs={"a": _unit, "b": lambda t: -_unit(t), "initial_state": lambda t: 0.1 * t},
    tags={Tag.SEQUENCE_MIXER, Tag.LINEAR_ATTENTION},
)
def iplr_delta_rule(
    q: Float[Tensor, "batch seq heads key_dim"],
    k: Float[Tensor, "batch seq heads key_dim"],
    v: Float[Tensor, "batch seq heads value_dim"],
    a: Float[Tensor, "batch seq heads key_dim"],
    b: Float[Tensor, "batch seq heads key_dim"],
    initial_state: Float[Tensor, "batch heads key_dim value_dim"],
    softmax_scale: float | None = None,
) -> tuple[
    Float[Tensor, "batch seq heads value_dim"],
    Float32[Tensor, "batch heads key_dim value_dim"],
]:
    r"""Identity-plus-low-rank delta rule with explicit recurrent state.

    $$S_t = \big(I + b_t a_t^\top\big) S_{t-1} + k_t v_t^\top, \qquad o_t = c \, q_t^\top S_t$$

    [DeltaNet (Yang et al., 2024)](https://arxiv.org/abs/2406.06484)
    """
    scale = default_scale(softmax_scale, q.shape[-1])
    q32, k32, v32, a32, b32, state = map(upcast, (q, k, v, a, b, initial_state))
    outs = []
    for t in range(q.shape[1]):
        projected = torch.einsum("bhkv,bhk->bhv", state, a32[:, t])
        state = state + b32[:, t][..., None] * projected[..., None, :]
        state = state + k32[:, t][..., None] * v32[:, t][..., None, :]
        outs.append(torch.einsum("bhkv,bhk->bhv", state, q32[:, t] * scale))
    return torch.stack(outs, 1).to(v.dtype), state.float()


@iplr_delta_rule.register(
    "fla",
    source="fla.ops.generalized_delta_rule.fused_recurrent_iplr_delta_rule",
    supports={"key_dim": Range(1, 128)},
)
def iplr_delta_rule_fla(
    q,
    k,
    v,
    a,
    b,
    initial_state,
    softmax_scale,
):
    return kernel(
        q,
        k,
        v,
        a,
        b,
        scale=softmax_scale,
        initial_state=initial_state,
        output_final_state=True,
    )
