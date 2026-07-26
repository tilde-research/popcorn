import torch
import torch.nn.functional as F
from jaxtyping import Float, Float32
from torch import Tensor

from popcorn import Tag, kernel, register_kernel
from popcorn.kernels._utils import default_scale, upcast


@register_kernel(
    test_args={"softmax_scale": [None, 0.25]},
    test_inputs={
        "k": lambda t: F.normalize(t, dim=-1),
        "p": lambda t: F.normalize(t, dim=-1),
        "g": F.logsigmoid,
        "beta": torch.sigmoid,
        "initial_state": lambda t: 0.1 * t,
    },
    tags={Tag.SEQUENCE_MIXER, Tag.LINEAR_ATTENTION},
)
def comba_decode(
    q: Float[Tensor, "batch heads key_dim"],
    k: Float[Tensor, "batch heads key_dim"],
    v: Float[Tensor, "batch heads value_dim"],
    p: Float[Tensor, "batch heads key_dim"],
    g: Float[Tensor, "batch heads"],
    beta: Float[Tensor, "batch heads"],
    initial_state: Float[Tensor, "batch heads key_dim value_dim"],
    softmax_scale: float | None = None,
) -> tuple[
    Float[Tensor, "batch heads value_dim"],
    Float32[Tensor, "batch heads key_dim value_dim"],
]:
    r"""One-token Comba update with explicit recurrent state.

    $$S_t = e^{g_t}S_{t-1}
    + \beta_t k_t\big(v_t-S_{t-1}^\top p_t\big)^\top,
    \qquad o_t = c \, q_t^\top S_t$$
    """
    scale = default_scale(softmax_scale, q.shape[-1])
    q32, k32, v32, p32, g32, beta32, state = map(upcast, (q, k, v, p, g, beta, initial_state))
    error = v32 - torch.einsum("bhkv,bhk->bhv", state, p32)
    state = state * g32.exp()[..., None, None]
    update = beta32[..., None] * error
    state = state + k32[..., None] * update[..., None, :]
    output = torch.einsum("bhkv,bhk->bhv", state, q32 * scale)
    return output.to(v.dtype), state.float()


@comba_decode.register(
    "fla",
    source="fla.ops.comba.fused_recurrent_comba",
    forward_only=True,
)
def comba_decode_fla(q, k, v, p, g, beta, initial_state, softmax_scale):
    output, final_state = kernel(
        q.unsqueeze(1),
        k.unsqueeze(1),
        p.unsqueeze(1),
        v.unsqueeze(1),
        g.unsqueeze(1),
        beta=beta.unsqueeze(1),
        scale=softmax_scale,
        initial_state=initial_state,
        output_final_state=True,
    )
    return output.squeeze(1), final_state
