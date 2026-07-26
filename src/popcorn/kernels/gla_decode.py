import torch
import torch.nn.functional as F
from jaxtyping import Float, Float32
from torch import Tensor

from popcorn import Tag, kernel, register_kernel
from popcorn.kernels._utils import default_scale, upcast


@register_kernel(
    test_args={"softmax_scale": [None, 0.25]},
    test_inputs={"g": F.logsigmoid, "initial_state": lambda t: 0.1 * t},
    tags={Tag.SEQUENCE_MIXER, Tag.LINEAR_ATTENTION},
)
def gla_decode(
    q: Float[Tensor, "batch heads key_dim"],
    k: Float[Tensor, "batch heads key_dim"],
    v: Float[Tensor, "batch heads value_dim"],
    g: Float[Tensor, "batch heads key_dim"],
    initial_state: Float[Tensor, "batch heads key_dim value_dim"],
    softmax_scale: float | None = None,
) -> tuple[
    Float[Tensor, "batch heads value_dim"],
    Float32[Tensor, "batch heads key_dim value_dim"],
]:
    r"""One-token gated linear attention with explicit recurrent state.

    $$S_t = \mathrm{diag}\!\big(e^{g_t}\big) S_{t-1} + k_t v_t^\top,
    \qquad o_t = c \, q_t^\top S_t$$
    """
    scale = default_scale(softmax_scale, q.shape[-1])
    q32, k32, v32, g32, state = map(upcast, (q, k, v, g, initial_state))
    state = g32.exp()[..., None] * state + k32[..., None] * v32[..., None, :]
    output = torch.einsum("bhkv,bhk->bhv", state, q32 * scale)
    return output.to(v.dtype), state.float()


@gla_decode.register(
    "fla",
    source="fla.ops.gla.fused_recurrent_gla",
)
def gla_decode_fla(q, k, v, g, initial_state, softmax_scale):
    output, final_state = kernel(
        q.unsqueeze(1),
        k.unsqueeze(1),
        v.unsqueeze(1),
        gk=g.unsqueeze(1),
        scale=softmax_scale,
        initial_state=initial_state,
        output_final_state=True,
    )
    return output.squeeze(1), final_state
