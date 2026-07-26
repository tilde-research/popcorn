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
        "g": F.logsigmoid,
        "beta": torch.sigmoid,
        "initial_state": lambda t: 0.1 * t,
    },
    tags={Tag.SEQUENCE_MIXER, Tag.LINEAR_ATTENTION},
)
def kda_decode(
    q: Float[Tensor, "batch heads key_dim"],
    k: Float[Tensor, "batch heads key_dim"],
    v: Float[Tensor, "batch heads value_dim"],
    g: Float[Tensor, "batch heads key_dim"],
    beta: Float[Tensor, "batch heads"],
    initial_state: Float[Tensor, "batch heads key_dim value_dim"],
    softmax_scale: float | None = None,
) -> tuple[
    Float[Tensor, "batch heads value_dim"],
    Float32[Tensor, "batch heads key_dim value_dim"],
]:
    r"""One-token Kimi Delta Attention update with explicit recurrent state.

    $$S_t = \mathrm{diag}\!\big(e^{g_t}\big) S_{t-1}
    + \beta_t k_t \big(v_t - (\mathrm{diag}(e^{g_t}) S_{t-1})^\top k_t\big)^\top,
    \qquad o_t = c \, q_t^\top S_t$$
    """
    scale = default_scale(softmax_scale, q.shape[-1])
    q32, k32, v32, g32, beta32, state = map(upcast, (q, k, v, g, beta, initial_state))
    state = state * g32.exp()[..., None]
    error = v32 - torch.einsum("bhkv,bhk->bhv", state, k32)
    update = beta32[..., None] * k32
    state = state + update[..., None] * error[..., None, :]
    output = torch.einsum("bhkv,bhk->bhv", state, q32 * scale)
    return output.to(v.dtype), state.float()


@kda_decode.register(
    "fla",
    source="fla.ops.kda.fused_recurrent_kda",
    forward_only=True,
)
def kda_decode_fla(q, k, v, g, beta, initial_state, softmax_scale):
    output, final_state = kernel(
        q.unsqueeze(1),
        k.unsqueeze(1),
        v.unsqueeze(1),
        g.unsqueeze(1),
        beta.unsqueeze(1),
        scale=softmax_scale,
        initial_state=initial_state,
        output_final_state=True,
    )
    return output.squeeze(1), final_state
