import torch
import torch.nn.functional as F
from jaxtyping import Float, Float32
from torch import Tensor

from popcorn import Range, Tag, kernel, register_kernel
from popcorn.kernels._utils import default_scale, upcast


@register_kernel(
    test_args={"softmax_scale": [None, 0.25]},
    test_inputs={
        "k": lambda t: F.normalize(t, dim=-1),
        "v": lambda t: F.normalize(t, dim=-1),
        "gv": F.logsigmoid,
        "beta": torch.sigmoid,
        "initial_state": lambda t: 0.1 * t,
    },
    tags={Tag.SEQUENCE_MIXER, Tag.LINEAR_ATTENTION},
)
def gated_oja_rule_decode(
    q: Float[Tensor, "batch heads key_dim"],
    k: Float[Tensor, "batch heads key_dim"],
    v: Float[Tensor, "batch heads value_dim"],
    gv: Float[Tensor, "batch heads value_dim"],
    beta: Float[Tensor, "batch heads"],
    initial_state: Float[Tensor, "batch heads key_dim value_dim"],
    softmax_scale: float | None = None,
) -> tuple[
    Float[Tensor, "batch heads value_dim"],
    Float32[Tensor, "batch heads key_dim value_dim"],
]:
    r"""One-token gated Oja update with explicit recurrent state.

    $$S_t = S_{t-1}\mathrm{diag}(e^{g^v_t})
    + \beta_t\big(k_t-S_{t-1}v_t\big)v_t^\top,
    \qquad o_t = c \, q_t^\top S_t$$
    """
    scale = default_scale(softmax_scale, q.shape[-1])
    q32, k32, v32, gv32, beta32, state = map(upcast, (q, k, v, gv, beta, initial_state))
    state = state * gv32.exp()[:, :, None, :]
    write = beta32[..., None] * (k32 - torch.einsum("bhkv,bhv->bhk", state, v32))
    state = state + write[..., None] * v32[..., None, :]
    output = torch.einsum("bhkv,bhk->bhv", state, q32 * scale)
    return output.to(v.dtype), state.float()


@gated_oja_rule_decode.register(
    "fla",
    source="fla.ops.gated_oja_rule.fused_recurrent_gated_oja_rule",
    supports={"value_dim": Range(1, 128)},
    forward_only=True,
)
def gated_oja_rule_decode_fla(q, k, v, gv, beta, initial_state, softmax_scale):
    output, final_state = kernel(
        q.unsqueeze(1),
        k.unsqueeze(1),
        v.unsqueeze(1),
        gv=gv.unsqueeze(1),
        beta=beta.unsqueeze(1),
        scale=softmax_scale,
        initial_state=initial_state,
        output_final_state=True,
    )
    return output.squeeze(1), final_state
