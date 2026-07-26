import torch
import torch.nn.functional as F
from jaxtyping import Float, Float32
from torch import Tensor

from popcorn import Tag, kernel, register_kernel
from popcorn.kernels._utils import default_scale, upcast


@register_kernel(
    test_args={"softmax_scale": [None, 0.25]},
    test_inputs={"w": F.logsigmoid, "initial_state": lambda t: 0.1 * t},
    tags={Tag.SEQUENCE_MIXER, Tag.LINEAR_ATTENTION},
)
def rwkv6_decode(
    r: Float[Tensor, "batch heads key_dim"],
    k: Float[Tensor, "batch heads key_dim"],
    v: Float[Tensor, "batch heads value_dim"],
    w: Float[Tensor, "batch heads key_dim"],
    u: Float[Tensor, "heads key_dim"],
    initial_state: Float[Tensor, "batch heads key_dim value_dim"],
    softmax_scale: float | None = None,
) -> tuple[
    Float[Tensor, "batch heads value_dim"],
    Float32[Tensor, "batch heads key_dim value_dim"],
]:
    r"""One-token RWKV-6 update with explicit recurrent state.

    $$o_t = c \, r_t^\top\big(S_{t-1}+\mathrm{diag}(u)k_tv_t^\top\big),
    \qquad S_t = \mathrm{diag}(e^{w_t})S_{t-1}+k_tv_t^\top$$
    """
    scale = default_scale(softmax_scale, r.shape[-1])
    r32, k32, v32, w32, u32, state = map(upcast, (r, k, v, w, u, initial_state))
    kv = k32[..., None] * v32[..., None, :]
    boosted = state + u32[None, :, :, None] * kv
    output = torch.einsum("bhkv,bhk->bhv", boosted, r32 * scale)
    state = state * w32.exp()[..., None] + kv
    return output.to(v.dtype), state.float()


@rwkv6_decode.register(
    "fla",
    source="fla.ops.rwkv6.fused_recurrent_rwkv6",
    forward_only=True,
)
def rwkv6_decode_fla(r, k, v, w, u, initial_state, softmax_scale):
    output, final_state = kernel(
        r.unsqueeze(1),
        k.unsqueeze(1),
        v.unsqueeze(1),
        w.unsqueeze(1),
        u,
        scale=softmax_scale,
        initial_state=initial_state,
        output_final_state=True,
    )
    return output.squeeze(1), final_state
