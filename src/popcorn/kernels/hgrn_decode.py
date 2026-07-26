import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from popcorn import Tag, kernel, register_kernel
from popcorn.kernels._utils import upcast


@register_kernel(
    test_shapes={"hidden": {64}},
    test_inputs={"g": F.logsigmoid, "initial_state": lambda t: 0.1 * t},
    tags={Tag.SEQUENCE_MIXER, Tag.LINEAR_ATTENTION},
)
def hgrn_decode(
    x: Float[Tensor, "batch hidden"],
    g: Float[Tensor, "batch hidden"],
    initial_state: Float[Tensor, "batch hidden"],
) -> tuple[Float[Tensor, "batch hidden"], Float[Tensor, "batch hidden"]]:
    r"""One-token HGRN recurrence with explicit state.

    $$h_t = e^{g_t} \odot h_{t-1} + x_t$$
    """
    state = upcast(g).exp() * upcast(initial_state) + upcast(x)
    state = state.to(x.dtype)
    return state, state


@hgrn_decode.register(
    "fla",
    source="fla.ops.hgrn.fused_recurrent_hgrn",
)
def hgrn_decode_fla(x, g, initial_state):
    output, final_state = kernel(
        x.unsqueeze(1),
        g.unsqueeze(1),
        initial_state=initial_state,
        output_final_state=True,
    )
    return output.squeeze(1), final_state
