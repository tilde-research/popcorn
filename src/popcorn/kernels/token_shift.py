import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from popcorn import Tag, register_kernel


@register_kernel(tags={Tag.SEQUENCE_MIXER})
def token_shift(x: Float[Tensor, "batch seq hidden"]) -> Float[Tensor, "batch seq hidden"]:
    r"""RWKV token shift: the previous token's features minus the current ones.

    $$y_t = x_{t-1} - x_t, \qquad x_{-1} = 0$$

    [RWKV (Peng et al., 2023)](https://arxiv.org/abs/2305.13048)
    """
    return F.pad(x, (0, 0, 1, -1)) - x


token_shift.register("fla", source="fla.modules.token_shift.token_shift")
