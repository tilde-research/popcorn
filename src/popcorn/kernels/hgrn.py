import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from popcorn import Tag, kernel, register_kernel
from popcorn.kernels._utils import upcast


@register_kernel(
    test_inputs={"g": F.logsigmoid},
    tags={Tag.SEQUENCE_MIXER, Tag.LINEAR_ATTENTION},
)
def hgrn(
    x: Float[Tensor, "batch seq hidden"],
    g: Float[Tensor, "batch seq hidden"],
) -> Float[Tensor, "batch seq hidden"]:
    r"""The HGRN elementwise gated recurrence.

    $$h_t = e^{g_t} \odot h_{t-1} + x_t$$

    [HGRN (Qin et al., 2023)](https://arxiv.org/abs/2311.04823)
    """
    dtype = x.dtype
    x, g = map(upcast, (x, g))
    h = torch.zeros_like(x[:, 0])
    outs = []
    for t in range(x.shape[1]):
        h = g[:, t].exp() * h + x[:, t]
        outs.append(h)
    return torch.stack(outs, 1).to(dtype)


@hgrn.register("fla", source="fla.ops.hgrn.chunk_hgrn")
def hgrn_fla(x, g):
    return kernel(x, g)[0]
