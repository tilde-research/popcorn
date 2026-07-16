import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from popcorn import Range, kernel, register_kernel
from popcorn.kernels._utils import upcast


@register_kernel(
    test_shapes={"batch": Range(1, 8), "seq": Range(2, 256), "hidden": {64}},
    test_inputs={"g": F.logsigmoid},
)
def hgrn(
    x: Float[Tensor, "batch seq hidden"],
    g: Float[Tensor, "batch seq hidden"],
) -> Float[Tensor, "batch seq hidden"]:
    """HGRN recurrence (arXiv:2311.04823): h_t = exp(g_t) * h_{t-1} + x_t."""
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
