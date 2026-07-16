from typing import Literal

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from popcorn import register_kernel
from popcorn.kernels._utils import upcast


@register_kernel(test_shapes={"rows": {17}, "inner": {16}, "cols": {19}})
def matmul(
    a: Float[Tensor, "rows inner"],
    b: Float[Tensor, "inner cols"],
    activation: Literal["", "leaky_relu", "relu", "sigmoid", "tanh"] = "",
) -> Float[Tensor, "rows cols"]:
    """Matrix product with FLA's optional fused activation."""
    out = upcast(a) @ upcast(b)
    if activation == "leaky_relu":
        out = F.leaky_relu(out, negative_slope=0.01)
    elif activation == "relu":
        out = F.relu(out)
    elif activation == "sigmoid":
        out = torch.sigmoid(out)
    elif activation == "tanh":
        out = torch.tanh(out)
    return out.to(a.dtype)


matmul.register("fla", source="fla.ops.utils.matmul", forward_only=True)
