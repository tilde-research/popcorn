import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from popcorn import register_kernel


@register_kernel
def swiglu_linear(
    x: Float[Tensor, "... hidden"],
    y: Float[Tensor, "... hidden"],
    weight: Float[Tensor, "out_features hidden"],
    bias: Float[Tensor, "out_features"] | None = None,
) -> Float[Tensor, "... out_features"]:
    """SwiGLU followed by a linear projection: `linear(silu(x) * y)`."""
    return F.linear(F.silu(x) * y, weight, bias)


swiglu_linear.register("fla", source="fla.modules.activations.swiglu_linear")
