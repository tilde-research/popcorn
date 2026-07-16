import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from popcorn import register_kernel


@register_kernel
def swiglu_mlp(
    x: Float[Tensor, "... hidden"],
    gate_weight: Float[Tensor, "intermediate hidden"],
    up_weight: Float[Tensor, "intermediate hidden"],
    down_weight: Float[Tensor, "hidden intermediate"],
) -> Float[Tensor, "... hidden"]:
    """The LLaMA MLP: down(silu(gate(x)) * up(x)). Torch-only for now; the old
    tilelang backend is deferred until tilelang ships in the environment."""
    return F.linear(F.silu(F.linear(x, gate_weight)) * F.linear(x, up_weight), down_weight)
