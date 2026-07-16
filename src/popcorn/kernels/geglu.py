import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from popcorn import register_kernel


@register_kernel
def geglu(a: Float[Tensor, "... hidden"], b: Float[Tensor, "... hidden"]) -> Float[Tensor, "... hidden"]:
    """GEGLU gating (arXiv:2002.05202): `gelu_tanh(a) * b`."""
    return F.gelu(a, approximate="tanh") * b


geglu.register("liger", source="liger_kernel.transformers.functional.liger_geglu")
