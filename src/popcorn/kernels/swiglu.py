import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from popcorn import register_kernel
from popcorn.impls import cuda_toolkit


@register_kernel
def swiglu(a: Float[Tensor, "... hidden"], b: Float[Tensor, "... hidden"]) -> Float[Tensor, "... hidden"]:
    """SwiGLU gating (arXiv:2002.05202): `silu(a) * b`."""
    return F.silu(a) * b


swiglu.register("fla", source="fla.modules.activations.swiglu")
swiglu.register("liger", source="liger_kernel.transformers.functional.liger_swiglu")
swiglu.register("popcorn", source="popcorn.impls.swiglu_cu.swiglu", predicate=cuda_toolkit)
