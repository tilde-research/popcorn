import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from popcorn import Tag, kernel, register_kernel


@register_kernel(tags={Tag.ACTIVATION, Tag.FEATURE_MIXER, Tag.FUSED})
def geglu(a: Float[Tensor, "... hidden"], b: Float[Tensor, "... hidden"]) -> Float[Tensor, "... hidden"]:
    r"""GEGLU gating with the tanh-approximate GELU.

    $$y = \mathrm{gelu}(a) \odot b$$

    [GLU (Dauphin et al., 2016)](https://arxiv.org/abs/1612.08083),
    [GLU Variants Improve Transformer (Shazeer, 2020)](https://arxiv.org/abs/2002.05202)
    """
    return F.gelu(a, approximate="tanh") * b


geglu.register("liger", source="liger_kernel.transformers.functional.liger_geglu")


# unsloth ships only the fused forward for the tanh-approximate GELU; the kernel
# is elementwise but expects a 3-D tensor.
@geglu.register("unsloth", source="unsloth.kernels.geglu.geglu_approx_forward_kernel", forward_only=True)
def geglu_unsloth(a, b):
    hidden = a.shape[-1]
    return kernel(a.reshape(1, -1, hidden), b.reshape(1, -1, hidden)).reshape(a.shape)
