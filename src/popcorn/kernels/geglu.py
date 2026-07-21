import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from popcorn import Tag, register_kernel


@register_kernel(tags={Tag.ACTIVATION, Tag.FEATURE_MIXER, Tag.FUSED})
def geglu(a: Float[Tensor, "... hidden"], b: Float[Tensor, "... hidden"]) -> Float[Tensor, "... hidden"]:
    r"""GEGLU gating with the tanh-approximate GELU.

    $$y = \mathrm{gelu}(a) \odot b$$

    [GLU Variants Improve Transformer (Shazeer, 2020)](https://arxiv.org/abs/2002.05202)
    """
    return F.gelu(a, approximate="tanh") * b


geglu.register("liger", source="liger_kernel.transformers.functional.liger_geglu")
