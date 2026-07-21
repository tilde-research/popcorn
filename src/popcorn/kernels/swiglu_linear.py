import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from popcorn import Tag, register_kernel


@register_kernel(tags={Tag.ACTIVATION, Tag.LINEAR, Tag.FUSED, Tag.FEATURE_MIXER})
def swiglu_linear(
    x: Float[Tensor, "... hidden"],
    y: Float[Tensor, "... hidden"],
    weight: Float[Tensor, "out_features hidden"],
    bias: Float[Tensor, "out_features"] | None = None,
) -> Float[Tensor, "... out_features"]:
    r"""SwiGLU gating fused with a linear projection.

    $$z = \big(\mathrm{silu}(x) \odot y\big) \, w^\top + b$$

    [GLU (Dauphin et al., 2016)](https://arxiv.org/abs/1612.08083),
    [GLU Variants Improve Transformer (Shazeer, 2020)](https://arxiv.org/abs/2002.05202)
    """
    return F.linear(F.silu(x) * y, weight, bias)


swiglu_linear.register("fla", source="fla.modules.activations.swiglu_linear")
