import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from popcorn import Tag, register_kernel


# Torch-only for now; the old tilelang backend is deferred until tilelang
# ships in the environment.
@register_kernel(tags={Tag.FEATURE_MIXER, Tag.ACTIVATION, Tag.LINEAR, Tag.FUSED})
def swiglu_mlp(
    x: Float[Tensor, "... hidden"],
    gate_weight: Float[Tensor, "intermediate hidden"],
    up_weight: Float[Tensor, "intermediate hidden"],
    down_weight: Float[Tensor, "hidden intermediate"],
) -> Float[Tensor, "... hidden"]:
    r"""The LLaMA MLP: gate and up projections, SwiGLU, then the down projection.

    $$y = \big(\mathrm{silu}(x w_g^\top) \odot x w_u^\top\big) \, w_d^\top$$

    [GLU Variants Improve Transformer (Shazeer, 2020)](https://arxiv.org/abs/2002.05202)
    """
    return F.linear(F.silu(F.linear(x, gate_weight)) * F.linear(x, up_weight), down_weight)
