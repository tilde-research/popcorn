import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from popcorn import Tag, register_kernel
from popcorn.kernels._utils import upcast


def _fan_in(tensor):
    return tensor / tensor.shape[-1] ** 0.5


# Torch-only for now; the old tilelang backend is deferred until tilelang
# ships in the environment.
@register_kernel(
    test_inputs={
        "gate_weight": _fan_in,
        "up_weight": _fan_in,
        "down_weight": _fan_in,
    },
    tags={Tag.FEATURE_MIXER, Tag.ACTIVATION, Tag.LINEAR, Tag.FUSED},
)
def swiglu_mlp(
    x: Float[Tensor, "... hidden"],
    gate_weight: Float[Tensor, "intermediate hidden"],
    up_weight: Float[Tensor, "intermediate hidden"],
    down_weight: Float[Tensor, "hidden intermediate"],
) -> Float[Tensor, "... hidden"]:
    r"""The LLaMA MLP: gate and up projections, SwiGLU, then the down projection.

    $$y = \big(\mathrm{silu}(x w_g^\top) \odot x w_u^\top\big) \, w_d^\top$$

    [GLU (Dauphin et al., 2016)](https://arxiv.org/abs/1612.08083),
    [GLU Variants Improve Transformer (Shazeer, 2020)](https://arxiv.org/abs/2002.05202),
    [LLaMA (Touvron et al., 2023)](https://arxiv.org/abs/2302.13971)
    """
    x = upcast(x)
    gate = F.linear(x, upcast(gate_weight))
    value = F.linear(x, upcast(up_weight))
    return F.linear(F.silu(gate) * value, upcast(down_weight))
