import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from popcorn import Tag, kernel, register_kernel
from popcorn.impls import cuda_toolkit


@register_kernel(tags={Tag.ACTIVATION, Tag.FEATURE_MIXER, Tag.FUSED})
def swiglu(a: Float[Tensor, "... hidden"], b: Float[Tensor, "... hidden"]) -> Float[Tensor, "... hidden"]:
    r"""SwiGLU gating.

    $$y = \mathrm{silu}(a) \odot b$$

    [GLU (Dauphin et al., 2016)](https://arxiv.org/abs/1612.08083),
    [GLU Variants Improve Transformer (Shazeer, 2020)](https://arxiv.org/abs/2002.05202)
    """
    return F.silu(a) * b


swiglu.register("fla", source="fla.modules.activations.swiglu")
swiglu.register("liger", source="liger_kernel.transformers.functional.liger_swiglu")
swiglu.register("popcorn", source="popcorn.impls.swiglu_cu.swiglu", predicate=cuda_toolkit)


# unsloth ships only the fused forward (its backward lives in the LoRA MLP
# path); the kernel is elementwise but expects a 3-D tensor.
@swiglu.register("unsloth", source="unsloth.kernels.swiglu.swiglu_fg_kernel", forward_only=True)
def swiglu_unsloth(a, b):
    hidden = a.shape[-1]
    return kernel(a.reshape(1, -1, hidden), b.reshape(1, -1, hidden)).reshape(a.shape)
