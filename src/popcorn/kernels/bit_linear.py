import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from popcorn import Tag, register_kernel
from popcorn.kernels._quant import activation_quant, weight_quant
from popcorn.kernels._utils import rms


# The normalization eps is fixed at 1e-6: fla's `bit_linear` accepts an `eps`
# argument but drops it (see ISSUES.md), so the reference does not expose one.
@register_kernel(tags={Tag.LINEAR, Tag.QUANTIZED, Tag.NORMALIZATION, Tag.FUSED})
def bit_linear(
    x: Float[Tensor, "... hidden"],
    weight: Float[Tensor, "out_features hidden"],
    bias: Float[Tensor, "out_features"] | None = None,
    norm_weight: Float[Tensor, "hidden"] | None = None,
    norm_bias: Float[Tensor, "hidden"] | None = None,
) -> Float[Tensor, "... out_features"]:
    r"""BitLinear: RMS norm, int8 activation quant, ternary weight quant, then the projection.

    $$y = Q_{\mathrm{int8}}\!\big(\mathrm{RMSNorm}(x)\big) \, Q_{\{-1,0,1\}}(w)^\top + b$$

    [BitNet (Wang et al., 2023)](https://arxiv.org/abs/2310.11453),
    [BitNet b1.58 (Ma et al., 2024)](https://arxiv.org/abs/2402.17764)
    """
    h = rms(x, 1e-6)
    if norm_weight is not None:
        h = h * norm_weight
    if norm_bias is not None:
        h = h + norm_bias
    return F.linear(activation_quant(h), weight_quant(weight), bias)


bit_linear.register("fla", source="fla.modules.fused_bitlinear.bit_linear")
