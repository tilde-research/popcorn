import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from popcorn import Tag, kernel, register_kernel
from popcorn.kernels._quant import activation_quant, weight_quant
from popcorn.kernels._utils import rms


@register_kernel(test_args={"eps": [1e-6, 1e-5]}, tags={Tag.NORMALIZATION, Tag.LINEAR, Tag.QUANTIZED, Tag.FUSED})
def rms_norm_linear_quant(
    x: Float[Tensor, "... hidden"],
    norm_weight: Float[Tensor, "hidden"],
    norm_bias: Float[Tensor, "hidden"] | None,
    linear_weight: Float[Tensor, "out_features hidden"],
    linear_bias: Float[Tensor, "out_features"] | None = None,
    eps: float = 1e-6,
) -> Float[Tensor, "... out_features"]:
    r"""BitNet-style linear on an RMS-normed input: int8/ternary fake quant around the matmul.

    $$y = Q_{\mathrm{int8}}\!\big(\mathrm{RMSNorm}_{w_n, b_n}(x)\big) \, Q_{\{-1,0,1\}}(w_l)^\top + b_l$$

    [BitNet (Wang et al., 2023)](https://arxiv.org/abs/2310.11453),
    [BitNet b1.58 (Ma et al., 2024)](https://arxiv.org/abs/2402.17764)
    """
    h = rms(x, eps)
    h = h * norm_weight if norm_bias is None else h * norm_weight + norm_bias
    return F.linear(activation_quant(h), weight_quant(linear_weight), linear_bias)


@rms_norm_linear_quant.register("fla", source="fla.modules.fused_bitlinear.rms_norm_linear_quant")
def rms_norm_linear_quant_fla(x, norm_weight, norm_bias, linear_weight, linear_bias, eps):
    return kernel(x, norm_weight, norm_bias, linear_weight, linear_bias, eps=eps)
