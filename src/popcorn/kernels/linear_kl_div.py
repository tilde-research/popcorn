from typing import Literal

import torch.nn.functional as F
from jaxtyping import Float, Float32
from torch import Tensor

from popcorn import Tag, kernel, register_kernel
from popcorn.kernels._utils import upcast


# The target branch is a constant, matching fla.
@register_kernel(tags={Tag.LOSS, Tag.LINEAR, Tag.FUSED})
def linear_kl_div(
    x: Float[Tensor, "tokens hidden"],
    target_x: Float[Tensor, "tokens target_hidden"],
    weight: Float[Tensor, "vocab hidden"],
    target_weight: Float[Tensor, "vocab target_hidden"],
    reduction: Literal["batchmean"] = "batchmean",
) -> Float[Tensor, ""]:
    r"""KL divergence between student and constant target lm-head outputs, fused with both projections.

    $$\mathcal{L} = \frac{1}{T} \sum_t \mathrm{KL}\!\Big(
    \operatorname{softmax}\!\big(u_t w_u^\top\big) \,\Big\|\, \operatorname{softmax}\!\big(x_t w^\top\big)\Big)$$

    [Liger Kernel (Hsu et al., 2024)](https://arxiv.org/abs/2410.10989)
    """
    log_p = F.linear(upcast(x), upcast(weight)).log_softmax(-1)
    log_q = F.linear(upcast(target_x), upcast(target_weight)).log_softmax(-1).detach()
    return (log_q.exp() * (log_q - log_p)).sum(-1).mean()


# fla always returns the loss in fp32, so only float32 inputs round-trip.
@linear_kl_div.register("fla", source="fla.modules.fused_kl_div.fused_kl_div_loss")
def linear_kl_div_fla(x: Float32[Tensor, "tokens hidden"], target_x, weight, target_weight, reduction):
    return kernel(x, target_x, weight, target_weight, reduction=reduction)
