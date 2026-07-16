from typing import Literal

import torch.nn.functional as F
from jaxtyping import Float, Float32
from torch import Tensor

from popcorn import Range, kernel, register_kernel


@register_kernel(test_shapes={"vocab": Range(2, 4096)})
def linear_kl_div(
    x: Float[Tensor, "tokens hidden"],
    target_x: Float[Tensor, "tokens target_hidden"],
    weight: Float[Tensor, "vocab hidden"],
    target_weight: Float[Tensor, "vocab target_hidden"],
    reduction: Literal["batchmean"] = "batchmean",
) -> Float[Tensor, ""]:
    """KL(target || student) between lm-head outputs, fused with both
    projections; the target branch is a constant, matching fla."""
    log_p = F.linear(x, weight).log_softmax(-1)
    log_q = F.linear(target_x, target_weight).log_softmax(-1).detach()
    return (log_q.exp() * (log_q - log_p)).sum() / x.shape[0]


# fla always returns the loss in fp32, so only float32 inputs round-trip.
@linear_kl_div.register("fla", source="fla.modules.fused_kl_div.fused_kl_div_loss")
def linear_kl_div_fla(x: Float32[Tensor, "tokens hidden"], target_x, weight, target_weight, reduction):
    return kernel(x, target_x, weight, target_weight, reduction=reduction)
