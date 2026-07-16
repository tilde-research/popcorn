from typing import Literal

import torch
import torch.nn.functional as F
from jaxtyping import BFloat16, Float, Float32
from torch import Tensor

from popcorn import Range, kernel, register_kernel
from popcorn.kernels._utils import upcast


@register_kernel(
    test_shapes={"batch": Range(1, 8), "channels": {4}, "channels_out": {4}, "seq": Range(8, 128), "ksize": {5}},
    test_args={"padding": [0, 2]},
)
def multi_token_attention(
    scores: Float[Tensor, "batch channels seq seq"],
    weight: Float[Tensor, "channels_out channels ksize ksize"],
    bias: Float[Tensor, "channels_out"] | None = None,
    padding: int = 0,
    sparse: Literal[False] = False,
) -> Float[Tensor, "batch channels_out seq2 seq2"]:
    """Multi-token attention (arXiv:2504.00927): causal softmax over raw scores,
    a conv2d mixing attention maps across heads and positions, then re-masking
    the future to zero."""
    causal = torch.ones(scores.shape[-2:], dtype=torch.bool, device=scores.device).triu(1)
    probs = upcast(scores).masked_fill(causal, -1e9).softmax(-1).to(scores.dtype)
    out = F.conv2d(probs, weight, bias, padding=padding)
    return out.masked_fill(torch.ones(out.shape[-2:], dtype=torch.bool, device=out.device).triu(1), 0.0)


# fp16 score gradients land just past tolerance.
@multi_token_attention.register("liger", source="liger_kernel.transformers.functional.liger_multi_token_attention")
def multi_token_attention_liger(
    scores: Float32[Tensor, "batch channels seq seq"] | BFloat16[Tensor, "batch channels seq seq"],
    weight,
    bias,
    padding,
    sparse,
):
    return kernel(scores, weight, bias, padding=padding, sparse=sparse)
