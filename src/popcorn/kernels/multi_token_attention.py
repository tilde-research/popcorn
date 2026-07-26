from typing import Literal

import torch
import torch.nn.functional as F
from jaxtyping import BFloat16, Float, Float32
from torch import Tensor

from popcorn import Tag, kernel, register_kernel
from popcorn.kernels._utils import upcast


@register_kernel(
    test_args={"padding": [0, 2]},
    tags={Tag.SEQUENCE_MIXER, Tag.ATTENTION, Tag.FUSED},
)
def multi_token_attention(
    scores: Float[Tensor, "batch channels seq seq"],
    weight: Float[Tensor, "out_channels channels kernel_size kernel_size"],
    bias: Float[Tensor, "out_channels"] | None = None,
    padding: int = 0,
    sparse: Literal[False] = False,
) -> Float[Tensor, "batch out_channels seq2 seq2"]:
    r"""Causal softmax over raw scores, a conv2d mixing attention maps, then re-masking the future.

    $$y = \mathrm{mask}_0\!\Big(\mathrm{conv2d}\big(\operatorname{softmax}(\mathrm{mask}_{-\infty}(s)),\, w, b\big)\Big)$$

    [Multi-Token Attention (Golovneva et al., 2025)](https://arxiv.org/abs/2504.00927)
    """
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
