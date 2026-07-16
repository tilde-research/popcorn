import torch
from jaxtyping import Float
from torch import Tensor

from popcorn import register_kernel


@register_kernel
def dyt(
    x: Float[Tensor, "... hidden"],
    alpha: Float[Tensor, "1"],
    gamma: Float[Tensor, "hidden"],
    beta: Float[Tensor, "hidden"] | None = None,
) -> Float[Tensor, "... hidden"]:
    """Dynamic Tanh (arXiv:2503.10622): a normalization-free elementwise transform."""
    out = gamma * torch.tanh(alpha * x)
    return out if beta is None else out + beta


dyt.register("liger", source="liger_kernel.transformers.functional.liger_dyt")
