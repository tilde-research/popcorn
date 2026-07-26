from typing import Literal

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from popcorn import Tag, kernel, register_kernel
from popcorn.kernels._utils import upcast


def _contiguous_cache(cache, **_):
    return cache.is_contiguous()


@register_kernel(
    test_shapes={"hidden": {128}, "kernel_size": {3, 4}},
    test_args={"activation": [None, "silu"]},
    tags={Tag.SEQUENCE_MIXER, Tag.FUSED},
)
def causal_conv1d_update(
    x: Float[Tensor, "batch hidden"],
    cache: Float[Tensor, "batch hidden kernel_size"],
    weight: Float[Tensor, "hidden kernel_size"],
    bias: Float[Tensor, "hidden"] | None = None,
    residual: Float[Tensor, "batch hidden"] | None = None,
    activation: Literal["silu"] | None = None,
) -> tuple[Float[Tensor, "batch hidden"], Float[Tensor, "batch hidden kernel_size"]]:
    r"""One-token depthwise causal convolution that updates and returns the same cache tensor.

    $$C'_t = [C_{t-1, 1:}, x_t], \qquad y_t = \sum_j C'_{t,j} w_j + b$$
    """
    tensors = (x, cache, weight, bias, residual)
    if torch.is_grad_enabled() and any(tensor is not None and tensor.requires_grad for tensor in tensors):
        raise RuntimeError("causal_conv1d_update is inference-only; inputs must not require gradients")
    cache.copy_(torch.cat((cache[..., 1:], x.unsqueeze(-1)), dim=-1))
    output = torch.einsum("bhk,hk->bh", upcast(cache), upcast(weight))
    if bias is not None:
        output = output + upcast(bias)
    if activation == "silu":
        output = F.silu(output)
    if residual is not None:
        output = output + upcast(residual)
    return output.to(x.dtype), cache


@causal_conv1d_update.register(
    "fla",
    source="fla.modules.conv.triton.causal_conv1d_update",
    predicate=_contiguous_cache,
    forward_only=True,
)
def causal_conv1d_update_fla(x, cache, weight, bias, residual, activation):
    return kernel(
        x=x,
        cache=cache,
        residual=residual,
        weight=weight,
        bias=bias,
        activation=activation,
    )
