import torch
from jaxtyping import BFloat16, Float, Float32
from torch import Tensor

from popcorn import Range, kernel, register_kernel
from popcorn.kernels._utils import default_scale, upcast


@register_kernel(
    test_shapes={"batch": Range(1, 8), "heads": {4}, "seq": Range(2, 128), "head_dim": {64}},
    test_args={"kernel_size": [3, 7], "dilation": [1, 2], "softmax_scale": [None, 0.25]},
)
def neighborhood_attn(
    q: Float[Tensor, "batch heads seq head_dim"],
    k: Float[Tensor, "batch heads seq head_dim"],
    v: Float[Tensor, "batch heads seq head_dim"],
    kernel_size: int = 7,
    dilation: int = 1,
    softmax_scale: float | None = None,
) -> Float[Tensor, "batch heads seq head_dim"]:
    """Neighborhood attention (arXiv:2204.07143): each query attends to keys
    within `kernel_size // 2 * dilation` positions, at multiples of `dilation`.
    The window clamps at sequence boundaries (liger convention, unlike NATTEN)."""
    scale = default_scale(softmax_scale, q.shape[-1])
    t = q.shape[2]
    i = torch.arange(t, device=q.device)[:, None]
    j = torch.arange(t, device=q.device)[None, :]
    half = kernel_size // 2 * dilation
    valid = (j >= i - half) & (j <= i + half)
    if dilation > 1:
        valid &= (j - i) % dilation == 0
    dtype = q.dtype
    q, k, v = map(upcast, (q, k, v))
    scores = (q @ k.transpose(-1, -2) * scale).masked_fill(~valid, -torch.inf)
    return (scores.softmax(-1) @ v).to(dtype)


# fp16 key gradients land just past tolerance.
@neighborhood_attn.register("liger", source="liger_kernel.transformers.functional.liger_fused_neighborhood_attention")
def neighborhood_attn_liger(
    q: Float32[Tensor, "batch heads seq head_dim"] | BFloat16[Tensor, "batch heads seq head_dim"],
    k,
    v,
    kernel_size,
    dilation,
    softmax_scale,
):
    return kernel(q, k, v, kernel_size, dilation, softmax_scale)
