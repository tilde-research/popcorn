import torch
from jaxtyping import Float
from torch import Tensor

from popcorn import Range, kernel, register_kernel
from popcorn.kernels._utils import default_scale, upcast


@register_kernel(
    test_shapes={"batch": Range(1, 8), "seq": Range(2, 256), "heads": {4}, "key_dim": {64}, "value_dim": {64}},
    test_args={"layer_idx": [0, 1], "num_layers": [2], "softmax_scale": [None, 0.25]},
)
def lightning_attn(
    q: Float[Tensor, "batch seq heads key_dim"],
    k: Float[Tensor, "batch seq heads key_dim"],
    v: Float[Tensor, "batch seq heads value_dim"],
    layer_idx: int,
    num_layers: int,
    softmax_scale: float | None = None,
) -> Float[Tensor, "batch seq heads value_dim"]:
    """Lightning attention (arXiv:2405.17381): linear attention whose per-head
    decay slope shrinks with layer depth, matching fla's derivation."""
    dtype = q.dtype
    q, k, v = map(upcast, (q, k, v))
    heads, seq = q.shape[2], q.shape[1]
    scale = default_scale(softmax_scale, q.shape[-1])
    slope = -(8 / heads * (1 - layer_idx / num_layers)) * torch.arange(heads, device=q.device, dtype=q.dtype)
    n = torch.arange(seq, device=q.device, dtype=q.dtype)
    decay = ((n[:, None] - n[None, :]) * slope[:, None, None]).tril().exp().tril()
    scores = torch.einsum("bqhk,bjhk->bhqj", q * scale, k)
    return torch.einsum("bhqj,bjhv->bqhv", scores * decay, v).to(dtype)


@lightning_attn.register("fla", source="fla.ops.lightning_attn.chunk_lightning_attn")
def lightning_attn_fla(q, k, v, layer_idx, num_layers, softmax_scale):
    return kernel(q, k, v, layer_idx, num_layers, scale=softmax_scale)[0]
