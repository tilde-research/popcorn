import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from popcorn import Range, kernel, register_kernel
from popcorn.kernels._utils import upcast


@register_kernel(
    test_shapes={"batch": Range(1, 4), "seq": Range(2, 64), "heads": {4}, "key_dim": {32}},
    test_inputs={"g": F.logsigmoid, "beta": torch.sigmoid, "lamb": F.softplus},
)
def mesa_net(
    q: Float[Tensor, "batch seq heads key_dim"],
    k: Float[Tensor, "batch seq heads key_dim"],
    v: Float[Tensor, "batch seq heads key_dim"],
    g: Float[Tensor, "batch seq heads"],
    beta: Float[Tensor, "batch seq heads"],
    lamb: Float[Tensor, "heads key_dim"],
) -> Float[Tensor, "batch seq heads key_dim"]:
    """MesaNet (arXiv:2506.05233): each step solves the ridge regression
    (H_kk + diag(lamb)) q* = q against gated key/value statistics."""
    q32, k32, v32, g32, beta32 = map(upcast, (q, k, v, g, beta))
    batch, seq, heads, dim = q.shape
    h_kk = q32.new_zeros(batch, heads, dim, dim)
    h_kv = q32.new_zeros(batch, heads, dim, dim)
    kk_all = q32.new_zeros(batch, seq, heads, dim, dim)
    kv_all = q32.new_zeros(batch, seq, heads, dim, dim)
    for t in range(seq):
        weighted = k32[:, t] * beta32[:, t, :, None]
        decay = g32[:, t, :, None, None].exp()
        h_kk = h_kk * decay + weighted[..., None] * k32[:, t, :, None, :]
        h_kv = h_kv * decay + weighted[..., None] * v32[:, t, :, None, :]
        kk_all[:, t] = h_kk
        kv_all[:, t] = h_kv
    ridge = kk_all + torch.diag_embed(upcast(lamb))[None, None]
    q_star = torch.linalg.solve(ridge, q32)
    return (q_star[..., None] * kv_all).sum(-2).to(q.dtype)


@mesa_net.register("fla", source="fla.ops.mesa_net.chunk_mesa_net")
def mesa_net_fla(q, k, v, g, beta, lamb):
    return kernel(q, k, v, g, beta, lamb)[0]
