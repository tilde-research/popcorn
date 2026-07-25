import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from popcorn import Tag, kernel, register_kernel
from popcorn.kernels._utils import upcast


@register_kernel(
    test_inputs={"g": F.logsigmoid, "beta": torch.sigmoid, "lamb": F.softplus},
    tags={Tag.SEQUENCE_MIXER, Tag.LINEAR_ATTENTION},
)
def mesa_net(
    q: Float[Tensor, "batch seq heads key_dim"],
    k: Float[Tensor, "batch seq heads key_dim"],
    v: Float[Tensor, "batch seq heads key_dim"],
    g: Float[Tensor, "batch seq heads"],
    beta: Float[Tensor, "batch seq heads"],
    lamb: Float[Tensor, "heads key_dim"],
) -> Float[Tensor, "batch seq heads key_dim"]:
    r"""MesaNet: each step solves a ridge regression against gated key/value statistics.

    $$H_t = e^{g_t} H_{t-1} + \beta_t k_t k_t^\top, \quad
    G_t = e^{g_t} G_{t-1} + \beta_t k_t v_t^\top, \quad
    o_t = G_t^\top \big(H_t + \mathrm{diag}(\lambda)\big)^{-1} q_t$$

    [Uncovering mesa-optimization algorithms in Transformers (von Oswald et al., 2023)](https://arxiv.org/abs/2309.05858),
    [MesaNet (von Oswald et al., 2025)](https://arxiv.org/abs/2506.05233)
    """
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
