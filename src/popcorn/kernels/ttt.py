import torch
from jaxtyping import Float
from torch import Tensor

from popcorn import Tag, kernel, register_kernel
from popcorn.kernels._utils import upcast


# seq must be a multiple of mini_batch_size; the singleton pool keeps the grid
# on 16-aligned lengths.
@register_kernel(
    test_shapes={"seq": {64}, "head_dim": {32}},
    test_inputs={
        "eta": lambda t: torch.sigmoid(t) * 0.02,
        "w": lambda t: 1 + 0.1 * t,
        "b": lambda t: 0.1 * t,
    },
    tags={Tag.SEQUENCE_MIXER, Tag.LINEAR_ATTENTION},
)
def ttt(
    q: Float[Tensor, "batch seq heads head_dim"],
    k: Float[Tensor, "batch seq heads head_dim"],
    v: Float[Tensor, "batch seq heads head_dim"],
    w: Float[Tensor, "heads head_dim"],
    b: Float[Tensor, "heads head_dim"],
    eta: Float[Tensor, "batch seq heads"],
    eps: float = 1e-6,
    mini_batch_size: int = 16,
) -> Float[Tensor, "batch seq heads head_dim"]:
    r"""TTT-linear: the hidden state is a linear model trained by gradient descent at test time.

    $$W_t = W_{t-1} - \eta_t \nabla_W \big\lVert \mathrm{LN}(k_t W_{t-1}) - (v_t - k_t) \big\rVert^2,
    \qquad o_t \approx \mathrm{LN}(q_t W_t)$$

    [TTT (Sun et al., 2024)](https://arxiv.org/abs/2407.04620)
    """

    def stats(t):
        mean = t.mean(-1, keepdim=True)
        rstd = (t.var(-1, unbiased=False, keepdim=True) + eps).rsqrt()
        return mean, rstd

    dtype = q.dtype
    q, k, v, w, b, eta = map(upcast, (q, k, v, w, b, eta))
    batch, seq, heads, dim = q.shape
    scale = dim**-0.5
    w = w.view(heads, 1, dim)
    b = b.view(heads, 1, dim)
    h = q.new_zeros(batch, heads, dim, dim)
    hb = q.new_zeros(batch, heads, 1, dim)
    lower = q.new_ones(mini_batch_size, mini_batch_size).tril()

    out = []
    for start in range(0, seq, mini_batch_size):
        chunk = slice(start, start + mini_batch_size)
        q_i, k_i, v_i = (t[:, chunk].transpose(1, 2) for t in (q, k, v))
        q_i = q_i * scale
        eta_i = eta[:, chunk].transpose(1, 2).unsqueeze(-1)

        kh = k_i @ h + hb
        mean, rstd = stats(kh)
        kh_hat = (kh - mean) * rstd
        g = (w * kh_hat + b - (v_i - k_i)) * w
        v_new = (dim * g - g.sum(-1, True) - kh_hat * (g * kh_hat).sum(-1, True)) * rstd / dim

        attn = (q_i @ k_i.transpose(-1, -2)) * lower
        o_i = q_i @ h + hb - (eta_i * (attn + lower)) @ v_new
        eta_last = eta_i[:, :, -1:]
        h = h - (eta_last * k_i).transpose(-1, -2) @ v_new
        hb = hb - (eta_last * v_new).sum(-2, keepdim=True)

        mean, rstd = stats(o_i)
        out.append(o_i + (o_i - mean) * rstd * w + b)
    return torch.cat(out, 2).transpose(1, 2).to(dtype)


# forward_only: the backward misses tolerance by ~3x in every dtype; see
# ISSUES.md.
@ttt.register("fla", source="fla.ops.ttt.chunk_ttt_linear", forward_only=True)
def ttt_fla(q, k, v, w, b, eta, eps, mini_batch_size):
    return kernel(q, k, v, w, b, eta[..., None], eps=eps, chunk_size=mini_batch_size)[0]
