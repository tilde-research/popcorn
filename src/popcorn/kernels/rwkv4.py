import torch
import torch.nn.functional as F
from jaxtyping import Float, Float32
from torch import Tensor

from popcorn import Range, Tag, kernel, register_kernel
from popcorn.kernels._utils import upcast


def _state(s):
    """(alpha, beta, eps) with beta, the wkv denominator, kept positive."""
    alpha, beta, eps = s.unbind(1)
    return torch.stack((alpha, F.softplus(beta), eps), 1)


# The effective decay is -exp(w); the running numerator, denominator, and
# their shared log offset travel in `state`, matching fla.
@register_kernel(
    test_shapes={"batch": Range(1, 8), "seq": Range(2, 128), "chans": Range(8, 1024)},
    test_inputs={"state": _state},
    tags={Tag.SEQUENCE_MIXER, Tag.LINEAR_ATTENTION},
)
def rwkv4(
    w: Float[Tensor, "chans"],
    u: Float[Tensor, "chans"],
    k: Float[Tensor, "batch seq chans"],
    v: Float[Tensor, "batch seq chans"],
    state: Float[Tensor, "batch 3 1 chans"],
) -> tuple[Float[Tensor, "batch seq chans"], Float[Tensor, "batch 3 1 chans"]]:
    r"""RWKV-4 wkv: a per-channel EMA of values weighted by exp(k), with a current-token bonus.

    $$o_t = \frac{a_{t-1} + e^{u + k_t} v_t}{b_{t-1} + e^{u + k_t}},
    \qquad a_t = e^{-e^{w}} a_{t-1} + e^{k_t} v_t, \quad b_t = e^{-e^{w}} b_{t-1} + e^{k_t}$$

    [RWKV (Peng et al., 2023)](https://arxiv.org/abs/2305.13048)
    """
    dtype = k.dtype
    w, u, k, v, state = map(upcast, (w, u, k, v, state))
    decay = -w.exp()
    bonus = u
    alpha, beta, eps = state.squeeze(2).unbind(1)
    outs = []
    for t in range(k.shape[1]):
        kt, vt = k[:, t], v[:, t]
        ukt = bonus + kt
        tau = torch.maximum(ukt, eps)
        up, cur = (eps - tau).exp(), (ukt - tau).exp()
        outs.append((up * alpha + cur * vt) / (up * beta + cur))
        shifted = decay + eps
        eps = torch.maximum(shifted, kt)
        up, cur = (shifted - eps).exp(), (kt - eps).exp()
        alpha = up * alpha + cur * vt
        beta = up * beta + cur
    wkv = torch.stack(outs, 1).to(dtype)
    return wkv, torch.stack((alpha, beta, eps), 1).unsqueeze(2).to(dtype)


# float32 only: the kernel saves the per-step state trajectory in the input
# dtype and 16-bit quantization of the log offset wrecks the backward; see
# ISSUES.md.
@rwkv4.register("fla", source="fla.ops.rwkv4.fused_recurrent_rwkv4")
def rwkv4_fla(w: Float32[Tensor, "chans"], u, k, v, state):
    return kernel(w, u, k, v, state)
