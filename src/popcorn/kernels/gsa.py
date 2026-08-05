import torch
import torch.nn.functional as F
from jaxtyping import BFloat16, Float, Float32
from torch import Tensor

from popcorn import Tag, kernel, register_kernel
from popcorn.kernels._utils import default_scale, upcast


@register_kernel(
    test_args={"softmax_scale": [None, 0.25]},
    test_inputs={"g": F.logsigmoid},
    tags={Tag.SEQUENCE_MIXER, Tag.LINEAR_ATTENTION},
)
def gsa(
    q: Float[Tensor, "batch seq q_heads key_dim"],
    k: Float[Tensor, "batch seq kv_heads key_dim"],
    v: Float[Tensor, "batch seq kv_heads value_dim"],
    s: Float[Tensor, "batch seq kv_heads slots"],
    g: Float[Tensor, "batch seq kv_heads slots"],
    softmax_scale: float | None = None,
) -> Float[Tensor, "batch seq q_heads value_dim"]:
    r"""Gated slot attention: two chained gated linear-attention passes through softmaxed slots.

    $$K_t = K_{t-1} \, \mathrm{diag}\!\big(e^{g_t}\big) + k_t s_t^\top, \quad
    V_t = \mathrm{diag}\!\big(e^{g_t}\big) V_{t-1} + s_t v_t^\top, \quad
    o_t = V_t^\top \operatorname{softmax}\!\big(K_t^\top q_t\big)$$

    [Gated Slot Attention (Zhang et al., 2024)](https://arxiv.org/abs/2409.07146)
    """
    scale = default_scale(softmax_scale, q.shape[-1])
    groups = q.shape[2] // k.shape[2]
    k32, v32, s32, gate = (upcast(t).repeat_interleave(groups, 2) for t in (k, v, s, g))
    q32 = upcast(q) * scale

    hk = q32.new_zeros(q.shape[0], q.shape[2], q.shape[3], s.shape[3])
    ok = []
    for t in range(q.shape[1]):
        hk = hk * gate[:, t].exp()[:, :, None, :] + k32[:, t, :, :, None] * s32[:, t, :, None, :]
        ok.append(torch.einsum("bhkm,bhk->bhm", hk, q32[:, t]))
    qv = torch.stack(ok, 1).softmax(-1)

    hv = q32.new_zeros(q.shape[0], q.shape[2], s.shape[3], v.shape[3])
    ov = []
    for t in range(q.shape[1]):
        hv = hv * gate[:, t].exp()[:, :, :, None] + s32[:, t, :, :, None] * v32[:, t, :, None, :]
        ov.append(torch.einsum("bhmv,bhm->bhv", hv, qv[:, t]))
    return torch.stack(ov, 1).to(q.dtype)


def _no_gqa(**arguments):
    """fla's backward breaks under grouped kv heads; see ISSUES.md."""
    return arguments["q"].shape[2] == arguments["k"].shape[2]


# float16 gradients through the two chained passes land just past tolerance;
# fp32 and bf16 hold.
@gsa.register("fla", source="fla.ops.gsa.chunk_gsa", predicate=_no_gqa, forward_only=True)
def gsa_fla(
    q: Float32[Tensor, "batch seq q_heads key_dim"] | BFloat16[Tensor, "batch seq q_heads key_dim"],
    k,
    v,
    s,
    g,
    softmax_scale,
):
    return kernel(q, k, v, s, g, scale=softmax_scale)[0]
