import torch
import torch.nn.functional as F
from jaxtyping import BFloat16, Float, Float16
from torch import Tensor

from popcorn import Tag, kernel, register_kernel
from popcorn.kernels._utils import default_scale, upcast


# The optional log forget gate `g` is added before scaling, i.e.
# softmax((qk + gate) * scale), following the fla kernel. `w` should be
# unit-norm, `beta` in [0, 2].
@register_kernel(
    test_args={"softmax_scale": [None, 0.25]},
    test_inputs={"w": lambda t: F.normalize(t, dim=-1), "beta": lambda t: 2 * torch.sigmoid(t), "g": F.logsigmoid},
    tags={Tag.SEQUENCE_MIXER, Tag.ATTENTION, Tag.POSITIONAL},
)
def path_attn(
    q: Float[Tensor, "batch seq q_heads head_dim"],
    k: Float[Tensor, "batch seq kv_heads head_dim"],
    v: Float[Tensor, "batch seq kv_heads head_dim"],
    w: Float[Tensor, "batch seq kv_heads head_dim"],
    beta: Float[Tensor, "batch seq kv_heads"],
    g: Float[Tensor, "batch seq q_heads"] | None = None,
    softmax_scale: float | None = None,
) -> Float[Tensor, "batch seq q_heads head_dim"]:
    r"""Causal attention where keys are position-encoded by accumulated Householder reflections.

    $$s_{ts} = q_t \big(H_t \cdots H_{s+1} k_s\big)^\top,
    \qquad H_t = I - \beta_t w_t w_t^\top$$

    [PaTH Attention (Yang et al., 2025)](https://arxiv.org/abs/2505.16381)
    """
    seq, scale = q.shape[1], default_scale(softmax_scale, q.shape[-1])
    groups = q.shape[2] // k.shape[2]
    q32 = upcast(q)
    k32, v32, w32 = (upcast(t).repeat_interleave(groups, 2) for t in (k, v, w))
    beta32 = upcast(beta).repeat_interleave(groups, 2)
    before = torch.arange(seq, device=q.device).view(1, -1, 1, 1)
    keys, rows = k32, []
    for t in range(seq):
        dots = (keys * w32[:, t, None]).sum(-1, keepdim=True)
        keys = keys - torch.where(before < t, beta32[:, t, None, :, None] * dots * w32[:, t, None], 0.0)
        rows.append(torch.einsum("bshd,bhd->bhs", keys, q32[:, t]))
    scores = torch.stack(rows, 2)
    if g is not None:
        decay = upcast(g).cumsum(1).transpose(1, 2)
        scores = scores + decay[..., :, None] - decay[..., None, :]
    scores = scores * scale
    mask = torch.ones(seq, seq, dtype=torch.bool, device=q.device).tril()
    weights = scores.masked_fill(~mask, -torch.inf).softmax(-1)
    return torch.einsum("bhqk,bkhd->bqhd", weights, v32).to(q.dtype)


# forward_only: the cumprod-householder backward kernel does not compile on
# current triton (constexpr/do_not_specialize conflict), see ISSUES.md.
@path_attn.register("fla", source="fla.ops.path_attn.parallel_path_attn", forward_only=True)
def path_attn_fla(
    q: Float16[Tensor, "batch seq q_heads head_dim"] | BFloat16[Tensor, "batch seq q_heads head_dim"],
    k,
    v,
    w,
    beta,
    g,
    softmax_scale,
):
    # the kernel asserts w/beta/g arrive in float32 whatever q/k/v are
    return kernel(q, k, v, w.float(), beta.float(), g=None if g is None else g.float(), scale=softmax_scale)[0]
