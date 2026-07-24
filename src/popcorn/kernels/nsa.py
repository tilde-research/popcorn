import functools
import importlib.util

import torch
import torch.nn.functional as F
from jaxtyping import BFloat16, Float
from torch import Tensor

from popcorn import Range, Tag, kernel, register_kernel
from popcorn.kernels._utils import default_scale, upcast


@functools.cache
def _nsa_deps():
    return all(importlib.util.find_spec(module) for module in ("fla", "einops", "triton"))


# fla's sliding-window branch calls flash-attn 2; the vendored kernel uses fa3.
def _fla_ready(window_size, **_):
    return _nsa_deps() and (window_size == 0 or bool(importlib.util.find_spec("flash_attn")))


def _popcorn_ready(window_size, **_):
    return _nsa_deps() and (window_size == 0 or bool(importlib.util.find_spec("flash_attn_interface")))


def _pool_blocks(x, block_size):
    """Mean over non-overlapping seq blocks; the trailing partial block
    averages over the positions that exist (fla `mean_pooling`)."""
    batch, seq = x.shape[:2]
    blocks = -(-seq // block_size)
    padded = F.pad(x, (0, 0) * (x.dim() - 2) + (0, blocks * block_size - seq))
    counts = torch.arange(blocks, device=x.device).mul(-block_size).add(seq).clamp(max=block_size)
    sums = padded.view(batch, blocks, block_size, *x.shape[2:]).sum(2)
    return sums / counts.view(1, blocks, *([1] * (x.dim() - 2)))


def _masked_softmax(scores, valid):
    scores = scores.masked_fill(~valid, -torch.inf)
    peak = scores.amax(-1, keepdim=True)
    peak = torch.where(torch.isfinite(peak), peak, 0.0)
    p = torch.exp(scores - peak)
    return p, p.sum(-1, keepdim=True), peak


# The tested grid keeps ceil(seq / block_size) <= block_count, so every causal
# block is selected and the case is deterministic: with more blocks than slots
# the top-k pick is discrete, and near-ties resolve differently between a
# kernel and any reference, so parity is only defined given the same selection.
#
# Compression attends to mean-pooled kv blocks, each visible once complete;
# selection attends token-causally inside the `block_count` blocks ranked by
# the compression softmax (block 0 and the two most recent are always kept,
# ranking is gradient-free); the sliding window covers the trailing
# `window_size` positions (branch skipped when 0, `g_swa` unused). Kernels
# require grouping `q_heads` a multiple of `16 * kv_heads`, and `block_count`
# at most half of `block_size` (fla `parallel_nsa`).
@register_kernel(
    test_shapes={"seq": Range(2, 512), "q_heads": {16}, "kv_heads": {1}},
    test_args={"block_count": [16], "block_size": [32], "window_size": [0, 64], "softmax_scale": [None, 0.25]},
    tags={Tag.SEQUENCE_MIXER, Tag.ATTENTION, Tag.FUSED},
)
def nsa(
    q: Float[Tensor, "batch seq q_heads head_dim"],
    k: Float[Tensor, "batch seq kv_heads head_dim"],
    v: Float[Tensor, "batch seq kv_heads head_dim"],
    g_cmp: Float[Tensor, "batch seq q_heads"],
    g_slc: Float[Tensor, "batch seq q_heads"],
    g_swa: Float[Tensor, "batch seq q_heads"],
    block_count: int = 16,
    block_size: int = 64,
    window_size: int = 0,
    softmax_scale: float | None = None,
) -> Float[Tensor, "batch seq q_heads head_dim"]:
    r"""Native sparse attention: gated compression, selection, and sliding-window branches.

    $$y = g_{\mathrm{cmp}} \odot \mathrm{Attn}(q, \tilde{k}, \tilde{v})
    + g_{\mathrm{slc}} \odot \mathrm{Attn}_{\mathcal{S}}(q, k, v)
    + g_{\mathrm{swa}} \odot \mathrm{Attn}_{W}(q, k, v)$$

    [Native Sparse Attention (Yuan et al., 2025)](https://arxiv.org/abs/2502.11089),
    [nsa-release](https://github.com/tilde-research/nsa-release)
    """
    scale = default_scale(softmax_scale, q.shape[-1])
    q32, k32, v32 = map(upcast, (q, k, v))
    batch, seq, q_heads, _ = q.shape
    groups = q_heads // k.shape[2]
    blocks = -(-seq // block_size)
    t_idx = torch.arange(seq, device=q.device)[:, None]
    j_idx = torch.arange(blocks, device=q.device)[None, :]

    # Compression: causal over complete blocks; rows before the first complete
    # block are all-masked and produce zeros (lse pinned to 0, as in the kernels).
    k_cmp, v_cmp = _pool_blocks(k32, block_size), _pool_blocks(v32, block_size)
    complete = j_idx < (t_idx + 1) // block_size
    s_cmp = torch.einsum("bthd,bjhd->bhtj", q32 * scale, k_cmp.repeat_interleave(groups, 2))
    p_cmp, denom, peak = _masked_softmax(s_cmp, complete)
    live = complete.any(-1)[:, None]
    o_cmp = torch.einsum("bhtj,bjhd->bthd", p_cmp / denom.clamp(min=1e-30), v_cmp.repeat_interleave(groups, 2))
    o_cmp = o_cmp * live[:, None]
    lse = torch.where(live.squeeze(-1), (peak + denom.log()).squeeze(-1), 0.0)

    # Selection: per kv head, block importance is the group's summed compression
    # probability; the forced picks weigh exactly `groups`, above any softmax
    # mass. Indices only, no gradient (the kernels' topk is not differentiable).
    current = t_idx // block_size
    candidate = j_idx <= current
    scored = torch.exp(s_cmp.detach() - lse.detach()[..., None]).masked_fill(~(j_idx < current), 0.0)
    importance = scored.view(batch, -1, groups, seq, blocks).sum(2)
    importance = torch.where((j_idx == 0) | (j_idx >= current - 1), float(groups), importance)
    importance = importance.masked_fill(~candidate, -torch.inf)
    picked = importance.topk(min(block_count, blocks), -1)
    chosen = torch.zeros_like(importance, dtype=torch.bool).scatter(-1, picked.indices, picked.values.isfinite())

    key_block = (t_idx.squeeze(-1) // block_size).expand(batch, chosen.shape[1], seq, seq)
    sel = chosen.gather(-1, key_block).repeat_interleave(groups, 1) & (t_idx >= t_idx.mT)
    s_slc = torch.einsum("bthd,bshd->bhts", q32 * scale, k32.repeat_interleave(groups, 2))
    p_slc, denom, _ = _masked_softmax(s_slc, sel)
    o_slc = torch.einsum("bhts,bshd->bthd", p_slc / denom, v32.repeat_interleave(groups, 2))

    out = o_cmp * upcast(g_cmp)[..., None] + o_slc * upcast(g_slc)[..., None]
    if window_size > 0:
        near = (t_idx >= t_idx.mT) & (t_idx - t_idx.mT < window_size)
        p_swa, denom, _ = _masked_softmax(s_slc, near)
        out = out + torch.einsum("bhts,bshd->bthd", p_swa / denom, v32.repeat_interleave(groups, 2)) * upcast(g_swa)[..., None]
    return out.to(q.dtype)


# bf16 only: fp32 matmuls land on tf32 tensor cores (as with `attn:fla`) and
# fp16 backward noise lands past the harness budget (see ISSUES.md).
@nsa.register("fla", source="fla.ops.nsa.parallel_nsa", predicate=_fla_ready)
def nsa_fla(
    q: BFloat16[Tensor, "batch seq q_heads head_dim"],
    k,
    v,
    g_cmp,
    g_slc,
    g_swa,
    block_count,
    block_size,
    window_size,
    softmax_scale,
):
    return kernel(
        q,
        k,
        v,
        g_cmp=g_cmp,
        g_slc=g_slc,
        g_swa=g_swa,
        block_counts=block_count,
        block_size=block_size,
        window_size=window_size,
        scale=softmax_scale,
    )


@nsa.register("popcorn", source="popcorn.impls.nsa_tl.nsa", predicate=_popcorn_ready)
def nsa_popcorn(
    q: BFloat16[Tensor, "batch seq q_heads head_dim"],
    k,
    v,
    g_cmp,
    g_slc,
    g_swa,
    block_count,
    block_size,
    window_size,
    softmax_scale,
):
    return kernel(q, k, v, g_cmp, g_slc, g_swa, block_count, block_size, window_size, softmax_scale)
