# Adapted from github.com/tilde-research/nsa-release, with mask, lse normalization, scale, stride, and guard fixes.

"""Native sparse attention assembled from first-party pieces: flex-attention
compression, fla topk, the tilde one-pass selection kernel, and an fa3 window."""

import torch
import triton
import triton.language as tl
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from fla.ops.nsa.parallel import parallel_nsa_topk
from fla.ops.utils.pooling import mean_pooling

_flex_compiled = None


def _flex():
    global _flex_compiled
    if _flex_compiled is None:
        # donated buffers forbid re-running the compiled backward with a
        # retained graph, which benchmark loops do
        torch._functorch.config.donated_buffer = False
        _flex_compiled = torch.compile(flex_attention, dynamic=False)
    return _flex_compiled


def compression_attention(q, k_cmp, v_cmp, block_size, scale):
    """Causal attention of every query over the complete compressed blocks.

    Queries before the first complete block see no keys; flex_attention's
    backward emits garbage for fully-masked rows, so those rows attend block 0
    spuriously and are zeroed afterwards (the zero multiplier also kills their
    gradient contribution exactly, matching fla's convention of o=0, lse=0)."""

    def visible(b, h, q_idx, kv_idx):
        return (q_idx >= (kv_idx + 1) * block_size - 1) | ((q_idx < block_size - 1) & (kv_idx == 0))

    block_mask = create_block_mask(visible, None, None, q.shape[1], k_cmp.shape[1], device=q.device.type)
    o_cmp, lse_cmp = _flex()(
        q.transpose(1, 2),
        k_cmp.transpose(1, 2),
        v_cmp.transpose(1, 2),
        block_mask=block_mask,
        scale=scale,
        enable_gqa=True,
        return_lse=True,
    )
    live = torch.arange(q.shape[1], device=q.device) >= block_size - 1
    o_cmp = o_cmp.transpose(1, 2) * live.view(1, -1, 1, 1)
    lse_cmp = torch.where(live.view(1, 1, -1), lse_cmp, 0.0)
    return o_cmp, lse_cmp.transpose(1, 2).contiguous()


_sel_fwd_configs = [triton.Config({}, num_warps=num_warps) for num_warps in [1, 2, 4, 8]]

_sel_bwd_preprocess_configs = [
    triton.Config({"BLOCK_M": 16, "num_stages": 1, "num_warps": 4}, num_ctas=1),
    triton.Config({"BLOCK_M": 32, "num_stages": 1, "num_warps": 4}, num_ctas=1),
    triton.Config({"BLOCK_M": 16, "num_stages": 2, "num_warps": 4}, num_ctas=1),
    triton.Config({"BLOCK_M": 32, "num_stages": 2, "num_warps": 4}, num_ctas=1),
    triton.Config({"BLOCK_M": 16, "num_stages": 1, "num_warps": 8}, num_ctas=1),
    triton.Config({"BLOCK_M": 32, "num_stages": 1, "num_warps": 8}, num_ctas=1),
]

_sel_bwd_configs = [triton.Config({}, num_warps=num_warps) for num_warps in [1, 2, 4, 8]]


@triton.autotune(configs=_sel_fwd_configs, key=["M", "N", "D", "SELECTION_BLOCK_SIZE", "T", "HEADS_PER_GROUP", "causal"])
@triton.jit
def _sel_attn_fwd_kernel(
    Q: tl.tensor,
    K: tl.tensor,
    V: tl.tensor,
    Top_idx: tl.tensor,
    softmax_scale: tl.constexpr,
    causal: tl.constexpr,
    Out: tl.tensor,
    Lse: tl.tensor,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_kb,
    stride_kg,
    stride_kn,
    stride_kd,
    stride_vb,
    stride_vg,
    stride_vn,
    stride_vd,
    stride_tb,
    stride_tg,
    stride_tm,
    stride_tt,
    stride_ob,
    stride_oh,
    stride_om,
    stride_od,
    stride_lb,
    stride_lh,
    stride_lm,
    B: tl.constexpr,
    H: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    T: tl.constexpr,
    DP: tl.constexpr,
    SELECTION_BLOCK_SIZE: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    OFFSET_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    stride_hg = stride_qh * HEADS_PER_GROUP

    b = tl.program_id(0)
    m = tl.program_id(1) + OFFSET_M
    g = tl.program_id(2)

    q_base = Q + b * stride_qb + m * stride_qm + g * stride_hg
    k_base = K + b * stride_kb + g * stride_kg
    v_base = V + b * stride_vb + g * stride_vg
    t_base = Top_idx + b * stride_tb + m * stride_tm + g * stride_tg
    o_base = Out + b * stride_ob + m * stride_om + g * stride_hg
    l_base = Lse + b * stride_lb + m * stride_lm + g * stride_lh * HEADS_PER_GROUP

    offs_h = tl.arange(0, BLOCK_H)
    mask_h = offs_h < HEADS_PER_GROUP
    offs_d = tl.arange(0, DP)
    mask_d = offs_d < D
    offs_n = tl.arange(0, SELECTION_BLOCK_SIZE)

    q_ptrs = q_base + offs_h[:, None] * stride_qh + offs_d[None, :] * stride_qd
    q_blck = tl.load(q_ptrs, mask=mask_h[:, None] & mask_d[None, :], other=0.0)

    max_log = tl.full([BLOCK_H], float("-inf"), dtype=tl.float32)
    sum_exp = tl.full([BLOCK_H], 1.0, dtype=tl.float32)
    accum = tl.zeros([BLOCK_H, DP], dtype=tl.float32)

    max_col = max(0, N - M + m) if causal else N

    for idx in range(T):
        top = tl.load(t_base + idx * stride_tt)

        col = top * SELECTION_BLOCK_SIZE
        col = tl.multiple_of(col, SELECTION_BLOCK_SIZE)

        if not causal or (col <= max_col and col >= 0):
            cols = col + offs_n
            mask_n = cols < N

            k_ptrs = k_base + offs_d[:, None] * stride_kd + cols[None, :] * stride_kn
            k_blck = tl.load(k_ptrs, mask=mask_d[:, None] & mask_n[None, :], other=0.0)

            v_ptrs = v_base + cols[:, None] * stride_vn + offs_d[None, :] * stride_vd
            v_blck = tl.load(v_ptrs, mask=mask_d[None, :] & mask_n[:, None], other=0.0).to(tl.float32)

            qk = tl.dot(q_blck, k_blck) * softmax_scale

            causal_mask = cols <= max_col
            qk = tl.where(causal_mask[None, :], qk, float("-inf"))

            new_max = tl.maximum(max_log, tl.max(qk, axis=1))
            exp_qk = tl.math.exp(qk - new_max[:, None])
            sum_qk = tl.sum(exp_qk, axis=1)

            alpha = tl.math.exp(max_log - new_max)
            sum_exp = sum_exp * alpha + sum_qk
            accum = accum * alpha[:, None]

            accum = tl.dot(exp_qk, v_blck, accum)
            max_log = new_max

    fin_log = max_log + tl.math.log(sum_exp)
    out_vals = accum / sum_exp[:, None]

    o_ptrs = o_base + offs_h[:, None] * stride_oh + offs_d[None, :] * stride_od
    tl.store(o_ptrs, out_vals, mask=mask_h[:, None] & mask_d[None, :])

    l_ptrs = l_base + offs_h * stride_lh
    tl.store(l_ptrs, fin_log, mask=mask_h)


@triton.autotune(configs=_sel_bwd_preprocess_configs, key=["M", "D", "H"])
@triton.jit
def _sel_attn_bwd_preprocess_kernel(
    Out,
    DOut,
    Delta,
    stride_ob,
    stride_oh,
    stride_om,
    stride_od,
    stride_dob,
    stride_doh,
    stride_dom,
    stride_dod,
    stride_db,
    stride_dh,
    stride_dm,
    B: tl.constexpr,
    H: tl.constexpr,
    M: tl.constexpr,
    D: tl.constexpr,
    DP: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    m = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H

    o_base = Out + b * stride_ob + h * stride_oh
    do_base = DOut + b * stride_dob + h * stride_doh

    offs_m = m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, DP)

    o_ptrs = o_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    do_ptrs = do_base + offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dod

    mask = (offs_m[:, None] < M) & (offs_d[None, :] < D)

    o = tl.load(o_ptrs, mask=mask, other=0.0).to(tl.float32)
    do = tl.load(do_ptrs, mask=mask, other=0.0).to(tl.float32)

    delta = tl.sum(o * do, axis=1)

    delta_ptr = Delta + b * stride_db + h * stride_dh + offs_m * stride_dm
    tl.store(delta_ptr, delta, mask=offs_m < M)


@triton.autotune(
    configs=_sel_bwd_configs,
    key=["M", "N", "D", "SELECTION_BLOCK_SIZE", "T", "HEADS_PER_GROUP", "causal"],
    reset_to_zero=["DK", "DV"],
)
@triton.jit
def _sel_attn_bwd_kernel(
    Q: tl.tensor,
    K: tl.tensor,
    V: tl.tensor,
    Top_idx: tl.tensor,
    Lse: tl.tensor,
    DOut: tl.tensor,
    Delta: tl.tensor,
    softmax_scale: tl.constexpr,
    causal: tl.constexpr,
    DQ: tl.tensor,
    DK: tl.tensor,
    DV: tl.tensor,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_kb,
    stride_kg,
    stride_kn,
    stride_kd,
    stride_vb,
    stride_vg,
    stride_vn,
    stride_vd,
    stride_tb,
    stride_tg,
    stride_tm,
    stride_tt,
    stride_dob,
    stride_doh,
    stride_dom,
    stride_dod,
    stride_lb,
    stride_lh,
    stride_lm,
    B: tl.constexpr,
    H: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    T: tl.constexpr,
    DP: tl.constexpr,
    SELECTION_BLOCK_SIZE: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    OFFSET_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    stride_hg = stride_qh * HEADS_PER_GROUP
    b = tl.program_id(0)
    m = tl.program_id(1) + OFFSET_M
    g = tl.program_id(2)

    q_base = Q + b * stride_qb + m * stride_qm + g * stride_hg
    k_base = K + b * stride_kb + g * stride_kg
    v_base = V + b * stride_vb + g * stride_vg
    t_base = Top_idx + b * stride_tb + m * stride_tm + g * stride_tg
    l_base = Lse + b * stride_lb + m * stride_lm + g * stride_lh * HEADS_PER_GROUP
    # upstream reused Out's strides for DOut; autograd cotangents need not share
    # Out's layout (e.g. an einsum upstream hands over a permuted grad), so the
    # kernel takes DOut's own strides
    do_base = DOut + b * stride_dob + m * stride_dom + g * stride_doh * HEADS_PER_GROUP
    d_base = Delta + b * stride_lb + m * stride_lm + g * stride_lh * HEADS_PER_GROUP
    dq_base = DQ + b * stride_qb + m * stride_qm + g * stride_hg
    dk_base = DK + b * stride_kb + g * stride_kg
    dv_base = DV + b * stride_vb + g * stride_vg

    offs_h = tl.arange(0, BLOCK_H)
    mask_h = offs_h < HEADS_PER_GROUP
    offs_d = tl.arange(0, DP)
    mask_d = offs_d < D
    offs_n = tl.arange(0, SELECTION_BLOCK_SIZE)

    q_ptrs = q_base + offs_h[:, None] * stride_qh + offs_d[None, :] * stride_qd
    q_blck = tl.load(q_ptrs, mask=mask_h[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    do_ptrs = do_base + offs_h[:, None] * stride_doh + offs_d[None, :] * stride_dod
    do_blck = tl.load(do_ptrs, mask=mask_h[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    l_ptrs = l_base + offs_h * stride_lh
    l_blck = tl.load(l_ptrs, mask=mask_h, other=0.0)

    d_ptrs = d_base + offs_h * stride_lh
    d_blck = tl.load(d_ptrs, mask=mask_h, other=0.0)

    accum = tl.zeros([BLOCK_H, DP], dtype=tl.float32)
    log_scale = softmax_scale * 1.44269504

    max_col = max(0, N - M + m) if causal else N

    for idx in range(T):
        top = tl.load(t_base + idx * stride_tt)

        col = top * SELECTION_BLOCK_SIZE
        col = tl.multiple_of(col, SELECTION_BLOCK_SIZE)

        # upstream dropped the `col >= 0` sentinel guard here (fwd has it);
        # fla's topk pads unused slots with -1, which indexed OOB
        if not causal or (col <= max_col and col >= 0):
            cols = col + offs_n
            mask_n = cols < N

            k_ptrs = k_base + cols[None, :] * stride_kn + offs_d[:, None] * stride_kd
            k_blck = tl.load(k_ptrs, mask=mask_d[:, None] & mask_n[None, :], other=0.0).to(tl.float32)

            qk = tl.dot(q_blck, k_blck) * log_scale

            causal_mask = cols <= max_col
            qk = tl.where(causal_mask[None, :], qk, -1e6)

            l2 = l_blck * 1.44269504
            exp_qk = tl.math.exp2(qk - l2[:, None])

            dv_inc = tl.dot(tl.trans(exp_qk), do_blck)

            dv_ptrs = dv_base + cols[:, None] * stride_vn + offs_d[None, :] * stride_vd
            tl.atomic_add(dv_ptrs, dv_inc.to(tl.float32), mask=mask_d[None, :] & mask_n[:, None], sem="release", scope="gpu")

            v_ptrs = v_base + cols[None, :] * stride_vn + offs_d[:, None] * stride_vd
            v_blck = tl.load(v_ptrs, mask=mask_d[:, None] & mask_n[None, :], other=0.0).to(tl.float32)
            dp = tl.dot(do_blck, v_blck)
            ds2 = exp_qk * (dp - d_blck[:, None])
            ds = ds2 * softmax_scale

            accum = tl.dot(ds, tl.trans(k_blck), acc=accum)

            dk_inc = tl.dot(tl.trans(ds), q_blck)

            dk_ptrs = dk_base + cols[:, None] * stride_kn + offs_d[None, :] * stride_kd
            tl.atomic_add(dk_ptrs, dk_inc.to(tl.float32), mask=mask_d[None, :] & mask_n[:, None], sem="release", scope="gpu")

    dq_ptrs = dq_base + offs_h[:, None] * stride_qh + offs_d[None, :] * stride_qd
    tl.store(dq_ptrs, accum, mask=mask_h[:, None] & mask_d[None, :])


class SelectionAttention(torch.autograd.Function):
    """Autograd for block-sparse selection attention over top-k block indices."""

    @staticmethod
    def forward(ctx, q, k, v, top_idx, selection_block_size, softmax_scale, causal):
        B, M, H, D = q.shape
        _, N, G, _ = k.shape
        _, _, _, T = top_idx.shape

        assert k.shape == (B, N, G, D)
        assert v.shape == (B, N, G, D)
        assert top_idx.shape == (B, M, G, T)

        if softmax_scale is None:
            softmax_scale = 1.0 / (D**0.5)

        out = torch.zeros_like(q)
        lse = torch.full((B, H, M), float("-inf"), device=q.device, dtype=torch.float32)

        DP = triton.next_power_of_2(D)
        HEADS_PER_GROUP = H // G
        OFFSET_M = max(0, M - N) if causal else 0
        BLOCK_H = max(16, HEADS_PER_GROUP)

        grid = (B, M - OFFSET_M, G)

        _sel_attn_fwd_kernel[grid](
            q,
            k,
            v,
            top_idx,
            softmax_scale,
            causal,
            out,
            lse,
            q.stride(0),
            q.stride(2),
            q.stride(1),
            q.stride(3),
            k.stride(0),
            k.stride(2),
            k.stride(1),
            k.stride(3),
            v.stride(0),
            v.stride(2),
            v.stride(1),
            v.stride(3),
            top_idx.stride(0),
            top_idx.stride(2),
            top_idx.stride(1),
            top_idx.stride(3),
            out.stride(0),
            out.stride(2),
            out.stride(1),
            out.stride(3),
            lse.stride(0),
            lse.stride(1),
            lse.stride(2),
            B,
            H,
            M,
            N,
            D,
            T,
            DP,
            SELECTION_BLOCK_SIZE=selection_block_size,
            HEADS_PER_GROUP=HEADS_PER_GROUP,
            OFFSET_M=OFFSET_M,
            BLOCK_H=BLOCK_H,
        )

        ctx.save_for_backward(q, k, v, top_idx, out, lse)
        ctx.selection_block_size = selection_block_size
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        return out

    @staticmethod
    def backward(ctx, d_out):
        q, k, v, top_idx, out, lse = ctx.saved_tensors
        B, M, H, D = q.shape
        _, N, G, _ = k.shape
        _, _, _, T = top_idx.shape

        selection_block_size = ctx.selection_block_size
        softmax_scale = ctx.softmax_scale
        causal = ctx.causal

        delta = torch.empty_like(lse)

        DP = triton.next_power_of_2(D)
        HEADS_PER_GROUP = H // G
        OFFSET_M = max(0, M - N) if causal else 0
        BLOCK_H = max(16, HEADS_PER_GROUP)

        def grid_preprocess(META):
            return (triton.cdiv(M, META["BLOCK_M"]), B * H)

        _sel_attn_bwd_preprocess_kernel[grid_preprocess](
            out,
            d_out,
            delta,
            out.stride(0),
            out.stride(2),
            out.stride(1),
            out.stride(3),
            d_out.stride(0),
            d_out.stride(2),
            d_out.stride(1),
            d_out.stride(3),
            delta.stride(0),
            delta.stride(1),
            delta.stride(2),
            B,
            H,
            M,
            D,
            DP,
        )

        dq = torch.empty_like(q)
        dk = torch.zeros_like(k, dtype=torch.float32)
        dv = torch.zeros_like(v, dtype=torch.float32)

        grid_bwd = (B, M - OFFSET_M, G)

        _sel_attn_bwd_kernel[grid_bwd](
            q,
            k,
            v,
            top_idx,
            lse,
            d_out,
            delta,
            softmax_scale,
            causal,
            dq,
            dk,
            dv,
            q.stride(0),
            q.stride(2),
            q.stride(1),
            q.stride(3),
            k.stride(0),
            k.stride(2),
            k.stride(1),
            k.stride(3),
            v.stride(0),
            v.stride(2),
            v.stride(1),
            v.stride(3),
            top_idx.stride(0),
            top_idx.stride(2),
            top_idx.stride(1),
            top_idx.stride(3),
            d_out.stride(0),
            d_out.stride(2),
            d_out.stride(1),
            d_out.stride(3),
            lse.stride(0),
            lse.stride(1),
            lse.stride(2),
            B,
            H,
            M,
            N,
            D,
            T,
            DP,
            SELECTION_BLOCK_SIZE=selection_block_size,
            HEADS_PER_GROUP=HEADS_PER_GROUP,
            OFFSET_M=OFFSET_M,
            BLOCK_H=BLOCK_H,
        )

        return dq, dk.to(k.dtype), dv.to(v.dtype), None, None, None, None


def selection_attention(q, k, v, block_indices, block_size, scale):
    return SelectionAttention.apply(q, k, v, block_indices, block_size, scale, True)


def nsa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_cmp: torch.Tensor,
    g_slc: torch.Tensor,
    g_swa: torch.Tensor,
    block_count: int,
    block_size: int,
    window_size: int,
    softmax_scale: float | None,
) -> torch.Tensor:
    """NSA with the popcorn reference signature: compression via flex
    attention, topk via fla, one-pass selection, fa3 sliding window."""
    scale = softmax_scale if softmax_scale is not None else q.shape[-1] ** -0.5

    k_cmp, v_cmp = mean_pooling(k, block_size), mean_pooling(v, block_size)
    o_cmp, lse_cmp = compression_attention(q, k_cmp, v_cmp, block_size, scale)

    block_indices = parallel_nsa_topk(
        q=q,
        k=k_cmp,
        TK=k.shape[1],
        lse=lse_cmp,
        block_counts=block_count,
        block_size=block_size,
        scale=scale,
        cu_seqlens=None,
    )
    o_slc = selection_attention(q, k, v, block_indices, block_size, scale)

    out = o_cmp * g_cmp.unsqueeze(-1) + o_slc * g_slc.unsqueeze(-1)
    if window_size > 0:
        from flash_attn_interface import flash_attn_func

        o_swa = flash_attn_func(q, k, v, softmax_scale=scale, causal=True, window_size=(window_size - 1, 0))
        out = out + o_swa * g_swa.unsqueeze(-1)
    return out
