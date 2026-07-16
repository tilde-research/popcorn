# Adapted from flash-linear-attention ((c) Songlin Yang, Yu Zhang, MIT) and flash-attention ((c) Tri Dao, BSD-3-Clause).

"""Triton rotary embedding, trimmed to the reference contract: fixed-length batches, no offsets, out of place."""

import torch
import triton
import triton.language as tl

from popcorn.impls._triton import sm_count


@triton.autotune(
    configs=[triton.Config({}, num_warps=warps, num_stages=stages) for warps in (2, 4, 8) for stages in (2, 3)],
    key=["H", "D", "INTERLEAVED"],
)
@triton.jit(do_not_specialize=["T"])
def _rotary_kernel(
    x,
    cos,
    sin,
    y,
    T,
    H: tl.constexpr,
    D: tl.constexpr,
    R: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    CONJUGATE: tl.constexpr,
):
    i_t, i_b, i_h = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    if i_t * BT >= T:
        return
    x += i_b * T * H * D + i_h * D
    y += i_b * T * H * D + i_h * D
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T

    if not INTERLEAVED:
        # GPT-NeoX halves: rotate x[..., :R] with x[..., R:2R].
        o_r = tl.arange(0, BD // 2)
        mask = m_t[:, None] & (o_r < R)[None, :]
        b_cos = tl.load(cos + o_t[:, None] * R + o_r[None, :], mask=mask, other=1.0).to(tl.float32)
        b_sin = tl.load(sin + o_t[:, None] * R + o_r[None, :], mask=mask, other=0.0).to(tl.float32)
        if CONJUGATE:
            b_sin = -b_sin
        p_x = x + o_t[:, None] * H * D + o_r[None, :]
        b_x0 = tl.load(p_x, mask=mask, other=0.0).to(tl.float32)
        b_x1 = tl.load(p_x + R, mask=mask, other=0.0).to(tl.float32)
        p_y = y + o_t[:, None] * H * D + o_r[None, :]
        tl.store(p_y, b_x0 * b_cos - b_x1 * b_sin, mask=mask)
        tl.store(p_y + R, b_x0 * b_sin + b_x1 * b_cos, mask=mask)
    else:
        # GPT-J pairs: load x as-is and pair-swapped, blend with tl.where, so
        # both loads stay contiguous-ish instead of strided by 2.
        o_d = tl.arange(0, BD)
        o_d_swap = o_d + ((o_d + 1) % 2) * 2 - 1  # 1, 0, 3, 2, ...
        o_d_repeat = o_d // 2
        mask = m_t[:, None] & (o_d_repeat < R)[None, :]
        b_cos = tl.load(cos + o_t[:, None] * R + o_d_repeat[None, :], mask=mask, other=1.0).to(tl.float32)
        b_sin = tl.load(sin + o_t[:, None] * R + o_d_repeat[None, :], mask=mask, other=0.0).to(tl.float32)
        if CONJUGATE:
            b_sin = -b_sin
        b_x0 = tl.load(x + o_t[:, None] * H * D + o_d[None, :], mask=mask, other=0.0).to(tl.float32)
        b_x1 = tl.load(x + o_t[:, None] * H * D + o_d_swap[None, :], mask=mask, other=0.0).to(tl.float32)
        b_y = tl.where(o_d[None, :] % 2 == 0, b_x0 * b_cos - b_x1 * b_sin, b_x0 * b_cos + b_x1 * b_sin)
        tl.store(y + o_t[:, None] * H * D + o_d[None, :], b_y, mask=mask)


def _apply(x, cos, sin, interleaved, conjugate):
    x, cos, sin = x.contiguous(), cos.contiguous(), sin.contiguous()
    B, T, H, D = x.shape
    R = cos.shape[-1]
    y = torch.empty_like(x)
    if 2 * R < D:
        y[..., 2 * R :].copy_(x[..., 2 * R :])
    BT = min(128, triton.next_power_of_2(triton.cdiv(T, sm_count(x.device.index))))
    _rotary_kernel[(triton.cdiv(T, BT), B, H)](
        x,
        cos,
        sin,
        y,
        T,
        H=H,
        D=D,
        R=R,
        BT=BT,
        BD=triton.next_power_of_2(2 * R),
        INTERLEAVED=interleaved,
        CONJUGATE=conjugate,
    )
    return y


class _Rotary(torch.autograd.Function):
    """Autograd wrapper: the backward applies the conjugate rotation."""

    @staticmethod
    def forward(ctx, x, cos, sin, interleaved):
        ctx.save_for_backward(cos, sin)
        ctx.interleaved = interleaved
        return _apply(x, cos, sin, interleaved, conjugate=False)

    @staticmethod
    def backward(ctx, dy):
        cos, sin = ctx.saved_tensors
        return _apply(dy, cos, sin, ctx.interleaved, conjugate=True), None, None, None


def rotary_embedding(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, interleaved: bool = False) -> torch.Tensor:
    return _Rotary.apply(x, cos, sin, interleaved)
