"""Small numeric helpers shared by kernel references."""

import torch


def upcast(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.promote_types(x.dtype, torch.float32))


def default_scale(value: float | None, dim: int) -> float:
    return dim**-0.5 if value is None else value


def rms(x: torch.Tensor, eps: float) -> torch.Tensor:
    h = upcast(x)
    return (h * torch.rsqrt(h.square().mean(-1, keepdim=True) + eps)).to(x.dtype)
