"""BitNet-style fake quantization (arXiv:2310.11453) with straight-through
gradients, matching fla's fused quant kernels bit for bit."""

import torch


def activation_quant(x: torch.Tensor) -> torch.Tensor:
    """Per-token symmetric int8 fake quant, straight-through."""
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
    q = (x * scale).round().clamp(-128, 127) / scale
    return x + (q - x).detach()


def weight_quant(w: torch.Tensor) -> torch.Tensor:
    """Per-tensor ternary (1.58-bit) fake quant, straight-through."""
    scale = 1.0 / w.abs().mean().clamp(min=1e-5)
    q = (w * scale).round().clamp(-1, 1) / scale
    return w + (q - w).detach()
