"""SwiGLU backed by the first-party CUDA extension (`impls/swiglu.cu`)."""

import torch

from popcorn.impls import _cuda


class _SwiGLU(torch.autograd.Function):
    """Autograd over the CUDA forward/backward pair, saving both inputs."""

    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return _cuda.load("swiglu").fwd(a, b)

    @staticmethod
    def backward(ctx, dout):
        a, b = ctx.saved_tensors
        return _cuda.load("swiglu").bwd(dout, a, b)


def swiglu(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return _SwiGLU.apply(a, b)
