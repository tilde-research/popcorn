import torch
from jaxtyping import Float
from torch import Tensor

from popcorn import Pow2, Tag, kernel, register_kernel
from popcorn.kernels._utils import upcast


@register_kernel(test_shapes={"dim": Pow2()}, test_args={"scale": [1.0, 0.5]}, tags={Tag.LINEAR})
def hadamard_transform(x: Float[Tensor, "... dim"], scale: float = 1.0) -> Float[Tensor, "... dim"]:
    r"""Sylvester Hadamard transform along the last dimension (a power of two).

    $$y = s \, x H_d, \qquad H_{2d} = \begin{pmatrix} H_d & H_d \\ H_d & -H_d \end{pmatrix}$$
    """
    xw = upcast(x)
    matrix = torch.ones(1, 1, dtype=xw.dtype, device=x.device)
    while matrix.shape[-1] < x.shape[-1]:
        matrix = torch.cat([torch.cat([matrix, matrix], -1), torch.cat([matrix, -matrix], -1)], 0)
    return (xw @ matrix * scale).to(x.dtype)


@hadamard_transform.register("quack", source="quack.hadamard.hadamard_transform")
def hadamard_transform_quack(x, scale):
    return kernel(x, scale)
