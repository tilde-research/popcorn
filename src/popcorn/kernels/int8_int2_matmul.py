import torch
from jaxtyping import Int8, Int32
from torch import Tensor

from popcorn import Div, Range, Tag, kernel, register_kernel


def _int8_input(tensor, _dims, generator):
    return torch.randint(-128, 128, tensor.shape, generator=generator).to(torch.int8)


def _int2_input(tensor, _dims, generator):
    return torch.randint(-1, 2, tensor.shape, generator=generator).to(torch.int8)


@register_kernel(
    test_shapes={"rows": Range(1, 128), "inner": {512}, "cols": Range(1, 128)},
    test_inputs={"a": _int8_input, "b": _int2_input},
    tags={Tag.LINEAR, Tag.QUANTIZED},
)
def int8_int2_matmul(
    a: Int8[Tensor, "rows inner"],
    b: Int8[Tensor, "inner cols"],
) -> Int32[Tensor, "rows cols"]:
    r"""Integer matrix product with int8 activations and ternary int2 weights.

    $$y = a b, \qquad a \in \mathbb{Z}_{\mathrm{int8}}, \; b \in \{-1, 0, 1\}$$
    """
    products = a.to(torch.int32).unsqueeze(-1) * b.to(torch.int32).unsqueeze(0)
    return products.sum(dim=1, dtype=torch.int32)


@int8_int2_matmul.register(
    "liger",
    source="liger_kernel.ops.experimental.mm_int8int2.matmul",
    supports={"inner": Div(512)},
    forward_only=True,
)
def int8_int2_matmul_liger(a, b):
    from liger_kernel.ops.experimental.mm_int8int2 import pack_weights

    return kernel(a, pack_weights(b.clone()))
