import torch
from jaxtyping import Int8, Int32
from torch import Tensor

from popcorn import Tag, kernel, register_kernel


def _int8_input(tensor, _dims, generator):
    return torch.randint(-128, 128, tensor.shape, generator=generator).to(torch.int8)


def _int2_input(tensor, _dims, generator):
    return torch.randint(-1, 2, tensor.shape, generator=generator).to(torch.int8)


@register_kernel(test_inputs={"a": _int8_input, "b": _int2_input}, tags={Tag.LINEAR, Tag.QUANTIZED})
def int8_int2_matmul(a: Int8[Tensor, "rows inner"], b: Int8[Tensor, "inner cols"]) -> Int32[Tensor, "rows cols"]:
    r"""Integer matrix product with int8 activations and ternary int2 weights.

    $$y = a b, \qquad a \in \mathbb{Z}_{\mathrm{int8}}, \; b \in \{-1, 0, 1\}$$

    [BitNet b1.58 (Ma et al., 2024)](https://arxiv.org/abs/2402.17764)
    """
    products = a.to(torch.int32).unsqueeze(-1) * b.to(torch.int32).unsqueeze(0)
    return products.sum(dim=1, dtype=torch.int32)


# the module source serves both callables: liger packs four ternary weights
# per byte before its matmul.
@int8_int2_matmul.register("liger", source="liger_kernel.ops.experimental.mm_int8int2", forward_only=True)
def int8_int2_matmul_liger(a, b):
    return kernel.matmul(a, kernel.pack_weights(b.clone()))
