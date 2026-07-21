from jaxtyping import Float
from torch import Tensor

from popcorn import Range, Tag, kernel, register_kernel
from popcorn.kernels._utils import upcast


@register_kernel(
    test_shapes={"batch": Range(1, 2), "seq": Range(2, 16), "heads": {4}, "head_dim": {64}},
    tags={Tag.ACTIVATION, Tag.FUSED},
)
def rwkv7_gate_output(
    o: Float[Tensor, "batch seq heads head_dim"],
    r: Float[Tensor, "batch seq heads head_dim"],
    k: Float[Tensor, "batch seq heads head_dim"],
    r_k: Float[Tensor, "heads head_dim"],
    v: Float[Tensor, "batch seq heads head_dim"],
    g: Float[Tensor, "batch seq heads head_dim"],
) -> Float[Tensor, "batch seq heads head_dim"]:
    r"""RWKV-7's receptance-key correction and output gate.

    $$y = \Big(o + \big\langle r \odot k \odot \rho \big\rangle \, v\Big) \odot g$$

    [RWKV-7 "Goose" (Peng et al., 2025)](https://arxiv.org/abs/2503.14456)
    """
    o32, r32, k32, r_k32, v32, g32 = map(upcast, (o, r, k, r_k, v, g))
    correction = (r32 * k32 * r_k32).sum(-1, keepdim=True) * v32
    return ((o32 + correction) * g32).to(o.dtype)


@rwkv7_gate_output.register(
    "fla",
    source="fla.ops.rwkv7.gate_output_correction.gate_output_correction",
)
def rwkv7_gate_output_fla(o, r, k, r_k, v, g):
    shape = o.shape
    return kernel(o.flatten(-2), r, k, r_k, v, g.flatten(-2)).reshape(shape)
