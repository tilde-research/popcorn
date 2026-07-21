from jaxtyping import Float
from torch import Tensor

from popcorn import Range, Tag, kernel, register_kernel
from popcorn.kernels._utils import upcast


@register_kernel(
    test_shapes={"batch": Range(1, 2), "seq": Range(2, 16), "hidden": {64}},
    tags={Tag.FEATURE_MIXER, Tag.FUSED},
)
def rwkv7_k_update(
    k: Float[Tensor, "batch seq hidden"],
    a: Float[Tensor, "batch seq hidden"],
    ka: Float[Tensor, "hidden"],
) -> Float[Tensor, "batch seq hidden"]:
    r"""Interpolate RWKV-7 keys toward their in-context learning update.

    $$y = k \odot \big(1 + (a - 1) \odot \mu_{ka}\big)$$

    [RWKV-7 "Goose" (Peng et al., 2025)](https://arxiv.org/abs/2503.14456)
    """
    k32, a32, ka32 = map(upcast, (k, a, ka))
    return k32.addcmul(k32 * (a32 - 1), ka32).to(k.dtype)


@rwkv7_k_update.register(
    "fla",
    source="fla.ops.rwkv7.fused_k_update.fused_k_rwkv7",
)
def rwkv7_k_update_fla(k, a, ka):
    return kernel(k, a, ka.reshape(1, 1, -1))
