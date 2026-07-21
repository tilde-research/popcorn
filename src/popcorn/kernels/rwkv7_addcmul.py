import torch
from jaxtyping import Float
from torch import Tensor

from popcorn import Range, Tag, register_kernel


@register_kernel(
    test_shapes={"batch": Range(1, 2), "seq": Range(2, 16), "hidden": {64}},
    tags={Tag.FEATURE_MIXER, Tag.FUSED},
)
def rwkv7_addcmul(
    hidden_states: Float[Tensor, "batch seq hidden"],
    delta: Float[Tensor, "batch seq hidden"],
    xr: Float[Tensor, "1 1 hidden"],
    xw: Float[Tensor, "1 1 hidden"],
    xk: Float[Tensor, "1 1 hidden"],
    xv: Float[Tensor, "1 1 hidden"],
    xa: Float[Tensor, "1 1 hidden"],
    xg: Float[Tensor, "1 1 hidden"] | None = None,
) -> tuple[
    Float[Tensor, "batch seq hidden"],
    Float[Tensor, "batch seq hidden"],
    Float[Tensor, "batch seq hidden"],
    Float[Tensor, "batch seq hidden"],
    Float[Tensor, "batch seq hidden"],
    Float[Tensor, "batch seq hidden"] | None,
]:
    r"""The RWKV-7 token-shift interpolation branches, fused.

    $$y_i = x + \delta \odot \mu_i, \qquad \mu_i \in \{\mu_r, \mu_w, \mu_k, \mu_v, \mu_a, \mu_g\}$$

    [RWKV-7 "Goose" (Peng et al., 2025)](https://arxiv.org/abs/2503.14456)
    """
    outputs = tuple(torch.addcmul(hidden_states, delta, weight) for weight in (xr, xw, xk, xv, xa))
    return (*outputs, None if xg is None else torch.addcmul(hidden_states, delta, xg))


rwkv7_addcmul.register(
    "fla",
    source="fla.ops.rwkv7.fused_addcmul.fused_addcmul_rwkv7",
)
