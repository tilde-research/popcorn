import torch
from jaxtyping import Float
from torch import Tensor

from popcorn import Tag, kernel, register_kernel


@register_kernel(
    test_inputs={
        "x_k": torch.sigmoid,
        "key_weight": lambda t: t / t.shape[0] ** 0.5,
        "value_weight": lambda t: t / t.shape[0] ** 0.5,
    },
    tags={Tag.FEATURE_MIXER, Tag.ACTIVATION, Tag.LINEAR, Tag.FUSED},
)
def rwkv7_channel_mixing(
    x: Float[Tensor, "batch seq hidden"],
    x_prev: Float[Tensor, "batch hidden"],
    x_k: Float[Tensor, "hidden"],
    key_weight: Float[Tensor, "hidden intermediate"],
    value_weight: Float[Tensor, "intermediate hidden"],
) -> tuple[Float[Tensor, "batch seq hidden"], Float[Tensor, "batch hidden"]]:
    r"""RWKV-7 channel mixing: token-shift interpolation, squared-ReLU MLP, final-token state out.

    $$y = \big(\max(m w_k, 0)\big)^2 w_v, \qquad m = x + (x_{\mathrm{prev}} - x) \odot \mu_k$$

    [RWKV-7 "Goose" (Peng et al., 2025)](https://arxiv.org/abs/2503.14456)
    """
    previous = torch.cat((x_prev[:, None], x[:, :-1]), dim=1)
    mixed = torch.addcmul(x, previous - x, x_k)
    output = torch.relu(mixed @ key_weight).square() @ value_weight
    return output, x[:, -1]


def _fla_supported(x, key_weight, **_):
    return x.shape[0] * x.shape[1] * key_weight.shape[-1] % 4096 == 0


@rwkv7_channel_mixing.register(
    "fla",
    source="fla.ops.rwkv7.channel_mixing.channel_mixing_rwkv7",
    predicate=_fla_supported,
)
def rwkv7_channel_mixing_fla(x, x_prev, x_k, key_weight, value_weight):
    batch = x.shape[0]
    key_weight = key_weight.unsqueeze(0).expand(batch, -1, -1)
    value_weight = value_weight.unsqueeze(0).expand(batch, -1, -1)
    output, _ = kernel(
        x,
        x_prev,
        x_k.reshape(1, 1, -1),
        key_weight,
        value_weight,
        inplace=False,
    )
    return output, x[:, -1]
