import torch
from jaxtyping import Float
from torch import Tensor

from popcorn import Range, kernel, register_kernel


@register_kernel(
    test_shapes={
        "batch": Range(1, 2),
        "seq": Range(2, 32),
        "hidden": {64},
        "intermediate": {128},
    },
    test_inputs={
        "x_k": torch.sigmoid,
        "key_weight": lambda t: t / t.shape[0] ** 0.5,
        "value_weight": lambda t: t / t.shape[0] ** 0.5,
    },
)
def rwkv7_channel_mixing(
    x: Float[Tensor, "batch seq hidden"],
    x_prev: Float[Tensor, "batch hidden"],
    x_k: Float[Tensor, "hidden"],
    key_weight: Float[Tensor, "hidden intermediate"],
    value_weight: Float[Tensor, "intermediate hidden"],
) -> tuple[Float[Tensor, "batch seq hidden"], Float[Tensor, "batch hidden"]]:
    """Apply RWKV-7 channel mixing and return the final-token state."""
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
