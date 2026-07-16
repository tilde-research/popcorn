import torch
from jaxtyping import Float
from torch import Tensor

from popcorn import Range, kernel, register_kernel


def _unit_pairs(t):
    angles = t[..., 0]
    return torch.stack((angles.cos(), angles.sin()), -1)


def _rotate(x, freqs):
    pairs = x.unflatten(-1, (-1, 2))
    real, imag = pairs.unbind(-1)
    freq_real, freq_imag = freqs.unbind(-1)
    return torch.stack(
        (real * freq_real - imag * freq_imag, real * freq_imag + imag * freq_real),
        -1,
    ).flatten(-2)


@register_kernel(
    test_shapes={
        "batch": Range(1, 8),
        "seq": Range(2, 512),
        "heads": {4},
        "kv_heads": {2},
        "half": {16},
        "head_dim": {32},
    },
    test_inputs={"freqs": _unit_pairs},
)
def llama4_rope(
    q: Float[Tensor, "batch seq heads head_dim"],
    k: Float[Tensor, "batch seq kv_heads head_dim"],
    freqs: Float[Tensor, "seq half 2"],
) -> tuple[Float[Tensor, "batch seq heads head_dim"], Float[Tensor, "batch seq kv_heads head_dim"]]:
    """Paired Llama 4 rotary embedding with frequencies stored as real/imaginary
    pairs, avoiding a complex tensor input."""
    freqs = freqs.detach()[None, :, None]
    return _rotate(q, freqs), _rotate(k, freqs)


@llama4_rope.register(
    "liger",
    source="liger_kernel.transformers.llama4_rope.liger_llama4_text_rotary_pos_emb",
)
def llama4_rope_liger(q, k, freqs):
    return kernel(q, k, freqs)
