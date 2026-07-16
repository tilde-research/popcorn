import torch
from jaxtyping import Float
from torch import Tensor

from popcorn import Range, register_kernel
from popcorn.kernels._utils import upcast


# head_dim must equal 2 * half (full-width rotation); singleton test pools keep
# the grid consistent. Partial rotary (head_dim > 2 * half) also dispatches.
@register_kernel(test_shapes={"batch": Range(1, 128), "seq": Range(2, 512), "heads": {4}, "half": {16}, "head_dim": {32}})
def rotary_embedding(
    x: Float[Tensor, "batch seq heads head_dim"],
    cos: Float[Tensor, "seq half"],
    sin: Float[Tensor, "seq half"],
    interleaved: bool = False,
) -> Float[Tensor, "batch seq heads head_dim"]:
    """Rotary position embedding (arXiv:2104.09864), GPT-NeoX halves or GPT-J
    interleaved pairs; `cos`/`sin` are constants, matching the kernels."""
    cos, sin = cos.detach(), sin.detach()
    half = cos.shape[-1]
    c = upcast(cos[None, :, None, :])
    s = upcast(sin[None, :, None, :])
    if interleaved:
        x1, x2 = upcast(x[..., 0 : 2 * half : 2]), upcast(x[..., 1 : 2 * half : 2])
        rotated = torch.stack((x1 * c - x2 * s, x1 * s + x2 * c), dim=-1).flatten(-2)
    else:
        x1, x2 = upcast(x[..., :half]), upcast(x[..., half : 2 * half])
        rotated = torch.cat((x1 * c - x2 * s, x1 * s + x2 * c), dim=-1)
    rotated = rotated.to(x.dtype)
    return rotated if 2 * half == x.shape[-1] else torch.cat((rotated, x[..., 2 * half :]), dim=-1)


rotary_embedding.register("fla", source="fla.modules.rotary.rotary_embedding")
rotary_embedding.register("popcorn", source="popcorn.impls.rotary_embedding_tl.rotary_embedding")
rotary_embedding.register("quack", source="quack.rotary.apply_rotary_emb")
