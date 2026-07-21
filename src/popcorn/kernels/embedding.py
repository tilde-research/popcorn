import torch.nn.functional as F
from jaxtyping import Float, Float32, Int
from torch import Tensor

from popcorn import Range, Tag, register_kernel


@register_kernel(
    test_shapes={
        "batch": Range(1, 8),
        "seq": Range(2, 128),
        "vocab": Range(2, 4096),
        "embedding_dim": {64},
    },
    tags={Tag.LINEAR},
)
def embedding(
    weight: Float[Tensor, "vocab embedding_dim"],
    indices: Int[Tensor, "batch seq"],
) -> Float[Tensor, "batch seq embedding_dim"]:
    r"""Embedding lookup with gradients accumulated into the table.

    $$y_t = w_{i_t}$$
    """
    return F.embedding(indices, weight)


@embedding.register("liger")
def embedding_liger(
    weight: Float32[Tensor, "vocab embedding_dim"],
    indices,
):
    from liger_kernel.ops import LigerEmbeddingFunction

    return LigerEmbeddingFunction.apply(weight, indices)
