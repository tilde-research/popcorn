import torch.nn.functional as F
from jaxtyping import Float, Float32, Int
from torch import Tensor

from popcorn import Tag, kernel, register_kernel


@register_kernel(
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


@embedding.register("liger", source="liger_kernel.ops.LigerEmbeddingFunction.apply")
def embedding_liger(
    weight: Float32[Tensor, "vocab embedding_dim"],
    indices,
):
    return kernel(weight, indices)
