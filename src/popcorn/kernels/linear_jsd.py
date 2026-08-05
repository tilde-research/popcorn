import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from popcorn import Tag, register_kernel
from popcorn.kernels._utils import upcast
from popcorn.kernels.jsd import generalized_jsd


# The teacher branch is a constant, matching liger.
@register_kernel(
    test_args={"beta": [0.5], "temperature": [1.0, 2.0]},
    tags={Tag.LOSS, Tag.LINEAR, Tag.FUSED},
)
def linear_jsd(
    student: Float[Tensor, "tokens hidden"],
    teacher: Float[Tensor, "tokens teacher_hidden"],
    student_weight: Float[Tensor, "vocab hidden"],
    teacher_weight: Float[Tensor, "vocab teacher_hidden"],
    beta: float = 0.5,
    temperature: float = 1.0,
) -> Float[Tensor, ""]:
    r"""Generalized JSD between student and teacher lm-head outputs, fused with both projections.

    $$\mathcal{L} = \frac{1}{T} \sum_t \mathrm{JSD}_\beta\!\Big(
    \operatorname{softmax}\!\big(s_t w_s^\top / \tau\big) \,\Big\|\,
    \operatorname{softmax}\!\big(u_t w_u^\top / \tau\big)\Big)$$

    [Liger Kernel (Hsu et al., 2024)](https://arxiv.org/abs/2410.10989)
    """
    log_p = (F.linear(upcast(student), upcast(student_weight)) / temperature).log_softmax(-1)
    log_q = (F.linear(upcast(teacher), upcast(teacher_weight)) / temperature).log_softmax(-1).detach()
    return generalized_jsd(log_p, log_q, beta).sum(-1).mean()
