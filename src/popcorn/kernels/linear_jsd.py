import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from popcorn import Range, kernel, register_kernel
from popcorn.kernels._utils import upcast
from popcorn.kernels.jsd import generalized_jsd


@register_kernel(test_shapes={"vocab": Range(2, 4096)}, test_args={"beta": [0.5], "temperature": [1.0, 2.0]})
def linear_jsd(
    student: Float[Tensor, "tokens hidden"],
    teacher: Float[Tensor, "tokens teacher_hidden"],
    student_weight: Float[Tensor, "vocab hidden"],
    teacher_weight: Float[Tensor, "vocab teacher_hidden"],
    beta: float = 0.5,
    temperature: float = 1.0,
) -> Float[Tensor, ""]:
    """JSD between student and teacher lm-head outputs, fused with both
    projections; the teacher branch is a constant, matching liger."""
    log_p = (upcast(F.linear(student, student_weight)) / temperature).log_softmax(-1)
    log_q = (upcast(F.linear(teacher, teacher_weight)) / temperature).log_softmax(-1).detach()
    return generalized_jsd(log_p, log_q, beta).sum() / student.shape[0]


@linear_jsd.register("liger", source="liger_kernel.transformers.functional.liger_fused_linear_jsd")
def linear_jsd_liger(student, teacher, student_weight, teacher_weight, beta, temperature):
    return kernel(student, student_weight, teacher, teacher_weight, jsd_beta=beta, temperature=temperature)
