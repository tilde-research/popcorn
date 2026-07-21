import torch
from jaxtyping import Float
from torch import Tensor

from popcorn import Range, Tag, kernel, register_kernel


def generalized_jsd(log_p, log_q, beta):
    """Per-element generalized JSD(beta) between student log-probs `log_p` and
    teacher log-probs `log_q`, with liger's KL conventions at the endpoints."""
    if beta == 0.0:
        return log_q.exp() * (log_q - log_p)  # KL(teacher || student)
    if beta == 1.0:
        return log_p.exp() * (log_p - log_q)  # KL(student || teacher)
    p, q = log_q.exp(), log_p.exp()
    m = beta * p + (1 - beta) * q
    return beta * p * log_q + (1 - beta) * q * log_p - m * torch.log(m)


# log_q is the teacher and treated as a constant, matching liger.
@register_kernel(
    test_shapes={"vocab": Range(2, 4096)},
    test_args={"beta": [0.0, 0.5, 1.0]},
    test_inputs={"log_p": lambda t: t.log_softmax(-1), "log_q": lambda t: t.log_softmax(-1)},
    tags={Tag.LOSS},
)
def jsd(
    log_p: Float[Tensor, "tokens vocab"],
    log_q: Float[Tensor, "tokens vocab"],
    beta: float = 0.5,
) -> Float[Tensor, ""]:
    r"""Generalized Jensen-Shannon divergence between student and constant teacher log-probs.

    $$\mathcal{L} = \frac{1}{T} \sum_t \mathrm{JSD}_\beta(p_t \,\|\, q_t),
    \qquad \mathrm{JSD}_0 = \mathrm{KL}(q \,\|\, p), \quad \mathrm{JSD}_1 = \mathrm{KL}(p \,\|\, q)$$
    """
    return generalized_jsd(log_p, log_q.detach(), beta).sum() / log_p.shape[0]


@jsd.register("liger", source="liger_kernel.transformers.functional.liger_jsd")
def jsd_liger(log_p, log_q, beta):
    return kernel(log_p, log_q, beta=beta)
