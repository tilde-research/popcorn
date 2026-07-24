from jaxtyping import Float, Float32, Int
from torch import Tensor

from popcorn import Tag, kernel, register_kernel


# logits carry one extra position (the unused next-token prediction, fla's
# contract). ref_logp, advantages, and the mask are constants, matching fla.
@register_kernel(
    test_args={"beta": [0.0, 0.1]},
    test_inputs={"completion_mask": lambda t: (t > 0).float()},
    tags={Tag.LOSS, Tag.FUSED},
)
def grpo(
    logits: Float[Tensor, "batch response+1 vocab"],
    ref_logp: Float[Tensor, "batch response"],
    input_ids: Int[Tensor, "batch response+1"],
    advantages: Float[Tensor, "batch"],
    beta: float = 0.1,
    completion_mask: Float[Tensor, "batch response"] | None = None,
) -> Float[Tensor, "batch response"]:
    r"""Per-token on-policy GRPO loss with a KL penalty against the reference policy.

    $$\mathcal{L}_t = -\Big(e^{\ell_t - \bar{\ell}_t} A - \beta \big(e^{\delta_t} - \delta_t - 1\big)\Big),
    \qquad \delta_t = \ell^{\mathrm{ref}}_t - \ell_t$$

    [GRPO (Shao et al., 2024)](https://arxiv.org/abs/2402.03300)
    """
    logps = logits[:, :-1].log_softmax(-1).gather(-1, input_ids[:, 1:, None]).squeeze(-1)
    delta = ref_logp.detach() - logps
    kl = delta.exp() - delta - 1
    loss = -((logps - logps.detach()).exp() * advantages.detach()[:, None] - beta * kl)
    return loss if completion_mask is None else loss * completion_mask.detach()


# fla always returns the loss in fp32, so only float32 inputs round-trip.
@grpo.register("fla", source="fla.modules.grpo.fused_grpo_loss")
def grpo_fla(logits: Float32[Tensor, "batch response+1 vocab"], ref_logp, input_ids, advantages, beta, completion_mask):
    return kernel(logits, ref_logp, input_ids, advantages, beta, completion_mask)
