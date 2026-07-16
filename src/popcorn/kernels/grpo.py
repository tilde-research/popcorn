from jaxtyping import Float, Float32, Int
from torch import Tensor

from popcorn import Range, kernel, register_kernel


# resp_plus must equal resp + 1 (the last logit is the unused next-token
# prediction, fla's contract); singleton test pools keep the grid consistent.
@register_kernel(
    test_shapes={"batch": Range(1, 8), "resp": {31}, "resp_plus": {32}, "vocab": Range(2, 4096)},
    test_args={"beta": [0.0, 0.1]},
    test_inputs={"completion_mask": lambda t: (t > 0).float()},
)
def grpo(
    logits: Float[Tensor, "batch resp_plus vocab"],
    ref_logp: Float[Tensor, "batch resp"],
    input_ids: Int[Tensor, "batch resp_plus"],
    advantages: Float[Tensor, "batch"],
    beta: float = 0.1,
    completion_mask: Float[Tensor, "batch resp"] | None = None,
) -> Float[Tensor, "batch resp"]:
    """Per-token GRPO loss (arXiv:2402.03300). `logits` covers the response
    plus one trailing position; `ref_logp`, `advantages`, and the mask are
    constants, matching fla."""
    logps = logits[:, :-1].log_softmax(-1).gather(-1, input_ids[:, 1:, None]).squeeze(-1)
    delta = ref_logp.detach() - logps
    kl = delta.exp() - delta - 1
    loss = -((logps - logps.detach()).exp() * advantages.detach()[:, None] - beta * kl)
    return loss if completion_mask is None else loss * completion_mask.detach()


# fla always returns the loss in fp32, so only float32 inputs round-trip.
@grpo.register("fla", source="fla.modules.grpo.fused_grpo_loss")
def grpo_fla(logits: Float32[Tensor, "batch resp_plus vocab"], ref_logp, input_ids, advantages, beta, completion_mask):
    return kernel(logits, ref_logp, input_ids, advantages, beta, completion_mask)
