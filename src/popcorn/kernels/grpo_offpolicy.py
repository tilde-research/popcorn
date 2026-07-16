import torch
from jaxtyping import Float, Float32, Int
from torch import Tensor

from popcorn import Range, kernel, register_kernel
from popcorn.kernels._utils import upcast


@register_kernel(
    test_shapes={"batch": Range(1, 8), "resp": {31}, "resp_plus": {32}, "vocab": Range(2, 4096)},
    test_args={"temperature": [0.9], "beta": [0.0, 0.04], "eps_low": [0.2], "eps_high": [0.4]},
    test_inputs={"completion_mask": lambda t: (t > 0).float()},
)
def grpo_offpolicy(
    logits: Float[Tensor, "batch resp_plus vocab"],
    old_logp: Float[Tensor, "batch resp"],
    ref_logp: Float[Tensor, "batch resp"],
    completion_ids: Int[Tensor, "batch resp"],
    advantages: Float[Tensor, "batch"],
    completion_mask: Float[Tensor, "batch resp"] | None = None,
    temperature: float = 0.9,
    beta: float = 0.04,
    eps_low: float = 0.2,
    eps_high: float = 0.4,
) -> tuple[
    Float32[Tensor, "batch resp"],
    Float32[Tensor, "batch resp"] | None,
    Float32[Tensor, "batch resp"],
]:
    """Clipped off-policy GRPO loss. Returns per-token loss, optional KL, and
    the clipping indicator, matching Liger's `GrpoLossFunction`."""
    logp = (upcast(logits[:, :-1]) / temperature).log_softmax(-1)
    logp = logp.gather(-1, completion_ids[..., None]).squeeze(-1)
    ratio = (logp - upcast(old_logp).detach()).exp()
    advantage = upcast(advantages).detach()[:, None]
    unclipped = ratio * advantage
    clipped = ratio.clamp(1 - eps_low, 1 + eps_high) * advantage
    loss = -torch.minimum(unclipped, clipped)
    is_clipped = (((ratio < 1 - eps_low) & (advantage < 0)) | ((ratio > 1 + eps_high) & (advantage > 0))).to(logp.dtype)

    kl = None
    if beta != 0.0:
        delta = upcast(ref_logp).detach() - logp
        loss_kl = delta.exp() - delta - 1
        loss = loss + beta * loss_kl
        kl = loss_kl.detach() + logp * 0

    is_clipped = is_clipped + logp * 0
    if completion_mask is not None:
        mask = upcast(completion_mask).detach()
        loss = loss * mask
        is_clipped = is_clipped * mask
        if kl is not None:
            kl = kl * mask
    return loss, kl, is_clipped


@grpo_offpolicy.register("liger", source="liger_kernel.transformers.grpo_loss.triton_grpo_loss")
def grpo_offpolicy_liger(
    logits,
    old_logp,
    ref_logp,
    completion_ids,
    advantages,
    completion_mask,
    temperature,
    beta,
    eps_low,
    eps_high,
):
    return kernel(
        logits,
        old_logp,
        ref_logp,
        completion_ids,
        advantages,
        completion_mask,
        temperature,
        beta,
        eps_low,
        eps_high,
        inplace=False,
        loss_type="grpo",
        reduce=False,
    )
