import importlib.util

import torch
from jaxtyping import Float, Float32, Int
from torch import Tensor

from popcorn import Tag, kernel, register_kernel
from popcorn.kernels._utils import upcast
from popcorn.kernels.grpo import _log_probability


# Returns (per-token loss, optional KL, clipping indicator), matching Liger's
# GrpoLossFunction.
@register_kernel(
    test_args={"temperature": [0.9], "beta": [0.0, 0.04], "eps_low": [0.2], "eps_high": [0.4]},
    test_inputs={
        "old_logp": _log_probability,
        "ref_logp": _log_probability,
        "completion_mask": lambda t: (t > 0).float(),
    },
    tags={Tag.LOSS, Tag.FUSED},
)
def grpo_offpolicy(
    logits: Float[Tensor, "batch response+1 vocab"],
    old_logp: Float[Tensor, "batch response"],
    ref_logp: Float[Tensor, "batch response"],
    completion_ids: Int[Tensor, "batch response"],
    advantages: Float[Tensor, "batch"],
    completion_mask: Float[Tensor, "batch response"] | None = None,
    temperature: float = 0.9,
    beta: float = 0.04,
    eps_low: float = 0.2,
    eps_high: float = 0.4,
) -> tuple[
    Float32[Tensor, "batch response"],
    Float32[Tensor, "batch response"] | None,
    Float32[Tensor, "batch response"],
]:
    r"""Clipped off-policy GRPO loss with a KL penalty against the reference policy.

    $$\mathcal{L}_t = -\min\!\big(r_t A,\; \mathrm{clip}(r_t, 1 - \epsilon_{\mathrm{lo}}, 1 + \epsilon_{\mathrm{hi}}) A\big)
    + \beta \big(e^{\delta_t} - \delta_t - 1\big), \qquad r_t = e^{\ell_t - \ell^{\mathrm{old}}_t}$$

    [PPO (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347),
    [GRPO (Shao et al., 2024)](https://arxiv.org/abs/2402.03300)
    """
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
        loss_kl = torch.expm1(delta) - delta
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


def _has_transformers(**_):
    """liger's grpo_loss imports transformers at call time and raises without it."""
    return importlib.util.find_spec("transformers") is not None


@grpo_offpolicy.register("liger", source="liger_kernel.transformers.grpo_loss.triton_grpo_loss", predicate=_has_transformers)
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
