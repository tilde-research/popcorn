import torch
from jaxtyping import Bool, Float, Float32, Int
from torch import Tensor

from popcorn import Tag, register_kernel
from popcorn.kernels._utils import upcast


# logits carry one extra position, Liger's next-token layout.
@register_kernel(
    test_args={"temperature": [0.9, 1.0]},
    test_inputs={"mask": lambda t: t.bool()},
    tags={Tag.REDUCTION, Tag.FUSED},
)
def selective_log_softmax(
    logits: Float[Tensor, "batch response+1 vocab"],
    input_ids: Int[Tensor, "batch response"],
    temperature: float = 0.9,
    mask: Bool[Tensor, "batch response"] | None = None,
) -> Float32[Tensor, "batch response"]:
    r"""Next-token log probabilities of the selected ids, computed in float32.

    $$y_t = \frac{x_{t, i_t}}{\tau} - \log \sum_v e^{x_{t, v} / \tau}$$
    """
    scaled = upcast(logits[:, :-1]) / temperature
    selected = scaled.gather(-1, input_ids[..., None]).squeeze(-1)
    log_probs = selected - torch.logsumexp(scaled, dim=-1)
    return log_probs if mask is None else log_probs.masked_fill(~mask, 0)


selective_log_softmax.register(
    "liger",
    source="liger_kernel.ops.grpo_loss.fused_selective_log_softmax",
    forward_only=True,
)
