import torch
from jaxtyping import Bool, Float, Float32, Int
from torch import Tensor

from popcorn import Range, register_kernel
from popcorn.kernels._utils import upcast


# response_plus must equal response + 1; singleton pools keep the grid
# consistent with Liger's next-token layout.
@register_kernel(
    test_shapes={"batch": Range(1, 8), "response": {31}, "response_plus": {32}, "vocab": Range(2, 4096)},
    test_args={"temperature": [0.9, 1.0]},
    test_inputs={"mask": lambda t: t.bool()},
)
def selective_log_softmax(
    logits: Float[Tensor, "batch response_plus vocab"],
    input_ids: Int[Tensor, "batch response"],
    temperature: float = 0.9,
    mask: Bool[Tensor, "batch response"] | None = None,
) -> Float32[Tensor, "batch response"]:
    """Selected next-token log probabilities, computed in float32."""
    scaled = upcast(logits[:, :-1]) / temperature
    selected = scaled.gather(-1, input_ids[..., None]).squeeze(-1)
    log_probs = selected - torch.logsumexp(scaled, dim=-1)
    return log_probs if mask is None else log_probs.masked_fill(~mask, 0)


selective_log_softmax.register(
    "liger",
    source="liger_kernel.ops.grpo_loss.fused_selective_log_softmax",
    forward_only=True,
)
