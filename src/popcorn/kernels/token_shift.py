import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from popcorn import Range, register_kernel


@register_kernel(test_shapes={"batch": Range(1, 8), "seq": Range(2, 512), "hidden": Range(8, 4096)})
def token_shift(x: Float[Tensor, "batch seq hidden"]) -> Float[Tensor, "batch seq hidden"]:
    """RWKV token shift: the previous token's features minus the current ones
    (the first position sees zeros)."""
    return F.pad(x, (0, 0, 1, -1)) - x


token_shift.register("fla", source="fla.modules.token_shift.token_shift")
