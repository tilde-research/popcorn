import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from popcorn import register_kernel


@register_kernel(test_args={"temperature": [1.0, 2.0]})
def log_sigmoid(x: Float[Tensor, "... hidden"], temperature: float = 1.0) -> Float[Tensor, "... hidden"]:
    """Temperature-scaled log-sigmoid: `logsigmoid(x) / temperature`."""
    return F.logsigmoid(x) / temperature


log_sigmoid.register("fla", source="fla.modules.activations.logsigmoid")
