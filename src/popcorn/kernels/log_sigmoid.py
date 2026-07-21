import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from popcorn import Tag, register_kernel


@register_kernel(test_args={"temperature": [1.0, 2.0]}, tags={Tag.ACTIVATION})
def log_sigmoid(x: Float[Tensor, "... hidden"], temperature: float = 1.0) -> Float[Tensor, "... hidden"]:
    r"""Temperature-scaled log-sigmoid.

    $$y = \frac{\log \sigma(x)}{\tau}$$
    """
    return F.logsigmoid(x) / temperature


log_sigmoid.register("fla", source="fla.modules.activations.logsigmoid")
