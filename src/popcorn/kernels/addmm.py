import torch
from jaxtyping import Float
from torch import Tensor

from popcorn import kernel, register_kernel
from popcorn.kernels._utils import upcast


@register_kernel(
    test_shapes={"rows": {17}, "inner": {16}, "cols": {19}},
    test_args={"alpha": [1.0, 0.5], "beta": [1.0, 0.25]},
)
def addmm(
    x: Float[Tensor, "rows cols"],
    a: Float[Tensor, "rows inner"],
    b: Float[Tensor, "inner cols"],
    alpha: float = 1.0,
    beta: float = 1.0,
) -> Float[Tensor, "rows cols"]:
    """Compute `beta * x + alpha * (a @ b)`."""
    return torch.addmm(upcast(x), upcast(a), upcast(b), beta=beta, alpha=alpha).to(a.dtype)


@addmm.register("fla", source="fla.ops.utils.addmm", forward_only=True)
def addmm_fla(x, a, b, alpha, beta):
    return kernel(
        x,
        a,
        b,
        alpha=torch.tensor(alpha, dtype=torch.float32, device=a.device),
        beta=torch.tensor(beta, dtype=torch.float32, device=a.device),
    )
