import functools
import importlib.util

import torch
import torch.nn.functional as F
from jaxtyping import BFloat16, Float
from torch import Tensor

from popcorn import Range, kernel, register_kernel
from popcorn.kernels._utils import upcast


@functools.cache
def _momoe_deps(**_):
    return all(importlib.util.find_spec(module) for module in ("einops", "triton"))


# The expert MLP is unit-init (weights scaled by 1/sqrt(fan_in), `fan_in` being
# the middle axis of each einsum-layout weight); without it the stacked matmuls
# blow the output to O(100) and bf16 rounding alone busts the floor.
@register_kernel(
    test_shapes={"batch": Range(1, 8), "seq": Range(2, 256), "dim": {64}, "intermediate": {128}, "experts": {8}},
    test_args={"top_k": [1, 6]},
    test_inputs={
        "gate_weight": lambda t: t / t.shape[1] ** 0.5,
        "up_weight": lambda t: t / t.shape[1] ** 0.5,
        "down_weight": lambda t: t / t.shape[1] ** 0.5,
    },
)
def momoe(
    x: Float[Tensor, "batch seq dim"],
    gate_weight: Float[Tensor, "experts dim intermediate"],
    up_weight: Float[Tensor, "experts dim intermediate"],
    down_weight: Float[Tensor, "experts intermediate dim"],
    router_logits: Float[Tensor, "batch seq experts"],
    top_k: int = 2,
) -> Float[Tensor, "batch seq dim"]:
    """Mixture of experts over SwiGLU MLPs: each token is routed to the `top_k`
    experts with the largest logits, and the expert outputs are blended with the
    picked softmax probabilities renormalized to sum 1 (softmax-then-topk; the
    selection itself is gradient-free). Weights are einsum-layout, per expert
    `down(silu(gate(x)) * up(x))` as in `swiglu_mlp`
    (github.com/tilde-research/momoe-release)."""
    probs = torch.softmax(upcast(router_logits), -1)
    picked = probs.topk(top_k, -1)
    gates = torch.zeros_like(probs).scatter(-1, picked.indices, picked.values / picked.values.sum(-1, keepdim=True))
    x32 = upcast(x)
    h = torch.einsum("bsd,edi->bsei", x32, upcast(up_weight)) * F.silu(torch.einsum("bsd,edi->bsei", x32, upcast(gate_weight)))
    y = torch.einsum("bsei,eid->bsed", h, upcast(down_weight))
    return torch.einsum("bsed,bse->bsd", y, gates).to(x.dtype)


# bf16 only: the kernels are hardwired to bfloat16 (inputs are force-cast and
# every intermediate buffer is allocated bf16 upstream).
@momoe.register("popcorn", source="popcorn.impls.momoe_tl.momoe", predicate=_momoe_deps)
def momoe_popcorn(
    x: BFloat16[Tensor, "batch seq dim"],
    gate_weight: BFloat16[Tensor, "experts dim intermediate"],
    up_weight: BFloat16[Tensor, "experts dim intermediate"],
    down_weight: BFloat16[Tensor, "experts intermediate dim"],
    router_logits: BFloat16[Tensor, "batch seq experts"],
    top_k,
):
    return kernel(x, torch.cat((up_weight, gate_weight), -1), down_weight, router_logits, top_k)
