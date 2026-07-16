import torch
import torch.nn.functional as F
from jaxtyping import Float, Float32
from torch import Tensor

from popcorn import Range, kernel, register_kernel
from popcorn.kernels._utils import upcast


def _standardize(x, eps):
    mean = x.mean(dim=-1, keepdim=True)
    std = (x.var(dim=-1, unbiased=False, keepdim=True) + eps).sqrt()
    return (x - mean) / std, std


@register_kernel(
    test_shapes={"batch": Range(1, 4), "seq": {32}, "heads": {2}, "head_dim": {16}},
    test_args={"eps": [1e-6], "chunk_size": [16]},
    test_inputs={
        "q": lambda t: F.normalize(t, dim=-1),
        "k": lambda t: F.normalize(t, dim=-1),
        "w": lambda t: 1 + 0.1 * t,
        "b": lambda t: 0.1 * t,
        "theta": lambda t: 0.1 * torch.sigmoid(t),
        "alpha": lambda t: 0.1 * torch.sigmoid(t),
        "eta": lambda t: 0.5 + 0.5 * torch.sigmoid(t),
        "initial_state": lambda t: 0.1 * t,
    },
)
def titans_linear(
    q: Float[Tensor, "batch seq heads head_dim"],
    k: Float[Tensor, "batch seq heads head_dim"],
    v: Float[Tensor, "batch seq heads head_dim"],
    w: Float[Tensor, "heads head_dim"],
    b: Float[Tensor, "heads head_dim"],
    theta: Float[Tensor, "batch seq heads"],
    alpha: Float[Tensor, "batch seq heads"],
    eta: Float[Tensor, "batch seq heads"],
    eps: float = 1e-6,
    chunk_size: int = 16,
    initial_state: Float[Tensor, "batch heads head_dim head_dim"] | None = None,
    output_final_state: bool = False,
) -> tuple[
    Float[Tensor, "batch seq heads head_dim"],
    Float[Tensor, "batch heads head_dim head_dim"] | None,
]:
    """Titans linear memory with chunk-local reconstruction gradients."""
    dtype = q.dtype
    q, k, v, w, b, theta, alpha, eta = map(upcast, (q, k, v, w, b, theta, alpha, eta))
    batch, seq, heads, head_dim = q.shape
    w = w.reshape(heads, 1, head_dim)
    b = b.reshape(heads, 1, head_dim)
    memory = q.new_zeros(batch, heads, head_dim, head_dim) if initial_state is None else upcast(initial_state)
    momentum = torch.zeros_like(memory)
    outputs = []

    for start in range(0, seq, chunk_size):
        stop = min(start + chunk_size, seq)
        q_chunk, k_chunk, v_chunk = (tensor[:, start:stop].transpose(1, 2) for tensor in (q, k, v))
        km = k_chunk @ memory
        km_hat, std = _standardize(km, eps)
        grad = (w * km_hat + b - (v_chunk - k_chunk)) * w
        v_new = head_dim * grad - grad.sum(dim=-1, keepdim=True) / (std * head_dim)
        v_new = v_new - km_hat * (grad * km_hat).sum(dim=-1, keepdim=True) / (std * head_dim)

        for offset in range(stop - start):
            position = start + offset
            item = slice(offset, offset + 1)
            scale = (slice(None), position, slice(None), None, None)
            update = k_chunk[:, :, item].transpose(-1, -2) @ v_new[:, :, item]
            momentum = eta[scale] * momentum - 2 * theta[scale] * update
            memory = (1 - alpha[scale]) * memory + momentum
            output = q_chunk[:, :, item] @ memory
            output_hat, _ = _standardize(output, eps)
            outputs.append(output + output_hat * w + b)

    output = torch.cat(outputs, dim=2).transpose(1, 2).to(dtype)
    final_state = memory.to(dtype) if output_final_state else None
    return output, final_state


def _aligned(**arguments):
    chunk_size = arguments["chunk_size"]
    return chunk_size > 0 and arguments["q"].shape[1] % chunk_size == 0


@titans_linear.register(
    "fla",
    source="fla.ops.titans.chunk_titans_linear",
    predicate=_aligned,
)
def titans_linear_fla(
    q: Float32[Tensor, "batch seq heads head_dim"],
    k,
    v,
    w,
    b,
    theta,
    alpha,
    eta,
    eps,
    chunk_size,
    initial_state,
    output_final_state,
):
    output, final_state = kernel(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        w,
        b,
        theta.transpose(1, 2)[..., None],
        alpha.transpose(1, 2)[..., None],
        eta.transpose(1, 2)[..., None],
        eps,
        chunk_size,
        initial_state,
        output_final_state,
    )
    return output.transpose(1, 2), final_state
