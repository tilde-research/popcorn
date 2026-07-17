"""torch.library bindings: single-tensor ops from `register_kernel` are also
exposed as `torch.ops.popcorn.<name>`, so calls survive torch.compile and
export as one graph node instead of being traced into the dispatcher.

The real implementation routes through the dispatcher; the fake is the pure-aten
reference run on fake tensors. Backends own their autograd as opaque
torch.autograd.Functions, which a custom-op boundary cannot reuse, so
`popcorn::<name>_backward` replays the forward on detached leaves under
`enable_grad` and differentiates through whichever backend the dispatcher
selects -- one extra forward per backward. Each direction picks its own
backend: the forward runs with grad off, keeping forward-only backends
eligible; the replay needs grad, which excludes them.
"""

from __future__ import annotations

import inspect
import types
import typing
import warnings
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from popcorn.core.dispatcher import Dispatcher

_SCALARS = (bool, int, float, str)


def _scalar(annotation: Any, default: Any) -> Any:
    if typing.get_origin(annotation) is typing.Literal:
        annotation = type(typing.get_args(annotation)[0])
    if typing.get_origin(annotation) in (typing.Union, types.UnionType):
        rest = [a for a in typing.get_args(annotation) if a is not type(None)]
        if len(rest) == 1 and rest[0] in _SCALARS:
            return typing.Optional[rest[0]]
    if annotation in _SCALARS:
        return annotation
    if annotation is inspect.Parameter.empty and type(default) in _SCALARS:
        return type(default)
    raise TypeError(f"cannot express {annotation!r} in a torch op schema")


def _clean_signature(op: Dispatcher) -> inspect.Signature:
    """The reference signature with jaxtyping annotations lowered to plain
    Tensor/scalar types, which torch.library can infer a schema from."""
    tensor = {s.param: typing.Optional[torch.Tensor] if s.optional else torch.Tensor for s in op.specs}
    params = [
        p.replace(annotation=tensor[name] if name in tensor else _scalar(p.annotation, p.default))
        for name, p in op._signature.parameters.items()
    ]
    return inspect.Signature(params, return_annotation=torch.Tensor)


def _bind_backward(op: Dispatcher, signature: inspect.Signature, tensor_at: list[int]) -> Any:
    """`popcorn::<name>_backward`: grads of the needed tensor params, by forward
    replay through the dispatcher."""

    def replay(*all_args: Any) -> list[torch.Tensor]:
        grad_output, *arguments, needs_input_grad = all_args
        # The impl runs with Autograd excluded in the dispatch TLS, where
        # nothing records a graph; lift just those exclusions so the replay is
        # differentiable (autocast and the rest stay as the dispatcher set them).
        include = torch._C._dispatch_tls_local_include_set()
        exclude = torch._C._dispatch_tls_local_exclude_set()
        for key in ("AutogradCPU", "AutogradCUDA", "AutogradOther", "ADInplaceOrView"):
            exclude = exclude.remove(getattr(torch._C.DispatchKey, key))
        with torch._C._ForceDispatchKeyGuard(include, exclude), torch.enable_grad():
            leaves = []
            for i, needed in zip(tensor_at, needs_input_grad):
                if isinstance(arguments[i], torch.Tensor):  # optional tensors may be None
                    arguments[i] = arguments[i].detach().requires_grad_(needed)
                if needed:
                    leaves.append(arguments[i])
            output = op(*arguments)
            return list(torch.autograd.grad(output, leaves, grad_output))

    def fake(*all_args: Any) -> list[torch.Tensor]:
        _, *arguments, needs_input_grad = all_args
        return [torch.empty_like(arguments[i]) for i, needed in zip(tensor_at, needs_input_grad) if needed]

    params = [
        inspect.Parameter("grad_output", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=torch.Tensor),
        *(p.replace(default=inspect.Parameter.empty) for p in signature.parameters.values()),
        inspect.Parameter("needs_input_grad", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=list[bool]),
    ]
    replay.__name__ = f"{op.name}_backward"
    replay.__signature__ = fake.__signature__ = inspect.Signature(params, return_annotation=list[torch.Tensor])  # type: ignore[attr-defined]
    backward_op = torch.library.custom_op(f"popcorn::{op.name}_backward", replay, mutates_args=())
    backward_op.register_fake(fake)
    return backward_op


def bind_torch_op(op: Dispatcher) -> None:
    if getattr(op._signature.return_annotation, "dim_str", None) is None:
        return
    try:
        signature = _clean_signature(op)
    except TypeError as e:
        warnings.warn(f"{op.name}: not exposed as torch.ops.popcorn.{op.name}: {e}")
        return

    if {"grad_output", "needs_input_grad"} & set(op._params):
        warnings.warn(f"{op.name}: not exposed as torch.ops.popcorn.{op.name}: reserved parameter name")
        return

    tensors = frozenset(spec.param for spec in op.specs)
    tensor_at = [i for i, name in enumerate(op._params) if name in tensors]
    scalar_at = [i for i, name in enumerate(op._params) if name not in tensors]

    def call(*args: Any, **kwargs: Any) -> Any:
        return op(*args, **kwargs)

    call.__name__, call.__signature__ = op.name, signature  # type: ignore[attr-defined]
    torch_op = torch.library.custom_op(f"popcorn::{op.name}", call, mutates_args=())
    torch_op.register_fake(op.reference)
    backward_op = _bind_backward(op, signature, tensor_at)

    def setup_context(ctx: Any, inputs: tuple[Any, ...], output: Any) -> None:
        # inputs arrive positional with defaults filled in
        ctx.save_for_backward(*(inputs[i] for i in tensor_at))
        ctx.popcorn_scalars = tuple(inputs[i] for i in scalar_at)
        ctx.popcorn_needs = [isinstance(inputs[i], torch.Tensor) and inputs[i].requires_grad for i in tensor_at]

    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[Any, ...]:
        args: list[Any] = [None] * len(op._params)
        for i, value in zip(tensor_at, ctx.saved_tensors):
            args[i] = value
        for i, value in zip(scalar_at, ctx.popcorn_scalars):
            args[i] = value
        grads = iter(backward_op(grad_output, *args, ctx.popcorn_needs))
        result: list[Any] = [None] * len(op._params)
        for i, needed in zip(tensor_at, ctx.popcorn_needs):
            if needed:
                result[i] = next(grads)
        return tuple(result)

    torch_op.register_autograd(backward, setup_context=setup_context)
    op.torch_op = torch_op
