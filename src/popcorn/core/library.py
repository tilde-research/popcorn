"""torch.library bindings: single-tensor ops from `register_kernel` are also
exposed as `torch.ops.popcorn.<name>`, so calls survive torch.compile and
export as one graph node instead of being traced into the dispatcher.

The real implementation routes through the dispatcher; the fake is the pure-aten
reference run on fake tensors. Backends own their autograd as opaque
torch.autograd.Functions, which a custom-op boundary cannot reuse, so we cannot
thread their saved tensors across the boundary.

Strategies:

- `torch.ops.popcorn.<name>`: opaque forward through the dispatcher. Backward
  replays the forward on detached leaves (one extra forward). Used for
  inference rewrites and direct calls.
- Scalar-output ops also get `torch.ops.popcorn.<name>_vg`: forward returns
  `(output, *unit_grads)` from one dispatcher call, backward is
  `grad_out * unit_grad`. Compile's joint (training) patterns use this via
  `op.torch_op_train`, matching eager popcorn cost — no second forward.
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


def _with_autograd_keys():
    """Custom-op kernels run with Autograd excluded in the dispatch TLS; lift
    those exclusions so a nested dispatcher call can record a graph."""
    include = torch._C._dispatch_tls_local_include_set()
    exclude = torch._C._dispatch_tls_local_exclude_set()
    for key in ("AutogradCPU", "AutogradCUDA", "AutogradOther", "ADInplaceOrView"):
        exclude = exclude.remove(getattr(torch._C.DispatchKey, key))
    return torch._C._ForceDispatchKeyGuard(include, exclude)


def _bind_backward(op: Dispatcher, signature: inspect.Signature, tensor_at: list[int]) -> Any:
    """`popcorn::<name>_backward`: grads of the needed tensor params, by forward
    replay through the dispatcher."""

    def replay(*all_args: Any) -> list[torch.Tensor]:
        grad_output, *arguments, needs_input_grad = all_args
        with _with_autograd_keys(), torch.enable_grad():
            leaves = []
            for i, needed in zip(tensor_at, needs_input_grad):
                if isinstance(arguments[i], torch.Tensor):
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


def _bind_public(op: Dispatcher, signature: inspect.Signature, tensor_at: list[int], scalar_at: list[int]) -> Any:
    """`torch.ops.popcorn.<name>`: single Tensor out, replay backward."""

    def call(*args: Any, **kwargs: Any) -> Any:
        return op(*args, **kwargs)

    call.__name__, call.__signature__ = op.name, signature  # type: ignore[attr-defined]
    torch_op = torch.library.custom_op(f"popcorn::{op.name}", call, mutates_args=())
    torch_op.register_fake(op.reference)
    backward_op = _bind_backward(op, signature, tensor_at)

    def setup_context(ctx: Any, inputs: tuple[Any, ...], output: Any) -> None:
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
    return torch_op


def _bind_unit_vjp(op: Dispatcher, signature: inspect.Signature, tensor_at: list[int]) -> Any:
    """`torch.ops.popcorn.<name>_vg` + wrapper: scalar training path without replay."""

    def _pack(*args: Any, **kwargs: Any) -> tuple[Any, ...]:
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        return bound.args

    def _placeholder(value: Any, like: torch.Tensor) -> torch.Tensor:
        """Unit-grad slot for an optional tensor that is absent."""
        return torch.zeros_like(value) if isinstance(value, torch.Tensor) else like.new_empty(0)

    def call(*args: Any, **kwargs: Any) -> tuple[torch.Tensor, ...]:
        values = _pack(*args, **kwargs)
        needs = [isinstance(values[i], torch.Tensor) and values[i].requires_grad for i in tensor_at]
        ref = next(values[i] for i in tensor_at if isinstance(values[i], torch.Tensor))
        with _with_autograd_keys(), torch.enable_grad():
            local = list(values)
            leaves: list[torch.Tensor] = []
            for i, needed in zip(tensor_at, needs):
                if isinstance(local[i], torch.Tensor):
                    local[i] = local[i].detach().requires_grad_(needed)
                if needed:
                    leaves.append(local[i])
            out = op(*local)
            if not leaves:
                return (out.detach(), *(_placeholder(values[i], ref) for i in tensor_at))
            grads = torch.autograd.grad(out, leaves, torch.ones_like(out))
        unit = []
        it = iter(grads)
        for i, needed in zip(tensor_at, needs):
            unit.append(next(it).detach() if needed else _placeholder(values[i], ref))
        return (out.detach(), *unit)

    def fake(*args: Any, **kwargs: Any) -> tuple[torch.Tensor, ...]:
        values = _pack(*args, **kwargs)
        out = op.reference(*values)
        ref = next(values[i] for i in tensor_at if isinstance(values[i], torch.Tensor))
        return (out, *(_placeholder(values[i], ref) for i in tensor_at))

    # Fixed-length tuple (not Tensor[]) so autograd packs one grad per output.
    ret = tuple[(torch.Tensor,) * (1 + len(tensor_at))]  # type: ignore[misc, valid-type]
    call.__name__ = f"{op.name}_vg"
    call.__signature__ = fake.__signature__ = signature.replace(return_annotation=ret)  # type: ignore[attr-defined]
    multi = torch.library.custom_op(f"popcorn::{op.name}_vg", call, mutates_args=())
    multi.register_fake(fake)

    def setup_context(ctx: Any, inputs: tuple[Any, ...], output: tuple[torch.Tensor, ...]) -> None:
        # output is (loss, *unit_grads); only the loss is consumed downstream, so
        # unit_grads are saved for backward and their incoming grads are ignored.
        ctx.save_for_backward(*output[1:])
        ctx.popcorn_needs = [isinstance(inputs[i], torch.Tensor) and inputs[i].requires_grad for i in tensor_at]

    def backward(ctx: Any, grad_loss: torch.Tensor, *grad_units: Any) -> tuple[Any, ...]:
        result: list[Any] = [None] * len(op._params)
        for i, needed, unit in zip(tensor_at, ctx.popcorn_needs, ctx.saved_tensors):
            if needed:
                result[i] = unit * grad_loss
        return tuple(result)

    multi.register_autograd(backward, setup_context=setup_context)

    def torch_op_train(*args: Any, **kwargs: Any) -> torch.Tensor:
        return multi(*_pack(*args, **kwargs))[0]

    torch_op_train.__name__ = op.name  # type: ignore[attr-defined]
    torch_op_train.__signature__ = signature  # type: ignore[attr-defined]
    return torch_op_train


def bind_torch_op(op: Dispatcher) -> None:
    dim_str = getattr(op._signature.return_annotation, "dim_str", None)
    if dim_str is None:
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

    op.torch_op = _bind_public(op, signature, tensor_at, scalar_at)
    # Scalar losses: compile's joint patterns use the unit-VJP binding.
    op.torch_op_train = _bind_unit_vjp(op, signature, tensor_at) if dim_str == "" else op.torch_op
