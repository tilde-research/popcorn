"""torch.library bindings: single-tensor ops from `register_kernel` are also
exposed as `torch.ops.popcorn.<name>`, so calls survive torch.compile and
export as one graph node instead of being traced into the dispatcher.

The real implementation routes through the dispatcher; the fake is the pure-aten
reference run on fake tensors. The binding is inference-only: gradients flow
through each backend's own autograd (they are torch.autograd.Functions or plain
torch code), which the custom-op boundary would discard -- so calls that require
grad raise and point at the direct call instead.
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


def bind_torch_op(op: Dispatcher) -> None:
    if getattr(op._signature.return_annotation, "dim_str", None) is None:
        return
    try:
        signature = _clean_signature(op)
    except TypeError as e:
        warnings.warn(f"{op.name}: not exposed as torch.ops.popcorn.{op.name}: {e}")
        return

    def call(*args: Any, **kwargs: Any) -> Any:
        return op(*args, **kwargs)

    def no_gradient(ctx: Any, inputs: Any, output: Any) -> None:
        raise RuntimeError(
            f"torch.ops.popcorn.{op.name} is inference-only; call {op.name} directly "
            "to train through the selected backend's own autograd"
        )

    call.__name__, call.__signature__ = op.name, signature
    torch_op = torch.library.custom_op(f"popcorn::{op.name}", call, mutates_args=())
    torch_op.register_fake(op.reference)
    torch_op.register_autograd(lambda ctx, grad: None, setup_context=no_gradient)
