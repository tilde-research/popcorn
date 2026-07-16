"""Shape grammar parsed from jaxtyping annotations on kernel references."""

import inspect
import typing
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import EllipsisType
from typing import Any

from popcorn.core.errors import DispatchError

DimToken = str | int | EllipsisType


@dataclass(frozen=True, slots=True)
class TensorSpec:
    """One tensor parameter's parsed shape tokens, accepted dtypes, and optionality."""

    param: str
    tokens: tuple[DimToken, ...]
    dtypes: tuple[str, ...]
    optional: bool


def _unwrap_optional(annotation: Any) -> tuple[Any, bool]:
    args = typing.get_args(annotation)
    if args and type(None) in args:
        rest = [a for a in args if a is not type(None)]
        if len(rest) == 1:
            return rest[0], True
    return annotation, False


def _token(text: str, param: str) -> DimToken:
    if text == "...":
        return ...
    if text.isdigit():
        return int(text)
    if text.isidentifier():
        return text
    raise TypeError(f"{param}: unsupported dim token {text!r}; references may use names, ints, and '...'")


def plans(signature: inspect.Signature) -> tuple[TensorSpec, ...]:
    specs = []
    for name, param in signature.parameters.items():
        annotation, optional = _unwrap_optional(param.annotation)
        dim_str = getattr(annotation, "dim_str", None)
        if dim_str is None:
            continue
        tokens = tuple(_token(t, name) for t in dim_str.split())
        if tokens.count(...) > 1:
            raise TypeError(f"{name}: at most one '...' per annotation")
        specs.append(TensorSpec(name, tokens, getattr(annotation, "dtypes", ()), optional))
    return tuple(specs)


def dim_names(specs: Iterable[TensorSpec]) -> set[str]:
    return {t for spec in specs for t in spec.tokens if isinstance(t, str)}


def extract(specs: Iterable[TensorSpec], args: Mapping[str, Any]) -> dict[str, int]:
    """Bind dim names to sizes from actual shapes, checking cross-argument consistency."""
    dims: dict[str, int] = {}
    for spec in specs:
        tensor = args.get(spec.param)
        if tensor is None:
            continue
        shape = tuple(tensor.shape)
        tokens = spec.tokens
        if ... in tokens:
            i = tokens.index(...)
            head, tail = tokens[:i], tokens[i + 1 :]
            if len(shape) < len(head) + len(tail):
                raise DispatchError(f"{spec.param}: expected at least {len(head) + len(tail)} dims, got shape {shape}")
            pairs = list(zip(head, shape)) + list(zip(tail, shape[len(shape) - len(tail) :]))
        else:
            if len(shape) != len(tokens):
                raise DispatchError(f"{spec.param}: expected {len(tokens)} dims, got shape {shape}")
            pairs = list(zip(tokens, shape))
        for token, size in pairs:
            if isinstance(token, int):
                if token != size:
                    raise DispatchError(f"{spec.param}: expected dim of {token}, got {size}")
            elif isinstance(token, str) and dims.setdefault(token, size) != size:
                raise DispatchError(f"dim {token!r}: {spec.param} has {size}, other arguments have {dims[token]}")
    return dims


def batch_shape(specs: Iterable[TensorSpec], args: Mapping[str, Any]) -> tuple[int, ...] | None:
    """The shared `...` shape, or None when variadic tensor arguments differ."""
    batches = []
    for spec in specs:
        if ... not in spec.tokens or (tensor := args.get(spec.param)) is None:
            continue
        split = spec.tokens.index(...)
        tail = len(spec.tokens) - split - 1
        shape = tuple(tensor.shape)
        batches.append(shape[split : len(shape) - tail if tail else None])
    return batches[0] if batches and all(batch == batches[0] for batch in batches) else None
