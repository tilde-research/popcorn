"""Shape grammar parsed from jaxtyping annotations on kernel references."""

import inspect
import re
import typing
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import EllipsisType
from typing import Any

from popcorn.core.errors import DispatchError


@dataclass(frozen=True, slots=True)
class Derived:
    """A dim derived from another: `base (+|-|*|/) operand`, the operand an int
    literal or a scalar parameter. Derived sizes are never their own dims: they
    are computed by the grid and validated (or inverted to bind the base) at
    call time."""

    base: str
    op: str
    operand: int | str

    def __str__(self) -> str:
        return f"{self.base}{self.op}{self.operand}"


DimToken = str | int | EllipsisType | Derived

_EXPRESSION = re.compile(r"^(?P<base>[^\W\d]\w*)(?P<op>[+\-*/])(?P<operand>\w+)$")


@dataclass(frozen=True, slots=True)
class TensorSpec:
    """One tensor annotation's parsed shape tokens, accepted dtypes, and optionality."""

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
    if match := _EXPRESSION.match(text):
        operand = match["operand"]
        return Derived(match["base"], match["op"], int(operand) if operand.isdigit() else operand)
    raise TypeError(
        f"{param}: unsupported dim token {text!r}; references may use names, ints, '...', "
        "and one derivation `name(+|-|*|/)k` with k an int or a scalar parameter"
    )


def _spec(name: str, annotation: Any) -> TensorSpec | None:
    annotation, optional = _unwrap_optional(annotation)
    dim_str = getattr(annotation, "dim_str", None)
    if dim_str is None:
        return None
    tokens = tuple(_token(token, name) for token in dim_str.split())
    if tokens.count(...) > 1:
        raise TypeError(f"{name}: at most one '...' per annotation")
    return TensorSpec(name, tokens, getattr(annotation, "dtypes", ()), optional)


def _validate(signature: inspect.Signature, specs: Iterable[TensorSpec], tensors: set[str]) -> None:
    for spec in specs:
        for token in spec.tokens:
            if isinstance(token, Derived) and isinstance(token.operand, str) and token.operand not in signature.parameters:
                raise TypeError(f"{spec.param}: dim {token}: operand {token.operand!r} is not a parameter")
            if isinstance(token, Derived) and token.operand in tensors:
                raise TypeError(f"{spec.param}: dim {token}: operand {token.operand!r} must be a scalar parameter")


def plans(signature: inspect.Signature) -> tuple[TensorSpec, ...]:
    specs = tuple(spec for name, param in signature.parameters.items() if (spec := _spec(name, param.annotation)) is not None)
    tensors = {spec.param for spec in specs}
    _validate(signature, specs, tensors)
    return specs


def return_plans(signature: inspect.Signature) -> tuple[TensorSpec, ...]:
    """Parsed tensor shapes in a reference's return annotation."""
    specs = []

    def visit(annotation: Any) -> None:
        if spec := _spec(f"out{len(specs)}", annotation):
            specs.append(spec)
        elif typing.get_origin(annotation) is tuple:
            for member in typing.get_args(annotation):
                if member is not ...:
                    visit(member)

    visit(signature.return_annotation)
    _validate(signature, specs, {spec.param for spec in plans(signature)})
    return tuple(specs)


def dim_names(specs: Iterable[TensorSpec]) -> set[str]:
    names = set()
    for spec in specs:
        for token in spec.tokens:
            if isinstance(token, str):
                names.add(token)
            elif isinstance(token, Derived):
                names.add(token.base)
    return names


def _operand(token: Derived, args: Mapping[str, Any], param: str) -> int:
    value = token.operand if isinstance(token.operand, int) else args.get(token.operand)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise DispatchError(f"{param}: dim {token} needs a positive int {token.operand!r}, got {value!r}")
    return value


def _apply(token: Derived, base: int, k: int, param: str) -> int:
    match token.op:
        case "+":
            return base + k
        case "-":
            return base - k
        case "*":
            return base * k
        case _:
            if base % k:
                raise DispatchError(f"{param}: dim {token}: {token.base}={base} is not divisible by {k}")
            return base // k


def _invert(token: Derived, size: int, k: int, param: str) -> int:
    match token.op:
        case "+":
            return size - k
        case "-":
            return size + k
        case "*":
            if size % k:
                raise DispatchError(f"{param}: dim {token}: size {size} is not a multiple of {k}")
            return size // k
        case _:
            return size * k


def token_size(token: DimToken, dims: Mapping[str, int], args: Mapping[str, Any], param: str = "?") -> int:
    """The concrete size of one non-variadic token given bound dims and scalar arguments."""
    if isinstance(token, int):
        return token
    if isinstance(token, str):
        return dims[token]
    assert isinstance(token, Derived)
    return _apply(token, dims[token.base], _operand(token, args, param), param)


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
            elif isinstance(token, str):
                if dims.setdefault(token, size) != size:
                    raise DispatchError(f"dim {token!r}: {spec.param} has {size}, other arguments have {dims[token]}")
            elif isinstance(token, Derived):
                k = _operand(token, args, spec.param)
                if token.base in dims:
                    expected = _apply(token, dims[token.base], k, spec.param)
                    if size != expected:
                        raise DispatchError(
                            f"dim {token}: {spec.param} has {size}, expected {expected} ({token.base}={dims[token.base]})"
                        )
                else:
                    dims[token.base] = _invert(token, size, k, spec.param)
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
