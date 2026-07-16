"""Shape and value constraint specs used by `supports=` and the test grid."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True, slots=True)
class Range:
    """Inclusive integer range constraint."""

    lo: int
    hi: int

    def __call__(self, value: int) -> bool:
        return self.lo <= value <= self.hi

    def __str__(self) -> str:
        return f"in [{self.lo}, {self.hi}]"


@dataclass(frozen=True, slots=True)
class Div:
    """Divisibility constraint."""

    n: int

    def __call__(self, value: int) -> bool:
        return value % self.n == 0

    def __str__(self) -> str:
        return f"divisible by {self.n}"


@dataclass(frozen=True, slots=True)
class Pow2:
    """Power-of-two constraint."""

    def __call__(self, value: int) -> bool:
        return value > 0 and value & (value - 1) == 0

    def __str__(self) -> str:
        return "a power of two"


def satisfies(spec: Any, value: Any) -> bool:
    match spec:
        case None:
            return value is None
        case tuple():
            return all(satisfies(s, value) for s in spec)
        case set() | frozenset():
            return not isinstance(value, torch.Tensor) and value in spec
        case _ if callable(spec):
            return bool(spec(value))
        case _:
            # Tensors only match None or predicate specs; == would compare elementwise.
            return not isinstance(value, torch.Tensor) and value == spec


def describe(spec: Any) -> str:
    match spec:
        case tuple():
            return " and ".join(describe(s) for s in spec)
        case set() | frozenset():
            return f"in {{{', '.join(sorted(str(s) for s in spec))}}}"
        case Range() | Div() | Pow2():
            return str(spec)
        case _ if callable(spec):
            return f"accepted by {getattr(spec, '__name__', 'predicate')}"
        case _:
            return repr(spec)


def fmt_value(value: Any) -> str:
    if isinstance(value, torch.Tensor):
        return f"Tensor{tuple(value.shape)}"
    return repr(value)


def check(supports: Mapping[str, Any], values: Mapping[str, Any]) -> str | None:
    """Return the first rejection among constraint keys present in `values`."""
    for key, spec in supports.items():
        if key in values and not satisfies(spec, values[key]):
            return f"{key}={fmt_value(values[key])} is not {describe(spec)}"
    return None
