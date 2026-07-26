"""Integer set algebra for declared pools and learned validity regions.

One frozen `Space` speaks for `DIMS` pools, fitted regions, and rejection text:
`64 in space`, `str(space)`, and the set operators `% | & - ^` are the whole API.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass(frozen=True, slots=True)
class Space:
    """An integer set: bounds, a membership test, and its own printed form."""

    lo: int
    hi: int
    test: Callable[[int], bool] = field(repr=False, compare=False, hash=False)
    text: str
    # Explicit members that must survive `grid` (curated sets and set-sides of unions).
    # `None` means "dense band — log-sample only".
    points: frozenset[int] | None = None

    def __contains__(self, value: object) -> bool:
        return isinstance(value, int) and not isinstance(value, bool) and self.lo <= value <= self.hi and self.test(value)

    def __str__(self) -> str:
        return self.text

    def __repr__(self) -> str:
        return f"Space({self.text!r})"

    def __mod__(self, n: int) -> Space:
        if n < 1:
            raise ValueError("modulus must be positive")
        points = None if self.points is None else frozenset(v for v in self.points if v % n == 0)
        return Space(self.lo, self.hi, lambda v, t=self.test, m=n: t(v) and v % m == 0, f"{self.text} % {n}", points)

    def __or__(self, other: Any) -> Space:
        other = space(other)
        points = (
            None
            if self.points is None and other.points is None
            else (self.points or frozenset()) | (other.points or frozenset())
        )
        return Space(
            min(self.lo, other.lo),
            max(self.hi, other.hi),
            lambda v, a=self, b=other: v in a or v in b,
            f"({self.text} | {other.text})",
            points,
        )

    def __and__(self, other: Any) -> Space:
        other = space(other)
        if self.points is not None and other.points is not None:
            points: frozenset[int] | None = self.points & other.points
        elif self.points is not None:
            points = frozenset(v for v in self.points if v in other)
        elif other.points is not None:
            points = frozenset(v for v in other.points if v in self)
        else:
            points = None
        return Space(
            max(self.lo, other.lo),
            min(self.hi, other.hi),
            lambda v, a=self, b=other: v in a and v in b,
            f"({self.text} & {other.text})",
            points,
        )

    def __sub__(self, other: Any) -> Space:
        other = space(other)
        points = None if self.points is None else frozenset(v for v in self.points if v not in other)
        return Space(
            self.lo, self.hi, lambda v, a=self, b=other: v in a and v not in b, f"({self.text} - {other.text})", points
        )

    def __xor__(self, other: Any) -> Space:
        other = space(other)
        points = (
            None
            if self.points is None and other.points is None
            else ((self.points or frozenset()) | (other.points or frozenset()))
            - ((self.points or frozenset()) & (other.points or frozenset()))
        )
        return Space(
            min(self.lo, other.lo),
            max(self.hi, other.hi),
            lambda v, a=self, b=other: (v in a) ^ (v in b),
            f"({self.text} ^ {other.text})",
            points,
        )

    __ror__ = __or__
    __rand__ = __and__
    __rxor__ = __xor__

    def grid(self, n: int = 10, seeds: Iterable[int] = ()) -> list[int]:
        """Log-spaced probes plus any curated `points` (sets / sparse ladders)."""
        candidates = {self.lo, self.hi, *seeds, *(self.points or ())}
        if self.hi > self.lo and n > 2:
            log_lo, log_hi = math.log2(max(1, self.lo)), math.log2(max(1, self.hi))
            for index in range(n):
                candidates.add(round(2 ** (log_lo + index / (n - 1) * (log_hi - log_lo))))
        mid = max(1, round(math.sqrt(max(1, self.lo) * max(1, self.hi))))
        candidates.update(mid + delta for delta in (-1, 0, 1))
        found = sorted(value for value in candidates if value in self)
        if not found:
            raise ValueError(f"space {self} admits no sizes")
        return found

    def sample(self, k: int, rng: random.Random | None = None) -> list[int]:
        """Up to `k` log-uniform draws by rejection."""
        rng = rng or random.Random()
        if k < 1 or self.hi < self.lo:
            return []
        log_lo, log_hi = math.log2(max(1, self.lo)), math.log2(max(1, self.hi))
        found: list[int] = []
        for _ in range(max(k * 64, k)):
            if len(found) >= k:
                break
            value = min(self.hi, max(self.lo, round(2 ** rng.uniform(log_lo, log_hi))))
            if value in self:
                found.append(value)
        return found


def Range(lo: int, hi: int) -> Space:
    """Inclusive integer band."""
    if lo > hi:
        raise ValueError(f"empty range [{lo}, {hi}]")
    return Space(lo, hi, lambda _v: True, f"[{lo}, {hi}]", None)


@dataclass(frozen=True, slots=True)
class Real:
    """Closed real interval minus optional exact holes — continuous-arg twin of `Space`."""

    lo: float
    hi: float
    drops: frozenset[float] = frozenset()

    def __post_init__(self) -> None:
        if self.hi < self.lo:
            raise ValueError(f"empty real [{self.lo}, {self.hi}]")

    def __contains__(self, value: object) -> bool:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        number = float(value)
        return self.lo <= number <= self.hi and number not in self.drops

    def __str__(self) -> str:
        text = f"[{self.lo}, {self.hi}]"
        if self.drops:
            text += " - {" + ", ".join(map(str, sorted(self.drops))) + "}"
        return text

    def __repr__(self) -> str:
        return f"Real({self!s})"

    def grid(self, n: int = 10, seeds: Iterable[float] = ()) -> list[float]:
        """Log-spaced probes when the band is positive; otherwise linear."""
        candidates = {self.lo, self.hi, *seeds}
        if self.hi > self.lo and n > 2:
            if self.lo > 0:
                log_lo, log_hi = math.log2(self.lo), math.log2(self.hi)
                for index in range(n):
                    candidates.add(2 ** (log_lo + index / (n - 1) * (log_hi - log_lo)))
            else:
                for index in range(n):
                    candidates.add(self.lo + index / (n - 1) * (self.hi - self.lo))
        found = sorted(value for value in candidates if value in self)
        if not found:
            raise ValueError(f"real {self} admits no values")
        return found

    def sample(self, k: int, rng: random.Random | None = None) -> list[float]:
        rng = rng or random.Random()
        if k < 1 or self.hi < self.lo:
            return []
        found: list[float] = []
        for _ in range(max(k * 64, k)):
            if len(found) >= k:
                break
            if self.lo > 0:
                value = 2 ** rng.uniform(math.log2(self.lo), math.log2(self.hi))
            else:
                value = rng.uniform(self.lo, self.hi)
            if value in self:
                found.append(value)
        return found


def Pow2(lo: int = 1, hi: int = 1 << 30) -> Space:
    """Powers of two inside `[lo, hi]`."""
    return Space(lo, hi, lambda v: v > 0 and v & (v - 1) == 0, f"pow2[{lo}, {hi}]", None)


def Div(n: int) -> Space:
    """Positive multiples of `n` — the pool-filter form of `Range(...) % n`."""
    if n < 1:
        raise ValueError("divisor must be positive")
    return Range(n, 1 << 30) % n


def space(value: Any) -> Space:
    """Lift a Space, non-empty integer set, or int into a Space."""
    match value:
        case Space():
            return value
        case set() | frozenset() if value:
            items = frozenset(int(item) for item in value)
            text = "{" + ", ".join(map(str, sorted(items))) + "}"
            return Space(min(items), max(items), lambda v, s=items: v in s, text, items)
        case int() if not isinstance(value, bool):
            return Space(value, value, lambda v, x=value: v == x, str(value), frozenset({value}))
        case _:
            raise TypeError(f"cannot lift {value!r} to Space")


def contains(spec: Any, value: Any) -> bool:
    """Membership for Spaces, sets, callables, and scalars (dtype/arg filters)."""
    match spec:
        case None:
            return value is None
        case Space():
            return value in spec
        case set() | frozenset():
            return not isinstance(value, torch.Tensor) and value in spec
        case _ if callable(spec):
            return bool(spec(value))
        case _:
            return not isinstance(value, torch.Tensor) and value == spec


def fmt_value(value: Any) -> str:
    if isinstance(value, torch.Tensor):
        return f"Tensor{tuple(value.shape)}"
    return repr(value)
