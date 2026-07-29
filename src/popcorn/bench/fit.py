"""Fit per-backend validity regions from report rows.

Pure: same rows in, same regions out. A region holds integer `Space`s for dims
and `Real` bands for continuous args, keyed by a stratum that uses only
*discrete* args (floats are generalized like shapes, not exact-matched).
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from popcorn.bench.model import Record
from popcorn.core.args import partition_args
from popcorn.core.spaces import Range, Real, Space


@dataclass(frozen=True, slots=True)
class Region:
    """Fitted validity region: Spaces for dims, Reals for continuous args."""

    spaces: dict[str, Space]
    reals: dict[str, Real] = field(default_factory=dict)
    confirm_n: int = 0

    @property
    def epsilon95(self) -> float:
        return 3 / self.confirm_n if self.confirm_n else float("inf")

    def contains(self, dims: Mapping[str, int], args: Mapping[str, Any] | None = None) -> bool:
        if not self.spaces and not self.reals:
            return False
        if not all(dims[name] in spec for name, spec in self.spaces.items() if name in dims):
            return False
        if not self.reals:
            return True
        _, continuous = partition_args(args or {})
        return all(continuous[name] in spec for name, spec in self.reals.items() if name in continuous)

    def reject(self, dims: Mapping[str, int], args: Mapping[str, Any] | None = None) -> str | None:
        for name, spec in self.spaces.items():
            if name in dims and dims[name] not in spec:
                return f"{name}={dims[name]} outside {spec}"
        _, continuous = partition_args(args or {})
        for name, spec in self.reals.items():
            if name in continuous and continuous[name] not in spec:
                return f"{name}={continuous[name]} outside {spec}"
        return None


def _arg_key(value: Any) -> Any:
    match value:
        case dict():
            return tuple(sorted((name, _arg_key(item)) for name, item in value.items()))
        case list() | tuple():
            return tuple(_arg_key(item) for item in value)
        case set() | frozenset():
            return tuple(sorted(map(_arg_key, value), key=repr))
        case _:
            return value


def stratum_key(
    device: str,
    grad: bool,
    dtype: str,
    args: Mapping[str, Any],
    present: Iterable[str],
    batched: bool,
) -> tuple[Any, ...]:
    """Stratum coordinates — discrete args only; floats live in the region."""
    discrete, _ = partition_args(args)
    return (
        device,
        grad,
        dtype,
        tuple(sorted((name, _arg_key(value)) for name, value in discrete.items())),
        tuple(sorted(present)),
        batched,
    )


def stratum(record: Record) -> tuple[Any, ...]:
    """Device/grad/dtype/discrete-args/presence, plus empty-vs-batched."""
    config = record.config
    return stratum_key(
        record.environment.device,
        record.result.grad,
        config["dtype"],
        config["args"],
        config["present"],
        bool(config["batch"]),
    )


def _congruence(passes: Sequence[int], fails: Sequence[int]) -> int | None:
    """Largest small modulus that separates passes (residue 0) from fails."""
    if len(passes) < 2:
        return None
    gap = 0
    for left, right in zip(passes, passes[1:]):
        gap = math.gcd(gap, right - left)
    candidates = [m for m in (2, 4, 8, 16, 32, 64) if gap % m == 0 or m == gap]
    if gap > 1:
        candidates.append(gap)
    for modulus in sorted(set(candidates), reverse=True):
        if modulus <= 1:
            continue
        if all(value % modulus == 0 for value in passes) and all(value % modulus != 0 for value in fails):
            return modulus
    return None


def _induce(labels: Mapping[int, str]) -> Space | None:
    """Induce `Range(lo, hi) [% m] [- drops]` from pass/fail/oom labels on one dim."""
    passes = sorted(value for value, status in labels.items() if status == "pass")
    fails = sorted(value for value, status in labels.items() if status == "fail")
    ooms = sorted(value for value, status in labels.items() if status == "oom")
    if not passes:
        return None
    lo, hi = passes[0], passes[-1]
    if ooms:
        if min(ooms) > max(passes):
            hi = min(hi, min(ooms) - 1)
        else:
            # Non-monotone footprint: trust only the outermost observed pass.
            hi = max(passes)
    if hi < lo:
        return None
    interior_fails = [value for value in fails if lo <= value <= hi]
    fitted = Range(lo, hi)
    if modulus := _congruence(passes, interior_fails):
        fitted = fitted % modulus
    drops = [value for value in interior_fails if value in fitted]
    if drops:
        fitted = fitted - set(drops)
    if any(value not in fitted for value in passes if lo <= value <= hi):
        # Congruence over-shrunk; fall back to range minus hard fails.
        fitted = Range(lo, hi) - set(interior_fails) if interior_fails else Range(lo, hi)
    return fitted


def _induce_real(labels: Mapping[float, str]) -> Real | None:
    """Induce a closed real band from pass/fail/oom labels on one continuous arg."""
    passes = sorted(value for value, status in labels.items() if status == "pass")
    fails = sorted(value for value, status in labels.items() if status == "fail")
    ooms = sorted(value for value, status in labels.items() if status == "oom")
    if not passes:
        return None
    lo, hi = passes[0], passes[-1]
    if ooms:
        if min(ooms) > max(passes):
            below = [value for value in passes if value < min(ooms)]
            if not below:
                return None
            hi = max(below)
        else:
            hi = max(passes)
    if hi < lo:
        return None
    drops = frozenset(value for value in fails if lo <= value <= hi)
    fitted = Real(lo, hi, drops)
    if any(value not in fitted for value in passes):
        return Real(lo, hi, frozenset(fails)) if fails else Real(lo, hi)
    return fitted


def _label_rank(label: str) -> int:
    return {"fail": 2, "pass": 1, "oom": 0}[label]


def fit(records: Iterable[Record], dims: Iterable[str]) -> dict[tuple[Any, ...], Region]:
    """Group conclusive rows by (implementation, stratum) and fit Spaces + Reals."""
    dim_names = list(dims)
    dim_buckets: dict[tuple[Any, ...], dict[str, dict[int, str]]] = defaultdict(lambda: defaultdict(dict))
    real_buckets: dict[tuple[Any, ...], dict[str, dict[float, str]]] = defaultdict(lambda: defaultdict(dict))
    counts: dict[tuple[Any, ...], int] = defaultdict(int)
    for record in records:
        status = record.result.status
        if status not in ("pass", "fail", "crash", "oom"):
            continue
        label = "fail" if status in ("fail", "crash") else status
        key = (record.impl, *stratum(record))
        counts[key] += status == "pass"
        for name in dim_names:
            if name in record.config["dims"]:
                value = record.config["dims"][name]
                prior = dim_buckets[key][name].get(value)
                if prior is None or _label_rank(label) >= _label_rank(prior):
                    dim_buckets[key][name][value] = label
        _, continuous = partition_args(record.config["args"])
        for name, value in continuous.items():
            prior = real_buckets[key][name].get(value)
            if prior is None or _label_rank(label) >= _label_rank(prior):
                real_buckets[key][name][value] = label
    regions: dict[tuple[Any, ...], Region] = {}
    keys = set(dim_buckets) | set(real_buckets)
    for key in keys:
        spaces = {
            name: induced for name, labels in dim_buckets.get(key, {}).items() if (induced := _induce(labels)) is not None
        }
        reals = {
            name: induced
            for name, labels in real_buckets.get(key, {}).items()
            if (induced := _induce_real(labels)) is not None
        }
        if spaces or reals:
            regions[key] = Region(spaces, reals, counts[key])
    return regions


def region_for(
    regions: Mapping[tuple[Any, ...], Region],
    backend: str,
    device: str,
    grad: bool,
    config: Mapping[str, Any],
) -> Region | None:
    """Exact stratum lookup for a call config (discrete args only)."""
    key = (
        backend,
        *stratum_key(
            device,
            grad,
            config["dtype"],
            config["args"],
            config["present"],
            bool(config["batch"]),
        ),
    )
    return regions.get(key)
