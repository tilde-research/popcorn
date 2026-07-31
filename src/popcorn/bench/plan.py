"""Adaptive probe planner: given rows and declared pools, emit the next cases.

Pure and deterministic. The driver runs the returned cases and refits until the
plan is empty. Effort tiers set confirm-draw counts and covering density.
"""

from __future__ import annotations

import itertools
import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from popcorn.bench.fit import fit, stratum, stratum_key
from popcorn.bench.grid import DTYPES, GRID, _dim_pool
from popcorn.bench.model import Case, Record

EFFORT = {
    "quick": {"confirm": 0, "pairwise": 8, "cap": 48},
    "standard": {"confirm": 100, "pairwise": 16, "cap": 160},
    "thorough": {"confirm": 1000, "pairwise": 32, "cap": 640},
}


def oom_dominates(oom: Case, candidate: Case) -> bool:
    """Whether `candidate` is no smaller than an OOM in every shape coordinate.

    Dtype, scalar arguments, optional tensors, and batch rank must match. A
    mixed tradeoff remains runnable: lowering any dim is enough to escape the
    dominated region even when another dim grows.
    """
    if (
        oom.dtype != candidate.dtype
        or dict(oom.args) != dict(candidate.args)
        or oom.present != candidate.present
        or len(oom.batch) != len(candidate.batch)
    ):
        return False
    oom_dims, candidate_dims = dict(oom.dims), dict(candidate.dims)
    if oom_dims.keys() != candidate_dims.keys():
        return False
    return all(candidate_dims[name] >= value for name, value in oom_dims.items()) and all(
        candidate_value >= oom_value for oom_value, candidate_value in zip(oom.batch, candidate.batch)
    )


class OOMFrontier:
    """Minimal observed OOM points under coordinate-wise shape dominance."""

    def __init__(self, cases: Sequence[Case] = ()) -> None:
        self._cases: list[Case] = []
        for case in cases:
            self.add(case)

    def blocker(self, candidate: Case) -> Case | None:
        return next((case for case in self._cases if oom_dominates(case, candidate)), None)

    def add(self, case: Case) -> None:
        if self.blocker(case) is not None:
            return
        self._cases = [current for current in self._cases if not oom_dominates(case, current)]
        self._cases.append(case)


def _anchor(op: Any) -> dict[str, int]:
    return {name: _dim_pool(op, name)[0] for name in sorted(op._dims)}


def _case(
    dims: Mapping[str, int],
    dtype: torch.dtype,
    batch: tuple[int, ...],
    args: Mapping[str, Any],
    present: frozenset[str],
) -> Case:
    return Case(tuple(sorted(dims.items())), batch, dtype, tuple(sorted(args.items())), present)


def _stratum_key(record: Record) -> tuple[Any, ...]:
    return stratum(record)


def _labels(records: Sequence[Record], backend: str) -> dict[tuple[Any, ...], dict[str, dict[int, str]]]:
    buckets: dict[tuple[Any, ...], dict[str, dict[int, str]]] = defaultdict(lambda: defaultdict(dict))
    for record in records:
        if record.impl != backend or record.result.status not in ("pass", "fail", "crash", "oom"):
            continue
        label = "fail" if record.result.status in ("fail", "crash") else record.result.status
        key = _stratum_key(record)
        for name, value in record.config["dims"].items():
            buckets[key][name][value] = label
    return buckets


def _mid(lo: int, hi: int) -> int | None:
    if hi <= lo + 1:
        return None
    mid = round(2 ** ((math.log2(max(1, lo)) + math.log2(max(1, hi))) / 2))
    return mid if lo < mid < hi else (lo + hi) // 2 if hi > lo + 1 else None


def _pairwise(axes: Mapping[str, Sequence[int]], limit: int) -> list[dict[str, int]]:
    names = sorted(axes)
    if not names:
        return []
    if len(names) == 1:
        return [{names[0]: value} for value in axes[names[0]][:limit]]
    pairs = list(itertools.combinations(names, 2))
    uncovered = {(left, right, a, b) for left, right in pairs for a in axes[left] for b in axes[right]}
    combos: list[dict[str, int]] = []
    rng = random.Random(0)
    while uncovered and len(combos) < limit:
        best, score = None, -1
        for _ in range(48):
            candidate = {name: rng.choice(list(axes[name])) for name in names}
            hit = sum((left, right, candidate[left], candidate[right]) in uncovered for left, right in pairs)
            if hit > score:
                best, score = candidate, hit
        assert best is not None
        combos.append(best)
        for left, right in pairs:
            uncovered.discard((left, right, best[left], best[right]))
    return combos


def _default_args(op: Any) -> dict[str, Any]:
    return {name: pool[0] for name, pool in op.arg_pools.items() if pool}


def plan(
    op: Any,
    records: Sequence[Record],
    backend: str,
    *,
    effort: str = "standard",
    device: str = "cuda",
    grad: bool = True,
) -> list[Case]:
    """Next cases to run for `backend` at the given effort tier."""
    if effort not in EFFORT:
        raise ValueError(f"unknown effort {effort!r}; expected one of {sorted(EFFORT)}")
    budget = EFFORT[effort]
    pools = {name: _dim_pool(op, name) for name in sorted(op._dims)}
    anchor = _anchor(op)
    labeled = _labels(records, backend)
    regions = fit((record for record in records if record.impl == backend), op._dims)
    dtype = next((dt for dt in DTYPES if any("float" in kind for spec in op.specs for kind in spec.dtypes)), torch.float32)
    batch = (2048,) if any(... in spec.tokens for spec in op.specs) else ()
    args, present = _default_args(op), frozenset()
    dtype_name = str(dtype).removeprefix("torch.")
    layer = stratum_key(device, grad, dtype_name, args, present, bool(batch))
    known = labeled.get(layer, {})
    region = regions.get((backend, *layer))

    todo: list[Case] = []
    seen: set[str] = set()

    def emit(dims: Mapping[str, int]) -> None:
        case = _case(dims, dtype, batch, args, present)
        if case.case_id not in seen:
            seen.add(case.case_id)
            todo.append(case)

    # 1. Screen: ladder points, one dim at a time.
    for name, pool in pools.items():
        for value in pool:
            if value not in known.get(name, {}):
                emit({**anchor, name: value})

    # 2. Bisect: log-midpoints between adjacent disagreeing labels.
    for name, labels in known.items():
        ordered = sorted(labels)
        for left, right in zip(ordered, ordered[1:]):
            if labels[left] == labels[right]:
                continue
            if (mid := _mid(left, right)) is not None and mid not in labels:
                emit({**anchor, name: mid})

    # 3. Congruence: residue probes at the largest observed magnitude.
    for name, labels in known.items():
        if not any(status == "pass" for status in labels.values()):
            continue
        if not any(status == "fail" for status in labels.values()):
            continue
        magnitude = max(labels)
        for modulus in (2, 4, 8, 16):
            base = magnitude - (magnitude % modulus)
            for residue in range(modulus):
                value = max(1, base + residue)
                if value not in labels and (value in pools[name] or value in GRID):
                    emit({**anchor, name: value})

    # 4–5. Pairwise covering + confirm draws inside the fitted region.
    if region is not None:
        axes = {name: spec.grid(n=6, seeds=GRID)[:6] for name, spec in region.spaces.items()}
        for combo in _pairwise(axes, budget["pairwise"]):
            emit({**anchor, **combo})
        rng = random.Random(hash((op.name, backend, layer, "confirm")) & 0xFFFFFFFF)
        for _ in range(budget["confirm"]):
            dims = dict(anchor)
            for name, spec in region.spaces.items():
                if drawn := spec.sample(1, rng):
                    dims[name] = drawn[0]
            emit(dims)

    # Corners of the declared pools.
    emit(anchor)
    emit({name: pool[-1] for name, pool in pools.items()})
    return todo[: budget["cap"]]
