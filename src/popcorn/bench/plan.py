"""Adaptive probe planner: given rows and declared pools, emit the next cases.

Pure and deterministic. The driver runs the returned cases and refits until the
plan is empty. Effort tiers set confirm-draw counts and covering density.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from popcorn.bench.fit import fit, stratum, stratum_key
from popcorn.bench.grid import (
    DTYPES,
    ELLIPSIS,
    GRID,
    CasePlan,
    CaseSeries,
    _build_case,
    _dim_pool,
    _pairwise_rows,
    _profile_anchors,
)
from popcorn.bench.model import Case, Record

EFFORT = {
    "quick": {"confirm": 0, "pairwise": 8, "cap": 48},
    "standard": {"confirm": 100, "pairwise": 16, "cap": 160},
    "thorough": {"confirm": 1000, "pairwise": 32, "cap": 640},
}


# Memory and runtime both grow monotonically with the shape, so a case that exhausted
# either budget bounds every case above it: one that is no smaller in each coordinate
# cannot fit, and cannot finish, either. A timeout costs the full per-case budget every
# time it is rediscovered, which is why it prunes alongside an OOM. Crashes deliberately
# do not: those are usually tile-boundary or alignment bugs a larger shape may not repeat.
EXHAUSTED = ("oom", "timeout")


def dominates(exhausted: Case, candidate: Case) -> bool:
    """Whether `candidate` is no smaller than an exhausted case in every shape coordinate.

    Dtype, scalar arguments, optional tensors, and batch rank must match. A
    mixed tradeoff remains runnable: lowering any dim is enough to escape the
    dominated region even when another dim grows.
    """
    if (
        exhausted.dtype != candidate.dtype
        or dict(exhausted.args) != dict(candidate.args)
        or exhausted.present != candidate.present
        or len(exhausted.batch) != len(candidate.batch)
    ):
        return False
    exhausted_dims, candidate_dims = dict(exhausted.dims), dict(candidate.dims)
    if exhausted_dims.keys() != candidate_dims.keys():
        return False
    return all(candidate_dims[name] >= value for name, value in exhausted_dims.items()) and all(
        candidate_value >= exhausted_value for exhausted_value, candidate_value in zip(exhausted.batch, candidate.batch)
    )


class BudgetFrontier:
    """Minimal observed out-of-memory and timeout points under coordinate-wise dominance."""

    def __init__(self, cases: Sequence[Case] = ()) -> None:
        self._cases: list[Case] = []
        for case in cases:
            self.add(case)

    def blocker(self, candidate: Case) -> Case | None:
        return next((case for case in self._cases if dominates(case, candidate)), None)

    def add(self, case: Case) -> None:
        if self.blocker(case) is not None:
            return
        self._cases = [current for current in self._cases if not dominates(case, current)]
        self._cases.append(case)


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


def _default_args(op: Any) -> dict[str, Any]:
    return {name: pool[0] for name, pool in op.arg_pools.items() if pool}


def adaptive_plan(
    op: Any,
    records: Sequence[Record],
    backend: str,
    *,
    effort: str = "standard",
    device: str = "cuda",
    grad: bool = True,
) -> CasePlan:
    """Named follow-up series for `backend` at the given effort tier."""
    if effort not in EFFORT:
        raise ValueError(f"unknown effort {effort!r}; expected one of {sorted(EFFORT)}")
    budget = EFFORT[effort]
    pools = {name: _dim_pool(op, name) for name in sorted(op._dims)}
    labeled = _labels(records, backend)
    regions = fit((record for record in records if record.impl == backend), op._dims)
    dtype = next((dt for dt in DTYPES if any("float" in kind for spec in op.specs for kind in spec.dtypes)), torch.float32)
    args, present = _default_args(op), frozenset()
    profile_pools = dict(pools)
    if any(... in spec.tokens for spec in op.specs):
        profile_pools[ELLIPSIS] = _dim_pool(op, ELLIPSIS)
    profile_values = _profile_anchors(profile_pools, "production")
    leading = {ELLIPSIS: profile_values[ELLIPSIS]} if ELLIPSIS in profile_values else {}
    context = _build_case(profile_values, dtype, args, present)
    anchor, batch, args = dict(context.dims), context.batch, dict(context.args)
    dtype_name = str(dtype).removeprefix("torch.")
    layer = stratum_key(device, grad, dtype_name, args, present, bool(batch))
    known = labeled.get(layer, {})
    region = regions.get((backend, *layer))

    staged: dict[tuple[str, str | None], list[Case]] = {}
    seen: set[str] = set()

    def emit(stage: str, dims: Mapping[str, int], axis: str | None = None) -> None:
        if len(seen) >= budget["cap"]:
            return
        case = _build_case({**dims, **leading}, dtype, args, present)
        if case.case_id not in seen:
            seen.add(case.case_id)
            staged.setdefault((stage, axis), []).append(case)

    # 1. Screen: ladder points, one dim at a time.
    for name, pool in pools.items():
        for value in pool:
            if value not in known.get(name, {}):
                emit("screen", {**anchor, name: value}, name)

    # 2. Bisect: log-midpoints between adjacent disagreeing labels.
    for name, labels in known.items():
        ordered = sorted(labels)
        for left, right in zip(ordered, ordered[1:]):
            if labels[left] == labels[right]:
                continue
            if (mid := _mid(left, right)) is not None and mid not in labels:
                emit("bisect", {**anchor, name: mid}, name)

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
                    emit("congruence", {**anchor, name: value}, name)

    # 4–5. Pairwise covering + confirm draws inside the fitted region.
    if region is not None:
        axes = {name: spec.grid(n=6, seeds=GRID)[:6] for name, spec in region.spaces.items()}
        for combo in _pairwise_rows(axes, budget["pairwise"]):
            emit("pairwise", {**anchor, **combo})
        rng = random.Random(repr((op.name, backend, layer, "confirm")))
        for _ in range(budget["confirm"]):
            dims = dict(anchor)
            for name, spec in region.spaces.items():
                if drawn := spec.sample(1, rng):
                    dims[name] = drawn[0]
            emit("confirm", dims)

    # Corners of the declared pools.
    emit("corners", anchor)
    emit("corners", {name: pool[-1] for name, pool in pools.items()})
    series = []
    for (stage, axis), cases in staged.items():
        fixed = tuple((name, value) for name, value in sorted(anchor.items()) if name != axis)
        suffix = f":{axis}" if axis is not None else ""
        series.append(
            CaseSeries(
                f"adaptive:{stage}{suffix}",
                "adaptive",
                tuple(cases),
                profile="production",
                axis=axis,
                fixed_dims=fixed,
                batch=batch,
                dtype=dtype,
                args=tuple(sorted(args.items())),
                present=present,
            )
        )
    return CasePlan(tuple(series))


def plan(
    op: Any,
    records: Sequence[Record],
    backend: str,
    *,
    effort: str = "standard",
    device: str = "cuda",
    grad: bool = True,
) -> list[Case]:
    """Compatibility API returning the flattened adaptive plan."""
    return adaptive_plan(op, records, backend, effort=effort, device=device, grad=grad).flatten()
