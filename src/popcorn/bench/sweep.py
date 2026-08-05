"""Typed configuration and deterministic planning helpers for benchmark sweeps."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from popcorn.bench.grid import backward_safe, case_plan, estimated_bytes
from popcorn.bench.store import static_allowed
from popcorn.core.sources import available, unavailable_reason

DEFAULT_NODES = 6
MAX_NODES = 6
GPUS_PER_NODE = 8


@dataclass(frozen=True, slots=True)
class PhaseSpec:
    """One isolated sweep environment."""

    name: str
    extra: str | None = None
    packages: tuple[str, ...] = ()
    build_requirements: bool = False

    @property
    def target(self) -> str:
        extra = self.extra if self.extra is not None else ("" if self.name == "reference" else self.name)
        return f".[{extra}]" if extra else "."


PHASE_SPECS = (
    PhaseSpec("reference"),
    # wall_attn and nsa reuse FLA utility ops even though their kernels are first-party.
    PhaseSpec("popcorn", extra="fla"),
    PhaseSpec("cudnn"),
    PhaseSpec("fa3", build_requirements=True),
    PhaseSpec("fla"),
    # grpo_loss imports transformers at call time without declaring it upstream.
    PhaseSpec("liger", packages=("transformers>=4.52.0,<5",)),
    PhaseSpec("quack"),
    PhaseSpec("transformer_engine"),
    PhaseSpec("unsloth"),
)
PHASES = tuple(spec.name for spec in PHASE_SPECS)
_PHASE_BY_NAME = {spec.name: spec for spec in PHASE_SPECS}


def phase_spec(name: str) -> PhaseSpec:
    try:
        return _PHASE_BY_NAME[name]
    except KeyError as error:
        raise ValueError(f"unknown sweep phase {name!r}") from error


@dataclass(frozen=True, slots=True)
class SweepConfig:
    """Serializable user-controlled sweep settings."""

    phases: tuple[str, ...] = PHASES
    ops: tuple[str, ...] | None = None
    reps: int = 10
    timeout: int = 300
    qos: str = "staff-prod"
    walltime: str = "24:00:00"
    hardware: str | None = "H100"
    local: bool = False
    nodes: int = DEFAULT_NODES
    curves_only: bool = False

    def __post_init__(self) -> None:
        if not self.phases or any(name not in _PHASE_BY_NAME for name in self.phases):
            raise ValueError("phases must be a non-empty ordered subset of the registered sweep phases")
        if self.reps < 1 or self.timeout < 1:
            raise ValueError("reps and timeout must be positive")
        if not 1 <= self.nodes <= MAX_NODES:
            raise ValueError(f"nodes must be between 1 and {MAX_NODES}")
        minutes(self.walltime)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None = None, **overrides: Any) -> SweepConfig:
        data = dict(value or {})
        data.update(overrides)
        phases = data.get("phases", PHASES)
        ops = data.get("ops")
        data["phases"] = tuple(phases)
        data["ops"] = tuple(ops) if ops else None
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["phases"] = list(self.phases)
        data["ops"] = list(self.ops) if self.ops else None
        return data

    @property
    def workers(self) -> int:
        return 1 if self.local else self.nodes * GPUS_PER_NODE


def new_state(run: Path, phases: Iterable[str]) -> dict[str, Any]:
    """Create the durable progress state for a run."""
    return {
        "run": str(run),
        "job_id": None,
        "status": "created",
        "attempts": 0,
        "workers": 0,
        "started": None,
        "finished": None,
        "phases": [
            {
                "name": name,
                "status": "pending",
                "total": 0,
                "cached": 0,
                "prior_pruned": 0,
                "infeasible": 0,
                "measured": 0,
                "pruned": 0,
                "bench_errors": 0,
                "statuses": {},
                "message": "",
            }
            for name in phases
        ],
    }


def _settled_count(phase: Mapping[str, Any]) -> int:
    return sum(int(phase.get(field, 0)) for field in ("cached", "prior_pruned", "infeasible", "measured", "pruned"))


def _bar(done: int, total: int, width: int = 24) -> str:
    if total <= 0:
        return "[" + " " * width + "]"
    filled = min(width, round(width * done / total))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def render(state: Mapping[str, Any]) -> str:
    """Render a stable overall and per-phase progress snapshot."""
    total = sum(int(phase.get("total", 0)) for phase in state.get("phases", []))
    done = sum(_settled_count(phase) for phase in state.get("phases", []))
    slurm = f", Slurm {state['slurm_state']}" if state.get("slurm_state") else ""
    lines = [
        f"Popcorn sweep job {state.get('job_id') or '-'}: {state.get('status', 'unknown')}{slurm} "
        f"({state.get('workers') or '?'} GPU workers, attempt {state.get('attempts', 0)})",
        f"Overall {_bar(done, total, 32)} {done:,}/{total:,}",
    ]
    phases = state.get("phases", [])
    for index, phase in enumerate(phases, start=1):
        total = int(phase.get("total", 0))
        done = _settled_count(phase)
        status = phase.get("status", "pending")
        marker = {
            "completed": "✓",
            "failed": "✗",
            "running": "▶",
            "installing": "↓",
            "planning": "…",
            "skipped": "-",
            "paused": "‖",
        }.get(status, "·")
        progress = f"{done:,}/{total:,}" if total else status
        line = f"{marker} {index}/{len(phases)} {phase['name']:<18} {_bar(done, total)} {progress}"
        details = []
        if phase.get("cached"):
            details.append(f"cached {phase['cached']:,}")
        if phase.get("measured"):
            details.append(f"ran {phase['measured']:,}")
        pruned = int(phase.get("prior_pruned", 0)) + int(phase.get("pruned", 0))
        if pruned:
            details.append(f"pruned {pruned:,}")
        if phase.get("infeasible"):
            details.append(f"infeasible {phase['infeasible']:,}")
        if phase.get("bench_errors"):
            details.append(f"bench_error={phase['bench_errors']:,}")
        if statuses := phase.get("statuses", {}):
            details.append(" ".join(f"{name}={count:,}" for name, count in sorted(statuses.items())))
        if details:
            line += " | " + ", ".join(details)
        if phase.get("message"):
            line += f" | {phase['message']}"
        lines.append(line)
    return "\n".join(lines)


def minutes(walltime: str) -> int:
    """Convert Slurm `HH:MM:SS` or plain minutes to submitit's integer minutes."""
    parts = walltime.split(":")
    if len(parts) == 1:
        value = int(parts[0])
        if value < 1:
            raise argparse.ArgumentTypeError("walltime must be positive")
        return value
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected HH:MM:SS or minutes")
    hours, minute, seconds = (int(part) for part in parts)
    if hours < 0 or not 0 <= minute < 60 or not 0 <= seconds < 60:
        raise argparse.ArgumentTypeError("expected HH:MM:SS or minutes")
    return hours * 60 + minute + math.ceil(seconds / 60)


def distribute(units: list[tuple[int, Any]], workers: int) -> tuple[list[list[Any]], list[int]]:
    """Assign longest units first to the currently lightest worker."""
    if workers < 1:
        raise ValueError("workers must be positive")
    bins: list[list[Any]] = [[] for _ in range(workers)]
    loads = [0] * workers
    for _, (weight, payload) in sorted(enumerate(units), key=lambda item: (-item[1][0], item[0])):
        target = min(range(workers), key=lambda index: (loads[index], index))
        bins[target].append(payload)
        loads[target] += weight
    return bins, loads


def affinity_units(groups: list[dict[str, Any]], workers: int) -> list[tuple[int, list[dict[str, Any]]]]:
    """Split only heavy op/implementation bundles, preserving compile affinity."""
    total = sum(len(group["cases"]) for group in groups)
    target = max(1, math.ceil(total / workers))
    affinities: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for group in groups:
        affinities.setdefault((group["op"], group["impl"]), []).append(group)

    units = []
    for affinity in sorted(affinities):
        members = affinities[affinity]
        weight = sum(len(group["cases"]) for group in members)
        chunks = min(len(members), max(1, math.ceil(weight / target)))
        bins, loads = distribute([(len(group["cases"]), group) for group in members], chunks)
        units.extend((load, chunk) for chunk, load in zip(bins, loads) if chunk)
    return units


def reference_modes(op: Any, case: Any | None = None) -> tuple[bool, ...]:
    """Gradient modes needed to compare every registered implementation."""
    implementations = [impl for impl in op._impls if impl.name != "torch"]
    if case is not None and implementations:
        implementations = [
            impl for impl in implementations if static_allowed(op, impl, case.dtype, dict(case.args), case.present)
        ]
        if not implementations:
            return ()
    modes = {not impl.forward_only for impl in implementations} or {True}
    if case is not None and True in modes and not backward_safe(op, case):
        modes = {False}
    return tuple(sorted(modes, reverse=True))


def estimated(op: Any, case: Any) -> int:
    """Estimate input bytes; unpriceable cases remain schedulable."""
    try:
        return estimated_bytes(op, case)
    except Exception:
        return 0


def infeasible(op: Any, case: Any, grad: bool, capacity: int) -> bool:
    """Conservatively reject cases whose inputs alone exceed device capacity."""
    return estimated(op, case) * (3 if grad else 2) > capacity


def expected_totals(
    ops: Iterable[Any],
    phases: Iterable[str],
    only: set[str] | None = None,
    curves_only: bool = False,
) -> dict[str, int]:
    """Registered work per phase before cache and frontier pruning."""
    names = tuple(phases)
    totals = {name: 0 for name in names}
    families = set()
    for op in ops:
        if only and op.name not in only:
            continue
        plan = case_plan(op)
        cases = _curve_cases(plan) if curves_only else plan.flatten()
        if "reference" in totals:
            totals["reference"] += sum(len(reference_modes(op, case)) for case in cases)
        for impl in op._impls:
            if impl.name == "torch":
                continue
            family = impl.name.split(":", 1)[0]
            families.add(family)
            if family in totals:
                totals[family] += sum(static_allowed(op, impl, case.dtype, dict(case.args), case.present) for case in cases)
    missing = families - set(names)
    if only is None and set(names) == set(PHASES) and missing:
        raise RuntimeError(f"sweep phases omit registered backend families: {', '.join(sorted(missing))}")
    return totals


def phase_work(
    ops: Iterable[Any],
    phase_name: str,
    only: set[str] | None = None,
    curves_only: bool = False,
) -> list[tuple[Any, str, Any, bool]]:
    """Build work for one isolated environment from the shared case plan."""
    ops = tuple(ops)
    installed = {
        impl.name.split(":", 1)[0] for op in ops for impl in op._impls if impl.name != "torch" and available(impl.name)
    }
    allowed = {"popcorn"}
    if phase_name != "reference":
        allowed.add(phase_name)
    if (extra := phase_spec(phase_name).extra) in _PHASE_BY_NAME:
        allowed.add(extra)
    unexpected = installed - allowed
    if unexpected:
        raise RuntimeError(
            f"backend environment is not isolated for {phase_name!r}; also found {', '.join(sorted(unexpected))}"
        )

    work = []
    registered = 0
    unavailable = set()
    for op in sorted(ops, key=lambda item: item.name):
        if only and op.name not in only:
            continue
        plan = case_plan(op)
        cases = _curve_cases(plan) if curves_only else plan.flatten()
        if phase_name == "reference":
            for case in cases:
                work.extend((op, "torch", case, grad) for grad in reference_modes(op, case))
            continue
        implementations = [impl for impl in op._impls if impl.name != "torch" and impl.name.split(":", 1)[0] == phase_name]
        registered += len(implementations)
        for impl in implementations:
            if not available(impl.name):
                unavailable.add(unavailable_reason(impl.name) or impl.name)
                continue
            work.extend(
                (op, impl.name, case, not impl.forward_only and backward_safe(op, case))
                for case in cases
                if static_allowed(op, impl, case.dtype, dict(case.args), case.present)
            )
    if phase_name != "reference" and not registered and only is None:
        raise RuntimeError(f"no implementations are registered for backend family {phase_name!r}")
    if phase_name != "reference" and registered and not work:
        raise RuntimeError(
            f"backend family {phase_name!r} is unavailable after installation: {'; '.join(sorted(unavailable))}"
        )
    return work


def _curve_cases(plan: Any) -> list[Any]:
    found = {}
    for series in plan.series:
        if series.kind == "curve":
            for case in series.cases:
                found.setdefault(case.case_id, case)
    return list(found.values())


def stratum(case: Any) -> tuple[Any, ...]:
    """Compile-affinity key for one case."""
    return (
        str(case.dtype),
        json.dumps(dict(case.args), sort_keys=True, separators=(",", ":")),
        tuple(sorted(case.present)),
        len(case.batch),
    )
