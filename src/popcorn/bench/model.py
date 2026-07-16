"""Data model for report rows: cases, gauges, results, and records."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

import torch

from popcorn.core.config import config_id, make_config

SCHEMA = 2


@dataclass(frozen=True, slots=True)
class Case:
    """One grid point: dim sizes, batch shape, dtype, scalar args, and present optionals."""

    dims: tuple[tuple[str, int], ...]
    batch: tuple[int, ...]
    dtype: torch.dtype
    args: tuple[tuple[str, Any], ...]
    present: frozenset[str]

    def __str__(self) -> str:
        parts = [f"{name}={value}" for name, value in self.dims]
        parts += [f"batch={self.batch}", str(self.dtype).removeprefix("torch.")]
        parts += [f"{name}={value}" for name, value in self.args]
        parts += [f"+{name}" for name in sorted(self.present)]
        return ",".join(parts)

    @property
    def case_id(self) -> str:
        return config_id(self.config())

    def config(self) -> dict[str, Any]:
        return make_config(dict(self.dims), self.batch, self.dtype, dict(self.args), self.present)


@dataclass(slots=True)
class Gauge:
    """Worst observed error for one output or gradient, with its reference budget and scale."""

    err: float = 0.0
    budget: float = 0.0
    scale: float = 1.0

    def note(self, err: float, budget: float, truth: torch.Tensor) -> None:
        self.err = max(self.err, err)
        self.budget = max(self.budget, budget)
        self.scale = max(self.scale, truth.detach().abs().max().item() if truth.numel() else 1.0)

    def verdict(self, floor: float) -> str | None:
        if not math.isfinite(self.err):
            return "err is non-finite"
        if self.err <= max(2 * self.budget, floor * self.scale):
            return None
        return f"err {self.err:.3e} > max(2*{self.budget:.3e}, {floor:.0e}*{self.scale:.1f})"


@dataclass(slots=True)
class Result:
    """Outcome of one comparison: status, per-gauge errors, and optional timings."""

    status: str = "pass"
    reason: str = ""
    grad: bool = False
    fwd: dict[str, Gauge] = field(default_factory=dict)
    bwd: dict[str, Gauge] = field(default_factory=dict)
    bench: dict[str, float] = field(default_factory=dict)
    benchmarked: bool = False
    bench_error: str = ""
    reps: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Environment:
    """Where a record was produced: device, versions, timestamp, and code fingerprints."""

    device: str
    torch: str
    backend_version: str | None
    ts: str
    ref_hash: str | None = None
    impl_hash: str | None = None


@dataclass(slots=True)
class Record:
    """One stored report row: an op/backend/case triple with its environment and result."""

    op: str
    backend: str
    case: str
    case_id: str
    config: dict
    environment: Environment
    result: Result

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "op": self.op,
            "backend": self.backend,
            "case": self.case,
            "case_id": self.case_id,
            "config": self.config,
            "device": self.environment.device,
            "torch": self.environment.torch,
            "backend_version": self.environment.backend_version,
            "ts": self.environment.ts,
            "ref_hash": self.environment.ref_hash,
            "impl_hash": self.environment.impl_hash,
            **self.result.to_dict(),
        }

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> Record:
        if row.get("schema") != SCHEMA:
            raise ValueError(f"unsupported report schema {row.get('schema')!r}; expected {SCHEMA}")
        result = Result(
            status=row["status"],
            reason=row.get("reason", ""),
            grad=row["grad"],
            fwd={name: Gauge(**gauge) for name, gauge in row.get("fwd", {}).items()},
            bwd={name: Gauge(**gauge) for name, gauge in row.get("bwd", {}).items()},
            bench=row.get("bench", {}),
            benchmarked=row.get("benchmarked", bool(row.get("bench"))),
            bench_error=row.get("bench_error", ""),
            reps=row.get("reps", 0),
        )
        environment = Environment(
            row["device"], row["torch"], row.get("backend_version"), row["ts"], row.get("ref_hash"), row.get("impl_hash")
        )
        return cls(row["op"], row["backend"], row.get("case", ""), row["case_id"], row["config"], environment, result)

    @property
    def key(self) -> tuple[Any, ...]:
        return (
            self.op,
            self.backend,
            self.environment.device,
            self.case_id,
            self.result.grad,
            self.environment.torch,
            self.environment.backend_version,
        )
