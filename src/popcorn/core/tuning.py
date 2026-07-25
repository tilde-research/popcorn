"""Backend selection from recorded validation and benchmark data."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from threading import RLock
from typing import TYPE_CHECKING, Any

import torch

from popcorn.bench.fit import Region, fit, region_for
from popcorn.bench.model import Record
from popcorn.bench.store import Store, matching
from popcorn.core.config import Call, device_name
from popcorn.core.errors import DispatchError
from popcorn.core.policy import Policy
from popcorn.core.sources import installed_version
from popcorn.core.spaces import contains

if TYPE_CHECKING:
    from popcorn.core.dispatcher import Backend, Dispatcher


def _key(value: Any) -> Any:
    match value:
        case dict():
            return tuple(sorted((name, _key(item)) for name, item in value.items()))
        case list() | tuple():
            return tuple(_key(item) for item in value)
        case set() | frozenset():
            return tuple(sorted(map(_key, value), key=repr))
        case _:
            try:
                hash(value)
                return value
            except TypeError:
                return repr(value)


@dataclass(frozen=True, slots=True)
class Sample:
    """One usable benchmark row: its case coordinates and measured timings."""

    backend: str
    device: str
    grad: bool
    dims: dict[str, int]
    batch: tuple[int, ...]
    dtype: str
    args: dict[str, Any]
    present: frozenset[str]
    fwd_ms: float
    ref_fwd_ms: float
    bwd_ms: float | None = None
    ref_bwd_ms: float | None = None

    @property
    def time(self) -> tuple[float, float]:
        return self.fwd_ms + (self.bwd_ms or 0), self.ref_fwd_ms + (self.ref_bwd_ms or 0)


class Tuner:
    """Selection only: reads recorded results and picks backends, never runs or writes benchmarks.

    Ranking / admission live on `policy` — replace it to reconfigure behavior.
    """

    def __init__(self, op: Dispatcher, store: Store | None = None, policy: Policy | None = None) -> None:
        self.op = op
        self.store = store or Store()
        self.policy = policy or Policy()
        self._selected: dict[Any, Backend] = {}
        self._lock = RLock()

    @cached_property
    def rows(self) -> tuple[Record, ...]:
        return self.store.merged(self.op.name)

    @cached_property
    def samples(self) -> tuple[Sample, ...]:
        samples = []
        for record in self.rows:
            result, environment, config = record.result, record.environment, record.config
            bench = result.bench
            usable = (
                result.status == "pass"
                and result.benchmarked
                and environment.torch == torch.__version__
                and environment.backend_version == installed_version(record.backend)
                and matching(self.op.fingerprint, environment.ref_hash)
                and matching(self._expected(record.backend), environment.impl_hash)
                and config
                and bench.get("fwd_ms")
                and bench.get("ref_fwd_ms")
            )
            if usable:
                samples.append(
                    Sample(
                        record.backend,
                        environment.device,
                        result.grad,
                        config["dims"],
                        tuple(config["batch"]),
                        config["dtype"],
                        config["args"],
                        frozenset(config["present"]),
                        bench["fwd_ms"],
                        bench["ref_fwd_ms"],
                        bench.get("bwd_ms"),
                        bench.get("ref_bwd_ms"),
                    )
                )
        return tuple(samples)

    @cached_property
    def regions(self) -> dict[tuple[Any, ...], Region]:
        """Validity regions fitted from current fingerprint-matching rows."""
        current = [
            record
            for record in self.rows
            if record.environment.torch == torch.__version__
            and record.environment.backend_version == installed_version(record.backend)
            and matching(self.op.fingerprint, record.environment.ref_hash)
            and matching(self._expected(record.backend), record.environment.impl_hash)
        ]
        return fit(current, self.op._dims)

    @property
    def op_name(self) -> str:
        return self.op.name

    def _expected(self, name: str) -> str | None:
        backend = next((candidate for candidate in self.op._backends if candidate.name == name), None)
        return backend.fingerprint if backend else None

    def exact(self, backend: Backend, call: Call) -> Record | None:
        return self.store.exact(
            self.op.name,
            backend.name,
            call,
            torch.__version__,
            installed_version(backend.name),
            self.rows,
            ref_hash=self.op.fingerprint,
            impl_hash=backend.fingerprint,
        )

    def region_for(self, backend: Backend, call: Call) -> Region | None:
        return region_for(self.regions, backend.name, call.device_name, call.grad, call.config)

    def mapped(self, backend: str) -> bool:
        return any(key[0] == backend for key in self.regions)

    def select(
        self,
        call: Call | None,
        candidates: Sequence[Backend],
        forced: str | None = None,
        signature: Any = None,
        policy: Policy | None = None,
    ) -> Backend:
        active = policy or self.policy
        with self._lock:
            key = (_key(signature), tuple(backend.name for backend in candidates), forced, active)
            if key not in self._selected:
                self._selected[key] = self._select(call, candidates, forced, active)
            return self._selected[key]

    def _select(
        self,
        call: Call | None,
        candidates: Sequence[Backend],
        forced: str | None,
        policy: Policy,
    ) -> Backend:
        eligible = [backend for backend in candidates if not policy.blocked(backend, call, forced, self)]
        if not eligible:
            raise DispatchError(f"{self.op.name}: no backend is eligible")
        return policy.choose(eligible, call, forced, self) or eligible[0]

    def best(self, *, device: torch.device | str | None = None, grad: bool = True, **region: Any) -> Backend:
        allowed = self.op._dims | self.op.arg_pools.keys() | {"dtype"}
        if unknown := set(region) - allowed:
            raise TypeError(f"{self.op.name}: unknown filter keys {sorted(unknown)}; allowed: {sorted(allowed)}")
        samples = [
            sample
            for sample in self.samples
            if sample.device == device_name(device)
            and sample.grad == grad
            and all(
                contains(spec, sample.dtype if name == "dtype" else (sample.dims | sample.args)[name])
                for name, spec in region.items()
            )
        ]
        if not samples:
            raise LookupError(f"{self.op.name}: no benchmark points match {region}")
        return self.op[self.policy.fastest(samples)]

    def forget(self) -> None:
        with self._lock:
            self.__dict__.pop("rows", None)
            self.__dict__.pop("samples", None)
            self.__dict__.pop("regions", None)
            self._selected.clear()
