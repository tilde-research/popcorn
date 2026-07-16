"""Backend selection from recorded validation and benchmark data."""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from statistics import median
from threading import RLock
from typing import TYPE_CHECKING, Any

import torch

from popcorn.bench.model import Record
from popcorn.bench.store import Store, matching
from popcorn.core.config import Call, device_name
from popcorn.core.constraints import satisfies
from popcorn.core.errors import DispatchError, UnvalidatedWarning
from popcorn.core.sources import installed_version

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
    """Selection only: reads recorded results and picks backends, never runs or writes benchmarks."""

    def __init__(self, op: Dispatcher, store: Store | None = None) -> None:
        self.op = op
        self.store = store or Store()
        self._selected: dict[Any, Backend] = {}
        self._warned: set[Any] = set()
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

    def select(
        self,
        call: Call | None,
        candidates: Sequence[Backend],
        forced: str | None = None,
        signature: Any = None,
        require_pass: bool = False,
    ) -> Backend:
        with self._lock:
            key = (_key(signature), tuple(backend.name for backend in candidates), forced, require_pass)
            if key not in self._selected:
                self._selected[key] = self._select(call, candidates, forced, signature, require_pass)
            return self._selected[key]

    def _select(
        self,
        call: Call | None,
        candidates: Sequence[Backend],
        forced: str | None,
        signature: Any,
        require_pass: bool,
    ) -> Backend:
        eligible = [backend for backend in candidates if not self._blocked(backend, call, forced, require_pass)]
        if not eligible:
            raise DispatchError(f"{self.op.name}: no backend is eligible")
        routed = self._nearest(eligible, call) if forced is None and call is not None else None
        selected = routed or eligible[0]
        if selected.name == "torch":
            return selected
        if call is None:
            self._warn_once(
                (selected.name, _key(signature)),
                f"{self.op.name}: backend {selected.name!r} cannot represent this call as a validation case",
            )
        elif not require_pass:
            record = self.exact(selected, call)
            if record is None or record.result.status != "pass":
                self._warn_once(
                    (selected.name, _key(call.config), call.grad, call.device_name),
                    f"{self.op.name}: backend {selected.name!r} has not been validated for this exact call; "
                    "run op.validate(...) or set POPCORN_VALIDATE=1",
                )
        return selected

    def _blocked(self, backend: Backend, call: Call | None, forced: str | None, require_pass: bool) -> bool:
        """Known exact failures always block; unvalidated backends block only in validation modes."""
        if backend.name == "torch" or call is None:
            return False
        record = self.exact(backend, call)
        status = record.result.status if record else None
        if status in ("fail", "crash"):
            if forced == backend.name:
                reason = (record.result.reason if record else "") or status
                raise DispatchError(f"{self.op.name}: backend {backend.name!r} failed validation: {reason}")
            return True
        if require_pass and status != "pass":
            if forced == backend.name:
                raise DispatchError(f"{self.op.name}: backend {backend.name!r} could not be validated")
            return True
        return False

    def _warn_once(self, key: Any, message: str) -> None:
        if key not in self._warned:
            warnings.warn(message, UnvalidatedWarning, stacklevel=5)
            self._warned.add(key)

    def _nearest(self, candidates: Sequence[Backend], call: Call) -> Backend | None:
        names = {backend.name for backend in candidates}
        config = call.config
        samples = [
            sample
            for sample in self.samples
            if sample.backend in names
            and sample.device == call.device_name
            and sample.grad == call.grad
            and sample.dtype == config["dtype"]
            and sample.args == config["args"]
            and sample.present == frozenset(config["present"])
        ]
        if not samples:
            return None
        nearest = min(self._distance(config, sample) for sample in samples)
        winner = self._fastest([sample for sample in samples if self._distance(config, sample) == nearest])
        return next((backend for backend in candidates if backend.name == winner), None)

    @staticmethod
    def _distance(config: Mapping[str, Any], sample: Sample) -> float:
        dims = config["dims"]
        if dims.keys() != sample.dims.keys():
            return math.inf
        distance = sum(abs(math.log2(max(1, dims[name])) - math.log2(max(1, sample.dims[name]))) for name in dims)
        return distance + 10 * (tuple(config["batch"]) != sample.batch)

    @staticmethod
    def _fastest(samples: Sequence[Sample]) -> str:
        times = {"torch": median(sample.time[1] for sample in samples)}
        for backend in {sample.backend for sample in samples}:
            times[backend] = median(sample.time[0] for sample in samples if sample.backend == backend)
        return min(times, key=lambda name: times[name])

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
                satisfies(spec, sample.dtype if name == "dtype" else (sample.dims | sample.args)[name])
                for name, spec in region.items()
            )
        ]
        if not samples:
            raise LookupError(f"{self.op.name}: no benchmark points match {region}")
        return self.op[self._fastest(samples)]

    def forget(self) -> None:
        with self._lock:
            self.__dict__.pop("rows", None)
            self.__dict__.pop("samples", None)
            self._selected.clear()
