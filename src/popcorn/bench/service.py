"""Runs registered backends through the comparison harness and records outcomes."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from itertools import chain, repeat
from typing import TYPE_CHECKING, Any

import torch

from popcorn.bench.compare import compare_inputs
from popcorn.bench.grid import cases, make_inputs
from popcorn.bench.model import Case, Environment, Record, Result
from popcorn.bench.store import Store, unmeasured
from popcorn.core.config import Call, device_name
from popcorn.core.errors import BackendUnavailableError, BackendVersionError, DispatchError
from popcorn.core.sources import installed_version

if TYPE_CHECKING:
    from popcorn.core.dispatcher import Dispatcher, Implementation

logger = logging.getLogger("popcorn.bench")


def _enabled(name: str) -> bool:
    return os.getenv(name, "").lower() in {"1", "true", "yes", "on"}


def validation_mode(bench: bool | None = None, validate: bool | None = None) -> str | None:
    """`POPCORN_BENCH=1` (or `bench=True`) lazily measures and records on first encounter."""
    if bench or _enabled("POPCORN_BENCH"):
        return "bench"
    return None


def _case_from_call(call: Call) -> Case:
    config = call.config
    return Case(
        tuple(config["dims"].items()),
        tuple(config["batch"]),
        getattr(torch, config["dtype"]),
        tuple(config["args"].items()),
        frozenset(config["present"]),
    )


class BenchmarkService:
    """Runs registered backends through `compare` and records the outcomes."""

    def __init__(self, op: Dispatcher, store: Store | None = None) -> None:
        self.op = op
        self.store = store or Store()

    def _record(self, backend: str, case: Case, result: Result, device: torch.device | str | None) -> Record:
        environment = Environment(
            device_name(device),
            torch.__version__,
            installed_version(backend),
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
            self.op.fingerprint,
            self._registered(backend).fingerprint,
        )
        return Record(self.op.name, backend, str(case), case.case_id, case.config(), environment, result)

    def _registered(self, backend: str) -> Implementation:
        return next(candidate for candidate in self.op._impls if candidate.name == backend)

    def _reference_bench(
        self,
        backend: str,
        case: Case,
        device: torch.device | str,
        grad: bool,
        benchmark: bool,
    ) -> Mapping[str, float] | None:
        if not benchmark:
            return None
        if backend == "torch":
            return {}  # one timing pass is enough when the subject is the reference
        return self.store.reference_bench(
            self.op.name,
            case.case_id,
            case.config(),
            device_name(device),
            torch.__version__,
            grad,
            self.op.fingerprint,
        )

    def _compare(
        self,
        chosen: Implementation,
        inputs: Iterable[Mapping[str, Any]],
        *,
        backward: bool,
        benchmark: bool,
        repeats: int,
        warmup: int,
        reference_bench: Mapping[str, float] | None = None,
    ) -> Result:
        try:
            return compare_inputs(
                lambda **kwargs: chosen.invoke(kwargs),
                self.op.reference,
                inputs,
                backward=backward,
                benchmark=benchmark,
                repeats=repeats,
                warmup=warmup,
                reference_bench=reference_bench,
            )
        except Exception as error:
            return Result("error", f"harness: {type(error).__name__}: {error}", grad=backward)

    def run_case(
        self,
        backend: str,
        case: Case,
        device: torch.device | str = "cuda",
        trials: int = 10,
        warmup: int = 2,
        benchmark: bool = True,
        grad: bool | None = None,
    ) -> Record:
        if trials < 1:
            raise ValueError("trials must be at least 1")
        registered = self._registered(backend)
        grad = not registered.forward_only if grad is None else grad
        try:
            chosen = self.op[backend]
        except (BackendUnavailableError, BackendVersionError) as error:
            return self._record(backend, case, Result("skip", str(error), grad=grad), device)
        if grad and chosen.forward_only:
            return self._record(backend, case, Result("skip", "no backward implementation", grad=grad), device)
        try:
            first = make_inputs(self.op, case, device, grad=grad)
        except torch.OutOfMemoryError as error:
            result = Result("oom", f"inputs: {error}", grad=grad)
            return self._record(backend, case, result, device)
        except Exception as error:
            result = Result("error", f"inputs: {type(error).__name__}: {error}", grad=grad)
            return self._record(backend, case, result, device)
        if rejection := chosen.rejects(self.op._values(first), first):
            return self._record(backend, case, Result("skip", rejection, grad=grad), device)
        inputs = chain((first,), (make_inputs(self.op, case, device, seed, grad) for seed in range(1, trials)))
        result = self._compare(
            chosen,
            inputs,
            backward=grad,
            benchmark=benchmark,
            repeats=trials,
            warmup=warmup,
            reference_bench=self._reference_bench(backend, case, device, grad, benchmark),
        )
        return self._record(backend, case, result, device)

    def run_backend(
        self,
        backend: str,
        device: torch.device | str = "cuda",
        trials: int = 10,
        limit: int | None = None,
    ) -> list[Record]:
        return [self.run_case(backend, case, device, trials) for case in cases(self.op, limit)]

    def run_call(
        self,
        backend: str,
        call: Call,
        arguments: Mapping[str, Any],
        *,
        benchmark: bool,
        repeats: int = 10,
        warmup: int = 2,
    ) -> Record:
        chosen = self.op[backend]
        if rejection := chosen.rejects(self.op._values(arguments), arguments):
            raise DispatchError(f"{self.op.name}: backend {backend!r} rejected call: {rejection}")
        case = _case_from_call(call)
        result = self._compare(
            chosen,
            repeat(arguments, repeats),
            backward=call.grad,
            benchmark=benchmark,
            repeats=repeats,
            warmup=warmup,
            reference_bench=self._reference_bench(backend, case, call.device, call.grad, benchmark),
        )
        return self._record(backend, case, result, call.device)

    def _run_and_store(
        self, call: Call, arguments: Mapping[str, Any], backends: Iterable[str], benchmark: bool
    ) -> list[Record]:
        records = [self.run_call(backend, call, arguments, benchmark=benchmark) for backend in backends]
        for record in records:
            result, bench = record.result, record.result.bench
            timing = " ".join(
                f"{name}={bench[name]:.3f}ms" for name in ("fwd_ms", "bwd_ms", "ref_fwd_ms", "ref_bwd_ms") if name in bench
            )
            logger.info(
                "%s:%s [%s] %s%s%s",
                record.op,
                record.impl,
                result.status,
                record.case,
                f" {timing}" if timing else "",
                f" ({result.reason})" if result.reason else "",
            )
        if records:
            self.store.write_user(records)
            logger.info("%s: recorded %d row(s) -> %s", self.op.name, len(records), self.store.user)
        return records

    def _pending(self, backend: str, call: Call, existing: Iterable[Record], benchmark: bool) -> bool:
        record = self.store.exact(
            self.op.name,
            backend,
            call,
            torch.__version__,
            installed_version(backend),
            existing,
            ref_hash=self.op.fingerprint,
            impl_hash=self._registered(backend).fingerprint,
        )
        return unmeasured(record, benchmark)

    def ensure(self, call: Call, arguments: Mapping[str, Any], backends: Iterable[str], *, benchmark: bool) -> list[Record]:
        """Fill in whatever conclusive (and, for bench mode, timed) records are still missing."""
        existing = self.store.merged(self.op.name)
        pending = [backend for backend in backends if self._pending(backend, call, existing, benchmark)]
        return self._run_and_store(call, arguments, pending, benchmark)

    def force(self, call: Call, arguments: Mapping[str, Any], backends: Iterable[str], *, benchmark: bool) -> list[Record]:
        return self._run_and_store(call, arguments, [backend for backend in backends if backend != "torch"], benchmark)

    def incomplete(
        self,
        backend: str,
        case: Case,
        device: torch.device | str,
        reason: str = "worker did not complete case",
        grad: bool | None = None,
    ) -> Record:
        result = Result("error", reason, grad=not self._registered(backend).forward_only if grad is None else grad)
        return self._record(backend, case, result, device)
