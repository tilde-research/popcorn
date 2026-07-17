"""The dispatcher: kernel references, registered backends, and call-time selection."""

from __future__ import annotations

import inspect
import re
import typing
from collections.abc import Callable, Mapping
from contextvars import ContextVar, Token
from functools import cached_property
from typing import Any

import torch

from popcorn.bench.model import Result
from popcorn.bench.service import BenchmarkService, validation_mode
from popcorn.core import annotations
from popcorn.core.config import call_config
from popcorn.core.constraints import check, fmt_value
from popcorn.core.errors import DispatchError
from popcorn.core.fingerprint import SCOPES, fingerprint
from popcorn.core.library import bind_torch_op
from popcorn.core.sources import bound_kernel, ensure_available, resolve, unavailable_reason
from popcorn.core.tuning import Tuner
from popcorn.core.typecheck import matches, narrows

KERNELS: dict[str, Dispatcher] = {}

_BACKEND_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]*(:[a-z0-9][a-z0-9_-]*)?$")


def _takes_context(transform: Callable[..., Any]) -> bool:
    try:
        signature = inspect.signature(transform)
    except (TypeError, ValueError):
        return False
    try:
        signature.bind(None)
        return False
    except TypeError:
        try:
            signature.bind(None, {}, None)
            return True
        except TypeError:
            raise TypeError(f"test input transform {transform!r} must accept one or three positional arguments") from None


class Backend:
    """One implementation of an op: an external `source` (a dotted path,
    resolved lazily and served as `kernel` while the adapter runs), an
    `adapter` bridging the reference signature to it, or both.

    A backend owns its eligibility -- `supports` constrains shapes
    values, `gates` are the narrowed annotations collected from its adapter,
    `predicate` is an arbitrary veto, and `forward_only=True` marks an
    implementation that must decline calls needing autograd -- and its
    lifecycle: conformance is validated when an adapter attaches, availability
    when first invoked.

    Calling a backend forces it for that call; entering it scopes every call
    in the block: `with rms_norm["fla"] as rms_norm: ...`.
    """

    def __init__(
        self,
        op: Dispatcher,
        name: str,
        adapter: Callable[..., Any] | None = None,
        source: str | None = None,
        supports: Mapping[str, Any] | None = None,
        predicate: Callable[..., bool] | None = None,
        forward_only: bool = False,
    ) -> None:
        if not _BACKEND_NAME.match(name):
            raise ValueError(f"invalid backend name {name!r}")
        self.op = op
        self.name = name
        self.adapter = adapter
        self.source = source
        self.supports: dict[str, Any] = dict(supports or {})
        self.predicate = predicate
        self.forward_only = forward_only
        self.gates: dict[str, Any] = {}
        self.source_fn: Callable[..., Any] | None = None
        self._prepared = False
        op._validate_keys(self.supports, "supports", op._dims)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.op(*args, backend=self.name, **kwargs)

    def __enter__(self) -> Dispatcher:
        """Scope the op to this backend: `with rms_norm["fla"] as rms_norm: ...`"""
        tokens = self.op._scope_tokens
        tokens.set((*tokens.get(), self.op._scoped.set(self.name)))
        return self.op

    def __exit__(self, *exc: object) -> None:
        *tokens, token = self.op._scope_tokens.get()
        self.op._scoped.reset(token)
        self.op._scope_tokens.set(tuple(tokens))

    def __str__(self) -> str:
        return f"{self.op.name}:{self.name}"

    def __repr__(self) -> str:
        return f"<{self}" + "".join(f" {note}" for note in self.notes()) + ">"

    def notes(self) -> list[str]:
        parts = ["fwd-only"] if self.forward_only else []
        parts += [f"supports={self.supports}"] if self.supports else []
        return parts + ([f"gates={self.gates}"] if self.gates else [])

    @cached_property
    def fingerprint(self) -> str | None:
        """Digest of the code this backend would run; None disables staleness checks."""
        parts = [self.adapter, self.predicate]
        if self.source is not None and self.source.startswith(SCOPES):
            try:
                self._prepare()  # first-party sources live in popcorn and carry no version signal
            except Exception:
                return None
            parts.append(self.source_fn)
        return fingerprint(*(part for part in parts if part is not None))

    def rejects(self, values: Mapping[str, Any], arguments: Mapping[str, Any]) -> str | None:
        """Why this backend must decline the call, or None if it is eligible."""
        if reason := unavailable_reason(self.name):
            return reason
        if self.forward_only and self.op._needs_grad(arguments):
            return "no backward implementation"
        if (rejection := check(self.supports, values)) is not None:
            return rejection
        for name, annotation in self.gates.items():
            if not matches(arguments[name], annotation):
                return f"{name}={fmt_value(arguments[name])} does not satisfy {annotation}"
        if self.predicate is not None and not self.predicate(**arguments):
            return f"rejected by {getattr(self.predicate, '__name__', 'predicate')}"
        return None

    def attach(self, adapter: Callable[..., Any]) -> None:
        """Validate and adopt an adapter: it must declare exactly the reference
        parameters, none with defaults (the reference owns them); annotations
        may narrow the reference types and become dispatch gates."""
        params = list(inspect.signature(adapter).parameters.values())
        names = tuple(p.name for p in params)
        if names != self.op._params:
            raise TypeError(f"{self}: parameters must be exactly {list(self.op._params)}, got {list(names)}")
        if defaulted := [p.name for p in params if p.default is not inspect.Parameter.empty]:
            raise TypeError(f"{self}: parameters must not declare defaults (the reference owns them): {defaulted}")
        gates = {}
        for p in params:
            reference = self.op._signature.parameters[p.name].annotation
            if p.annotation is inspect.Parameter.empty or p.annotation == reference:
                continue
            if not narrows(p.annotation, reference):
                raise TypeError(f"{self}: {p.name}: {p.annotation} does not narrow {reference}")
            gates[p.name] = p.annotation
        self.gates = gates
        self.adapter = adapter

    def invoke(self, arguments: Mapping[str, Any]) -> Any:
        """Run on already-bound arguments; `__call__` is the public form."""
        self._prepare()
        args = tuple(arguments[p] for p in self.op._params)
        if self.adapter is None:
            assert self.source_fn is not None
            return self.source_fn(*args)
        if self.source_fn is not None:
            with bound_kernel(self.source_fn):
                return self.adapter(*args)
        return self.adapter(*args)

    def _prepare(self) -> None:
        if self._prepared:
            return
        if self.adapter is None and self.source is None:
            raise DispatchError(f"{self}: registration has neither adapter nor source")
        ensure_available(self.name)
        if self.source is not None:
            self.source_fn = resolve(self.source)
            if self.adapter is None:
                self._conform_source(self.source_fn)
        self._prepared = True

    def _conform_source(self, fn: Callable[..., Any] | None) -> None:
        """A bare source must already look like the reference; otherwise demand an adapter."""
        if fn is None:
            return

        try:
            params = list(inspect.signature(fn).parameters.values())
        except (TypeError, ValueError):
            return
        kinds = {p.kind for p in params}
        if inspect.Parameter.VAR_POSITIONAL in kinds or inspect.Parameter.VAR_KEYWORD in kinds:
            return
        names = tuple(p.name for p in params[: len(self.op._params)])
        extra = params[len(self.op._params) :]
        if names != self.op._params or any(p.default is inspect.Parameter.empty for p in extra):
            raise TypeError(
                f"{self}: source signature {list(names)} does not match reference {list(self.op._params)}; register an adapter"
            )


class Dispatcher:
    """An op: the torch reference plus registered backends, called like the
    reference. The backend is decided at call time, never stored: the user
    forces one (`backend=`, `op["fla"](...)`, or `with op["fla"]:`), otherwise
    the tuner picks the fastest eligible one from recorded benchmarks. The
    tuner's per-signature memoization is the only cache, purely an optimization.
    """

    def __init__(
        self,
        reference: Callable[..., Any],
        test_shapes: Mapping[str, Any] | None = None,
        test_args: Mapping[str, Any] | None = None,
        test_inputs: Mapping[str, Callable[..., Any]] | None = None,
        name: str | None = None,
    ) -> None:
        # identity: look like the reference to introspection
        self.reference = reference
        self.name = name or reference.__name__
        self.__name__ = self.name
        self.__doc__ = reference.__doc__

        # signature: the reference's, plus the keyword-only dispatch controls
        self._signature = inspect.signature(reference)
        self._params = tuple(self._signature.parameters)
        controls = [
            inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY, default=None, annotation=annotation)
            for name, annotation in (("backend", str), ("bench", bool), ("validate", bool))
        ]
        self.__signature__ = self._signature.replace(parameters=[*self._signature.parameters.values(), *controls])

        # shape grammar parsed from the annotations
        self.specs = annotations.plans(self._signature)
        self._dims = annotations.dim_names(self.specs)

        # test grid metadata
        self.test_shapes: dict[str, Any] = dict(test_shapes or {})
        self._validate_keys(self.test_shapes, "test_shapes", self._dims)
        self.arg_pools = self._arg_pools(dict(test_args or {}))
        self.test_inputs: dict[str, Callable[..., Any]] = dict(test_inputs or {})  # tensor param -> input transform
        self._validate_keys(self.test_inputs, "test_inputs", {spec.param for spec in self.specs})
        if invalid := [name for name, transform in self.test_inputs.items() if not callable(transform)]:
            raise TypeError(f"{self.name}: test_inputs values must be callable: {invalid}")
        self._contextual_inputs = frozenset(name for name, transform in self.test_inputs.items() if _takes_context(transform))

        # dispatch state
        self.torch_op: Any = None  # torch.library binding, set by bind_torch_op
        self._backends: list[Backend] = [Backend(self, "torch", adapter=reference)]
        self.bench = BenchmarkService(self)
        self.tuner = Tuner(self, self.bench.store)
        self._scoped: ContextVar[str | None] = ContextVar(f"popcorn.{self.name}", default=None)
        self._scope_tokens: ContextVar[tuple[Token[str | None], ...]] = ContextVar(
            f"popcorn.{self.name}.scope_tokens", default=()
        )

    @cached_property
    def fingerprint(self) -> str | None:
        """Digest of the reference; a changed reference invalidates every record of the op."""
        return fingerprint(self.reference)

    def __call__(
        self,
        *args: Any,
        backend: str | None = None,
        bench: bool | None = None,
        validate: bool | None = None,
        **kwargs: Any,
    ) -> Any:
        arguments = self._bind(*args, **kwargs)
        values = self._values(arguments)
        forced = backend or self._scoped.get()
        candidates = self._eligible(values, arguments, forced)
        call = call_config(self, values, arguments)
        if mode := validation_mode(bench, validate):
            if call is None:
                raise DispatchError(f"{self.name}: this call cannot be represented as a validation case")
            names = [candidate.name for candidate in candidates if candidate.name != "torch"]
            if self.bench.ensure(call, arguments, names, benchmark=mode == "bench"):
                self.tuner.forget()
        selected = self.tuner.select(
            call,
            candidates,
            forced,
            self._cache_signature(arguments),
            require_pass=bool(mode),
        )
        return selected.invoke(arguments)

    def validate(self, *args: Any, backend: str | None = None, **kwargs: Any) -> list[Result]:
        return self._assess(args, kwargs, backend, benchmark=False)

    def benchmark(self, *args: Any, backend: str | None = None, **kwargs: Any) -> list[Result]:
        return self._assess(args, kwargs, backend, benchmark=True)

    def __getitem__(self, name: str) -> Backend:
        for backend in self._backends:
            if backend.name == name:
                ensure_available(name)  # forcing an uninstalled backend errors with the install hint
                return backend
        raise DispatchError(f"{self.name}: unknown backend {name!r}; available: {', '.join(self.available_backends())}")

    def register(
        self,
        name: str,
        source: str | None = None,
        supports: Mapping[str, Any] | None = None,
        predicate: Callable[..., bool] | None = None,
        forward_only: bool = False,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        if any(b.name == name for b in self._backends):
            raise ValueError(f"{self.name}: backend {name!r} already registered")
        backend = Backend(self, name, source=source, supports=supports, predicate=predicate, forward_only=forward_only)
        self._backends.insert(-1, backend)  # registration order, torch reference last
        self.tuner.forget()

        def attach(fn: Callable[..., Any]) -> Callable[..., Any]:
            try:
                backend.attach(fn)
            except Exception:
                self._backends.remove(backend)
                raise
            return fn

        return attach

    def available_backends(self) -> tuple[str, ...]:
        return tuple(b.name for b in self._backends)

    def __repr__(self) -> str:
        lines = [f"{self.name}{self._signature}"]
        for b in self._backends:
            lines.append(f"  {b.name}" + "".join(f"  {note}" for note in b.notes()))
        return "\n".join(lines)

    def _validate_keys(self, mapping: Mapping[str, Any], what: str, allowed: set[str]) -> None:
        unknown = set(mapping) - allowed
        if unknown:
            raise TypeError(f"{self.name}: unknown {what} keys {sorted(unknown)}; allowed: {sorted(allowed)}")

    def _arg_pools(self, test_args: dict[str, Any]) -> dict[str, list[Any] | None]:
        """Tested values per scalar argument: declared, or derived from the annotation/default.

        None means no envelope is declared; such arguments are never warned about
        and cannot be gridded by the harness."""
        tensors = {spec.param for spec in self.specs}
        if unknown := set(test_args) - (set(self._params) - tensors):
            raise TypeError(f"{self.name}: test_args keys {sorted(unknown)} are not scalar arguments")
        pools: dict[str, list[Any] | None] = {}
        for name, param in self._signature.parameters.items():
            if name in tensors:
                continue
            if name in test_args:
                pool = list(test_args[name])
                for value in pool:
                    if not matches(value, param.annotation):
                        raise TypeError(
                            f"{self.name}: test_args[{name!r}] value {value!r} does not satisfy {param.annotation}"
                        )
            elif typing.get_origin(param.annotation) is typing.Literal:
                pool = list(typing.get_args(param.annotation))
            elif param.annotation is bool or isinstance(param.default, bool):
                pool = [False, True]
            elif param.default is not inspect.Parameter.empty:
                pool = [param.default]
            else:
                pools[name] = None
                continue
            if param.default in pool:  # default-valued case comes first
                pool.remove(param.default)
                pool.insert(0, param.default)
            pools[name] = pool
        return pools

    def _values(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        values: dict[str, Any] = annotations.extract(self.specs, arguments)
        floating = {
            value.dtype for value in arguments.values() if isinstance(value, torch.Tensor) and value.is_floating_point()
        }
        if len(floating) == 1:
            values["dtype"] = floating.pop()
        return values

    def _bind(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        bound = self._signature.bind(*args, **kwargs)
        bound.apply_defaults()
        return bound.arguments

    def _eligible(self, values: Mapping[str, Any], arguments: Mapping[str, Any], forced: str | None = None) -> list[Backend]:
        if forced is not None:
            backend = self[forced]
            if rejection := backend.rejects(values, arguments):
                raise DispatchError(f"{self.name}: backend {forced!r} rejected call: {rejection}")
            return [backend]
        return [backend for backend in self._backends if backend.rejects(values, arguments) is None]

    def _assess(self, args: tuple[Any, ...], kwargs: dict[str, Any], backend: str | None, *, benchmark: bool) -> list[Result]:
        arguments = self._bind(*args, **kwargs)
        values = self._values(arguments)
        candidates = self._eligible(values, arguments, backend)
        call = call_config(self, values, arguments)
        if call is None:
            raise DispatchError(f"{self.name}: this call cannot be represented as a validation case")
        records = self.bench.force(
            call,
            arguments,
            [candidate.name for candidate in candidates],
            benchmark=benchmark,
        )
        self.tuner.forget()
        return [record.result for record in records]

    @staticmethod
    def _cache_signature(arguments: Mapping[str, Any]) -> tuple[Any, ...]:
        def key(value: Any) -> Any:
            if isinstance(value, torch.Tensor):
                return (tuple(value.shape), str(value.dtype), str(value.device), value.requires_grad)
            return value

        return torch.is_grad_enabled(), tuple((name, key(value)) for name, value in arguments.items())

    @staticmethod
    def _needs_grad(arguments: Mapping[str, Any]) -> bool:
        return torch.is_grad_enabled() and any(
            isinstance(value, torch.Tensor) and value.requires_grad for value in arguments.values()
        )


@typing.overload
def register_kernel(fn: Callable[..., Any]) -> Dispatcher: ...


@typing.overload
def register_kernel(
    *,
    test_shapes: Mapping[str, Any] | None = None,
    test_args: Mapping[str, Any] | None = None,
    test_inputs: Mapping[str, Callable[..., Any]] | None = None,
    name: str | None = None,
) -> Callable[[Callable[..., Any]], Dispatcher]: ...


def register_kernel(
    fn: Callable[..., Any] | None = None,
    *,
    test_shapes: Mapping[str, Any] | None = None,
    test_args: Mapping[str, Any] | None = None,
    test_inputs: Mapping[str, Callable[..., Any]] | None = None,
    name: str | None = None,
) -> Dispatcher | Callable[[Callable[..., Any]], Dispatcher]:
    def wrap(reference: Callable[..., Any]) -> Dispatcher:
        dispatcher = Dispatcher(reference, test_shapes=test_shapes, test_args=test_args, test_inputs=test_inputs, name=name)
        if dispatcher.name in KERNELS:
            raise ValueError(f"kernel {dispatcher.name!r} already registered")
        KERNELS[dispatcher.name] = dispatcher
        bind_torch_op(dispatcher)
        return dispatcher

    return wrap(fn) if fn is not None else wrap
