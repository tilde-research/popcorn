"""Backend package declarations and lazy source resolution."""

from __future__ import annotations

import importlib
import importlib.metadata
import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from packaging.version import Version

from popcorn.core.errors import BackendUnavailableError, BackendVersionError

_BACKENDS: dict[str, _BackendSpec] = {}


@dataclass(slots=True)
class _BackendSpec:
    """A declared backend package: its pin range, extra, and cached install state."""

    package: str
    min_version: str | None
    max_version: str | None
    extra: str | None
    installed: bool | None = None
    version: str | None = None
    warned: bool = False


def declare_backend(
    name: str,
    package: str,
    min_version: str | None = None,
    max_version: str | None = None,
    extra: str | None = None,
) -> None:
    _BACKENDS[name] = _BackendSpec(package, min_version, max_version, extra)


def _failure(backend_name: str) -> BackendUnavailableError | BackendVersionError | None:
    declared = _BACKENDS.get(backend_name.split(":")[0])
    if declared is None:
        return None
    if declared.installed is None:
        try:
            declared.version = importlib.metadata.version(declared.package)
            declared.installed = True
        except importlib.metadata.PackageNotFoundError:
            declared.installed = False
    if not declared.installed or declared.version is None:
        install = f"popcorn[{declared.extra}]" if declared.extra else declared.package
        return BackendUnavailableError(f"backend {backend_name!r} needs {declared.package!r}; install {install}")
    lo, hi = declared.min_version, declared.max_version
    if lo is None and hi is None:
        return None
    if (lo is not None and Version(declared.version) < Version(lo)) or (
        hi is not None and Version(declared.version) > Version(hi)
    ):
        return BackendVersionError(
            f"{declared.package} {declared.version} is outside the supported range "
            f"[{lo or '*'}, {hi or '*'}] of backend {backend_name!r}"
        )
    return None


def unavailable_reason(backend_name: str) -> str | None:
    """Why this backend cannot run, or None when its dependency is usable."""
    failure = _failure(backend_name)
    return str(failure) if failure else None


def available(backend_name: str) -> bool:
    """Whether the backend's package is installed at a supported version."""
    return _failure(backend_name) is None


def installed_version(backend_name: str) -> str | None:
    """Installed distribution version, if this backend has one."""
    declared = _BACKENDS.get(backend_name.split(":")[0])
    if declared is None:
        return None
    _failure(backend_name)
    return declared.version


def ensure_available(backend_name: str) -> None:
    failure = _failure(backend_name)
    if failure:
        raise failure
    declared = _BACKENDS.get(backend_name.split(":")[0])
    if declared is not None and declared.min_version is None and declared.max_version is None and not declared.warned:
        warnings.warn(f"backend {backend_name!r}: no supported version range declared for {declared.package}")
        declared.warned = True


def resolve(path: str) -> Any:
    """Resolve a dotted source path: import the longest module prefix, then walk
    attributes. `pkg.mod.Class.apply` reaches methods, and a plain module path
    serves several callables to its adapter through `kernel` attributes."""
    segments = path.split(".")
    found = importlib.import_module(segments[0])
    consumed = segments[0]
    for index, segment in enumerate(segments[1:], start=1):
        candidate = f"{consumed}.{segment}"
        try:
            found = importlib.import_module(candidate)
            consumed = candidate
        except ModuleNotFoundError as error:
            # Attributes start only where the module path ends; a missing
            # dependency inside an existing module surfaces as is.
            if error.name != candidate:
                raise
            for attribute in segments[index:]:
                found = getattr(found, attribute)
            return found
    return found


_active_source: ContextVar[Callable[..., Any]] = ContextVar("popcorn_kernel")


class _Kernel:
    """The current backend's source, bound by the dispatcher for the duration of
    a call. Calling it calls the source; attribute access reaches into it, so a
    module source serves several callables (`kernel.matmul`, `kernel.pack_weights`)
    and a class source serves its methods."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        try:
            source = _active_source.get()
        except LookupError:
            raise RuntimeError("kernel() is only valid inside a backend registered with source=") from None
        return source(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        # AttributeError, not RuntimeError: introspection probes the unbound
        # proxy (hasattr, getattr with a default) and must see a plain miss.
        try:
            source = _active_source.get()
        except LookupError:
            raise AttributeError(f"kernel.{name} is only valid inside a backend registered with source=") from None
        return getattr(source, name)


kernel = _Kernel()


@contextmanager
def bound_kernel(source: Callable[..., Any]) -> Iterator[None]:
    token = _active_source.set(source)
    try:
        yield
    finally:
        _active_source.reset(token)
