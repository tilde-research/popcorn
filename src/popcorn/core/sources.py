"""Backend library declarations and lazy source resolution.

A backend is a library (`fla`, `liger`); an implementation is named after the
backend providing it, optionally with a variant suffix (`fla:recurrent`). Every
lookup here takes an implementation name and resolves the backend from its prefix.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from types import ModuleType
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


def _failure(impl: str) -> BackendUnavailableError | BackendVersionError | None:
    declared = _BACKENDS.get(impl.split(":")[0])
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
        return BackendUnavailableError(f"implementation {impl!r} needs {declared.package!r}; install {install}")
    lo, hi = declared.min_version, declared.max_version
    if lo is None and hi is None:
        return None
    if (lo is not None and Version(declared.version) < Version(lo)) or (
        hi is not None and Version(declared.version) > Version(hi)
    ):
        return BackendVersionError(
            f"{declared.package} {declared.version} is outside the supported range "
            f"[{lo or '*'}, {hi or '*'}] required by implementation {impl!r}"
        )
    return None


def unavailable_reason(impl: str) -> str | None:
    """Why this implementation cannot run, or None when its backend is usable."""
    failure = _failure(impl)
    return str(failure) if failure else None


def available(impl: str) -> bool:
    """Whether the implementation's backend package is installed at a supported version."""
    return _failure(impl) is None


def installed_version(impl: str) -> str | None:
    """Installed distribution version of this implementation's backend, if it has one."""
    declared = _BACKENDS.get(impl.split(":")[0])
    if declared is None:
        return None
    _failure(impl)
    return declared.version


def ensure_available(impl: str) -> None:
    failure = _failure(impl)
    if failure:
        raise failure
    declared = _BACKENDS.get(impl.split(":")[0])
    if declared is not None and declared.min_version is None and declared.max_version is None and not declared.warned:
        warnings.warn(f"implementation {impl!r}: no supported version range declared for {declared.package}")
        declared.warned = True


def resolve(path: str) -> Any:
    """Resolve a dotted source path segment by segment, preferring attributes
    over same-named submodules like `from x import y` does — `fla.ops.utils.solve_tril`
    is the re-exported function, not its defining module. `pkg.mod.Class.apply`
    reaches methods, and a plain module path serves several callables to its
    adapter through `kernel` attributes."""
    segments = path.split(".")
    found: Any = importlib.import_module(segments[0])
    prefix = segments[0]
    for segment in segments[1:]:
        prefix = f"{prefix}.{segment}"
        try:
            found = getattr(found, segment)
        except AttributeError:
            if not isinstance(found, ModuleType):
                raise
            # Not yet imported (or genuinely absent): a missing dependency
            # inside an existing module surfaces as is.
            found = importlib.import_module(prefix)
    return found


_active_source: ContextVar[Callable[..., Any]] = ContextVar("popcorn_kernel")


class _Kernel:
    """The current implementation's source, bound by the dispatcher for the duration of
    a call. Calling it calls the source; attribute access reaches into it, so a
    module source serves several callables (`kernel.matmul`, `kernel.pack_weights`)
    and a class source serves its methods."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        try:
            source = _active_source.get()
        except LookupError:
            raise RuntimeError("kernel() is only valid inside an implementation registered with source=") from None
        return source(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        # AttributeError, not RuntimeError: introspection probes the unbound
        # proxy (hasattr, getattr with a default) and must see a plain miss.
        try:
            source = _active_source.get()
        except LookupError:
            raise AttributeError(f"kernel.{name} is only valid inside an implementation registered with source=") from None
        return getattr(source, name)


kernel = _Kernel()


@contextmanager
def bound_kernel(source: Callable[..., Any]) -> Iterator[None]:
    token = _active_source.set(source)
    try:
        yield
    finally:
        _active_source.reset(token)
