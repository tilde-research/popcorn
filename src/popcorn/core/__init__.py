"""Public core API: the dispatcher, registration, constraints, and errors."""

from popcorn.core.constraints import Div, Pow2, Range
from popcorn.core.dispatcher import KERNELS, Dispatcher, register_kernel
from popcorn.core.errors import BackendUnavailableError, BackendVersionError, DispatchError, PopcornError, UnvalidatedWarning
from popcorn.core.sources import declare_backend, kernel
from popcorn.core.tuning import Tuner

__all__ = [
    "KERNELS",
    "Dispatcher",
    "register_kernel",
    "kernel",
    "declare_backend",
    "Tuner",
    "Range",
    "Div",
    "Pow2",
    "PopcornError",
    "DispatchError",
    "BackendUnavailableError",
    "BackendVersionError",
    "UnvalidatedWarning",
]
