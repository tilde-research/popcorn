"""Public core API: the dispatcher, registration, spaces, and errors."""

from popcorn.core.dispatcher import KERNELS, Dispatcher, register_kernel
from popcorn.core.errors import BackendUnavailableError, BackendVersionError, DispatchError, PopcornError
from popcorn.core.sources import declare_backend, kernel
from popcorn.core.args import KNOWN_ARGS, NEUTRAL_ARGS, SPEED_ARGS
from popcorn.core.policy import Policy
from popcorn.core.spaces import Div, Pow2, Range, Real, Space
from popcorn.core.tags import Tag
from popcorn.core.tuning import Tuner

__all__ = [
    "KERNELS",
    "Dispatcher",
    "register_kernel",
    "kernel",
    "declare_backend",
    "Tag",
    "Tuner",
    "Policy",
    "Space",
    "Range",
    "Real",
    "Div",
    "Pow2",
    "SPEED_ARGS",
    "NEUTRAL_ARGS",
    "KNOWN_ARGS",
    "PopcornError",
    "DispatchError",
    "BackendUnavailableError",
    "BackendVersionError",
]
