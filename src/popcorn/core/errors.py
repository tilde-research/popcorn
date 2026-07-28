"""Exception and warning types raised by the dispatcher and harness."""


class PopcornError(Exception):
    """Base class for all popcorn errors."""


class DispatchError(PopcornError):
    """No implementation can serve a call, or a forced one rejected it."""


class BackendUnavailableError(PopcornError):
    """The backend library's package is not installed."""


class BackendVersionError(PopcornError):
    """The backend library's package is installed at an unsupported version."""
