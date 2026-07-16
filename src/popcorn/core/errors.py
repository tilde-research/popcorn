"""Exception and warning types raised by the dispatcher and harness."""


class PopcornError(Exception):
    """Base class for all popcorn errors."""


class DispatchError(PopcornError):
    """No backend can serve a call, or a forced backend rejected it."""


class BackendUnavailableError(PopcornError):
    """The backend's package is not installed."""


class BackendVersionError(PopcornError):
    """The backend's package is installed at an unsupported version."""


class UnvalidatedWarning(UserWarning):
    """A selected backend has no recorded validation pass for the exact call."""
