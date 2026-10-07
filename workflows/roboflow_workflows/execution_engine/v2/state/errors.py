"""Errors raised by managed state.

Every state error derives from ``StateError``. Backend failures say whether
the operation is known not to have happened::

    StateBackendError           the backend failed; the operation was not applied
    └─ StateOutcomeUnknownError a mutation was sent but its reply was lost: it
                                may or may not have been applied

Managed state never retries a mutation. The caller decides what an unknown
outcome means for its own data.
"""

__all__ = [
    "StateBackendError",
    "StateError",
    "StateOutcomeUnknownError",
    "StateOverflowError",
    "StateScopeError",
    "StateTypeError",
    "StateValueError",
]


class StateError(RuntimeError):
    """Base class of managed state errors."""


class StateValueError(StateError, ValueError):
    """A key, namespace, source, amount or value is not portable."""


class StateTypeError(StateError, TypeError):
    """``incr`` met a stored value that is not an integer."""


class StateOverflowError(StateError, OverflowError):
    """``incr`` would leave the signed 64-bit integer range; nothing changed."""


class StateScopeError(StateError, LookupError):
    """The current call has no source, or is batch-delivering (use ``at``)."""


class StateBackendError(StateError):
    """The backend failed and the operation was not applied."""


class StateOutcomeUnknownError(StateBackendError):
    """A mutation was sent, but whether the backend applied it is unknown."""
