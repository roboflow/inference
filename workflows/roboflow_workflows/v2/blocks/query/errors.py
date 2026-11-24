"""Errors raised by the V2 query helper.

A ``QueryDefinitionError`` means the literal query JSON is invalid or names an
undeclared runtime parameter. A ``QueryRuntimeError`` means a
valid query met a runtime value it cannot process. Both carry the JSON path,
the ``type`` literal involved and, when one is responsible, the parameter name.
Callers add their own step context.
"""

from typing import Any, Optional


class QueryError(Exception):
    """Base class for query helper errors.

    Attributes:
        path: JSON path of the failing node, rooted at ``$``.
        operation: ``type`` literal of the failing operation, comparator or
            structure, when known.
        parameter: Name of the responsible parameter, when one is responsible.
        detail: Human-readable description of the failure.
    """

    def __init__(
        self,
        detail: str,
        *,
        path: str,
        operation: Optional[str] = None,
        parameter: Optional[str] = None,
    ) -> None:
        location = path
        if operation is not None:
            location = f"{location} ({operation})"
        if parameter is not None:
            location = f"{location}, parameter `{parameter}`"
        super().__init__(f"{location}: {detail}")
        self.path = path
        self.operation = operation
        self.parameter = parameter
        self.detail = detail


class QueryDefinitionError(QueryError, ValueError):
    """The literal query is malformed or names an undeclared symbol."""


class QueryRuntimeError(QueryError, RuntimeError):
    """A runtime value cannot be processed by a valid query."""


class OperationFailure(Exception):
    """Raised by an operation function to name the responsible parameter."""

    def __init__(self, detail: str, *, parameter: Optional[str] = None) -> None:
        super().__init__(detail)
        self.parameter = parameter


def describe_value(value: Any) -> str:
    """Return a short, safe description of a runtime value for error messages."""
    description = f"value `{preview_value(value)}` of type {type(value).__name__}"

    return description


def preview_value(value: Any, max_characters: int = 128) -> str:
    """Stringify ``value``, truncated like V1 ``safe_stringify``.

    SequenceElementsCount uses this text as its result keys, so the format is
    part of observable behaviour.
    """
    try:
        text = str(value)
    except Exception:
        return "could not get string representation of value"
    if len(text) > max_characters:
        text = f"{text[:max_characters]} [...]"

    return text
