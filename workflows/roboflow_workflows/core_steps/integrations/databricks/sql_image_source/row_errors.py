"""Bounded, value-free diagnostics for rejected Databricks metadata rows."""

from typing import Any

METADATA_ERROR_MESSAGES = {
    "METADATA_VALUE_TOO_LONG": "Metadata text exceeds 4096 characters.",
    "METADATA_TOO_DEEP": "Metadata exceeds 4 nesting levels.",
    "METADATA_TOO_MANY_ITEMS": "Metadata collection exceeds 256 items.",
    "METADATA_TOO_LARGE": "Serialized row metadata exceeds 64 KiB.",
}


class MetadataLimitExceeded(ValueError):
    """Identify a metadata limit without retaining the rejected value.

    Args:
        code (str): Stable key in METADATA_ERROR_MESSAGES.
        field (str | None): Affected top-level column, or None for the row limit.
    """

    def __init__(self, code: str, *, field: str | None = None):
        self.code = code
        self.field = field
        super().__init__(METADATA_ERROR_MESSAGES[code])


def validate_row_error(error: Any, *, max_rows: int) -> None:
    """Validate an error record before accepting it from the worker.

    Args:
        error (Any): Worker-provided diagnostic to validate.
        max_rows (int): Maximum fetched rows for this invocation.

    Raises:
        RuntimeError: The diagnostic has an unexpected schema or value.
    """
    if not isinstance(error, dict) or set(error) != {
        "row_index",
        "image_name",
        "field",
        "code",
        "message",
    }:
        raise RuntimeError("Invalid Databricks worker row error")

    if (
        type(error["row_index"]) is not int
        or not 0 <= error["row_index"] < max_rows
        or not isinstance(error["code"], str)
        or error["code"] not in METADATA_ERROR_MESSAGES
        or error["message"] != METADATA_ERROR_MESSAGES[error["code"]]
        or any(
            value is not None and (not isinstance(value, str) or len(value) > 255)
            for value in (error["image_name"], error["field"])
        )
    ):
        raise RuntimeError("Invalid Databricks worker row error")
