"""Errors shared by selectable inference-stage implementations."""


class RecoverableStageExecutionError(Exception):
    """Report an execution failure that may follow a declared stage fallback."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
