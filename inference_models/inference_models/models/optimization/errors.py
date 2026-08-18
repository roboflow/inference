"""Errors shared by selectable inference-stage implementations."""

from typing import Optional


class RecoverableStageExecutionError(Exception):
    """Report an execution failure that may follow a declared stage fallback."""

    def __init__(self, message: str, help_url: Optional[str] = None) -> None:
        super().__init__(message)
        self._help_url = help_url

    @property
    def help_url(self) -> Optional[str]:
        """Return documentation associated with the eventual public error."""
        return self._help_url
