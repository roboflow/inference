"""Errors shared by selectable inference-stage implementations."""

from inference_models.errors import ModelRuntimeError


class RecoverableStageExecutionError(ModelRuntimeError):
    """Report an execution failure that may follow a declared stage fallback."""
