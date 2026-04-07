"""Errors raised by subworkflow composition validation."""


class SubworkflowCompositionError(Exception):
    """Base class for subworkflow composition failures at compile time."""


class SubworkflowCompositionCycleError(SubworkflowCompositionError):
    """Raised when the workflow composition graph contains a cycle."""


class SubworkflowNestingDepthError(SubworkflowCompositionError):
    """Raised when nesting from a root workflow exceeds the configured maximum depth."""
