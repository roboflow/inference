"""Errors raised by inner-workflow composition validation."""


class InnerWorkflowCompositionError(Exception):
    """Base class for inner workflow composition failures at compile time."""


class InnerWorkflowCompositionCycleError(InnerWorkflowCompositionError):
    """Raised when the workflow composition graph contains a cycle."""


class InnerWorkflowNestingDepthError(InnerWorkflowCompositionError):
    """Raised when nesting from a root workflow exceeds the configured maximum depth."""
