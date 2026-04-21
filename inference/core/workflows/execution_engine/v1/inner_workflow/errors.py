"""Errors raised by inner-workflow compilation and composition validation."""

from typing import Optional

from inference.core.workflows.errors import (
    WorkflowCompilerError,
    WorkflowExecutionEngineError,
)

_INNER_WORKFLOW_COMPOSITION_CONTEXT = (
    "workflow_compilation | inner_workflow | composition"
)
_INNER_WORKFLOW_PARAMETER_BINDINGS_CONTEXT = (
    "workflow_compilation | inner_workflow | parameter_bindings"
)
_INNER_WORKFLOW_RUN_NOT_SUPPORTED_CONTEXT = (
    "workflow_execution | inner_workflow | run_not_supported"
)


class InnerWorkflowCompositionError(WorkflowCompilerError):
    """Base class for inner workflow composition failures at compile time."""

    def __init__(
        self,
        public_message: str,
        context: Optional[str] = None,
        inner_error: Optional[Exception] = None,
    ) -> None:
        super().__init__(
            public_message,
            context if context is not None else _INNER_WORKFLOW_COMPOSITION_CONTEXT,
            inner_error,
        )


class InnerWorkflowCompositionCycleError(InnerWorkflowCompositionError):
    """Raised when the workflow composition graph contains a cycle."""


class InnerWorkflowNestingDepthError(InnerWorkflowCompositionError):
    """Raised when nesting from a root workflow exceeds the configured maximum depth."""


class InnerWorkflowTotalCountError(InnerWorkflowCompositionError):
    """Raised when the number of ``inner_workflow`` steps in the composition tree exceeds the limit."""


class InnerWorkflowInvalidStepEntryError(InnerWorkflowCompositionError):
    """Raised when ``steps`` contains an entry that is not a JSON object (mapping)."""


class InnerWorkflowInliningStructureError(InnerWorkflowCompositionError):
    """Raised when ``inner_workflow`` inlining cannot make progress (unexpected nested structure)."""


class InnerWorkflowParameterBindingsError(WorkflowCompilerError):
    """Raised when an ``inner_workflow`` step's ``parameter_bindings`` fail compile-time checks."""

    def __init__(
        self,
        public_message: str,
        context: Optional[str] = None,
        inner_error: Optional[Exception] = None,
    ) -> None:
        super().__init__(
            public_message,
            (
                context
                if context is not None
                else _INNER_WORKFLOW_PARAMETER_BINDINGS_CONTEXT
            ),
            inner_error,
        )


class InnerWorkflowParameterBindingsUnknownInputError(
    InnerWorkflowParameterBindingsError
):
    """Raised when ``parameter_bindings`` keys are not declared child workflow inputs."""


class InnerWorkflowParameterBindingsMissingRequiredError(
    InnerWorkflowParameterBindingsError
):
    """Raised when required child workflow inputs have no ``parameter_bindings`` entry."""


class InnerWorkflowRunNotSupportedError(WorkflowExecutionEngineError):
    """Raised if the inner_workflow block's ``run()`` is invoked; the block is inlined at compile time."""

    def __init__(
        self,
        public_message: str,
        context: Optional[str] = None,
        inner_error: Optional[Exception] = None,
    ) -> None:
        super().__init__(
            public_message,
            (
                context
                if context is not None
                else _INNER_WORKFLOW_RUN_NOT_SUPPORTED_CONTEXT
            ),
            inner_error,
        )
