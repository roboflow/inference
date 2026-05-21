"""Pinned assertions for the workflow error class hierarchy.

The CS-237 per-frame failure classifier matches on error class names. These
assertions encode the contracts that downstream code depends on; changing
the hierarchy without coordinating with CS-237 is a wire-breaking change.
"""

from inference.core.workflows.errors import (
    DynamicBlockCodeError,
    DynamicBlockError,
    DynamicBlockTimeoutError,
    WorkflowCompilerError,
    WorkflowExecutionEngineError,
)


def test_dynamic_block_timeout_error_is_execution_engine_side() -> None:
    """Per-frame timeout is a RUNTIME failure, not a COMPILE-time failure."""
    assert issubclass(DynamicBlockTimeoutError, DynamicBlockCodeError)
    assert issubclass(DynamicBlockTimeoutError, WorkflowExecutionEngineError)
    assert not issubclass(DynamicBlockTimeoutError, WorkflowCompilerError)
    assert not issubclass(DynamicBlockTimeoutError, DynamicBlockError)


def test_dynamic_block_code_error_remains_execution_engine_side() -> None:
    assert issubclass(DynamicBlockCodeError, WorkflowExecutionEngineError)
    assert not issubclass(DynamicBlockCodeError, WorkflowCompilerError)


def test_dynamic_block_error_remains_compiler_side() -> None:
    """`DynamicBlockError` is the compile/setup-time exception. Modal HTTP
    runtime failures must NOT raise this — they belong on the execution-engine
    side."""
    assert issubclass(DynamicBlockError, WorkflowCompilerError)
    assert not issubclass(DynamicBlockError, WorkflowExecutionEngineError)
