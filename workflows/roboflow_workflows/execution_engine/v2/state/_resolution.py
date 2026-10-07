"""Source of the block call running now, for source-scoped state.

The runtime owns the policy: ``ExecutionContext.source_id`` gives the source
of a per-invocation call and ``ExecutionContext.source_id_at(index)`` the
source of one index. ``source_id`` rejects batch-delivering calls even when
every member shares one source. This module only turns the runtime's errors
into ``StateScopeError``.
"""

from typing import Tuple

from roboflow_workflows.execution_engine.v2.context import (
    ExecutionContext,
    NoExecutionContextError,
    get_execution_context,
)
from roboflow_workflows.execution_engine.v2.errors import ContractError
from roboflow_workflows.execution_engine.v2.state.errors import StateScopeError


def current_source_id() -> str:
    """Return the source of the current per-invocation call."""
    context = _current_context(operation="state.source")
    try:
        source_id = context.source_id
    except ContractError as error:
        raise StateScopeError(
            f"state.source: {error}; use state.at(index), state.for_source(...) "
            "or state.global_"
        ) from error

    return source_id


def source_id_at(index: Tuple[int, ...]) -> str:
    """Return the source of one index of the current call."""
    context = _current_context(operation="state.at")
    try:
        source_id = context.source_id_at(tuple(index))
    except ContractError as error:
        raise StateScopeError(f"state.at({tuple(index)}): {error}") from error

    return source_id


def _current_context(*, operation: str) -> ExecutionContext:
    try:
        context = get_execution_context()
    except NoExecutionContextError as error:
        raise StateScopeError(
            f"{operation} is available only inside a block call; outside one use "
            "state.for_source(...)"
        ) from error

    return context
