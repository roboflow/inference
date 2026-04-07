"""
Pluggable execution backends for nested workflows.

The execution engine dispatches sub-workflow steps to a :class:`SubworkflowRunner`
implementation supplied via ``init_parameters`` (design target; wiring is future work).

Blocks must not instantiate the execution engine; see docs/workflows/subworkflow_design.md.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from inference.core.workflows.execution_engine.v1.compiler.entities import (
        CompiledWorkflow,
    )


class SubworkflowExecutionMode(str, Enum):
    """How a nested workflow run should be carried out."""

    LOCAL = "local"
    REMOTE_SYNC = "remote_sync"
    REMOTE_ASYNC = "remote_async"


class SubworkflowRunner(ABC):
    """
    Engine-level strategy for executing a compiled child workflow.

    Implementations may run in-process, call a remote API synchronously, or enqueue
    async work; the block manifest may declare a preferred mode, subject to policy.
    """

    @abstractmethod
    def run(
        self,
        *,
        compiled_child: "CompiledWorkflow",
        runtime_parameters: Dict[str, Any],
        mode: SubworkflowExecutionMode,
        parent_context: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Execute the child workflow and return a result in the shape the parent step expects.

        For ``REMOTE_ASYNC``, a future design may return a handle or partial result;
        not implemented in the base OSS stub.
        """
        raise NotImplementedError


class LocalSubworkflowRunner(SubworkflowRunner):
    """
    In-process nested execution (stub).

    Full implementation will delegate to the same executor path as the root workflow
    without embedding that logic inside the ``use_subworkflow`` block class.
    """

    def run(
        self,
        *,
        compiled_child: "CompiledWorkflow",
        runtime_parameters: Dict[str, Any],
        mode: SubworkflowExecutionMode,
        parent_context: Optional[Dict[str, Any]] = None,
    ) -> Any:
        if mode is not SubworkflowExecutionMode.LOCAL:
            raise NotImplementedError(
                f"{self.__class__.__name__} only supports "
                f"{SubworkflowExecutionMode.LOCAL.value!r} for now."
            )
        raise NotImplementedError(
            "LocalSubworkflowRunner.run is a placeholder; wire _run_workflow / "
            "ExecutionEngine execution here without calling from the block class."
        )
