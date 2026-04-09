"""
Pluggable execution backends for nested workflows.

The execution engine dispatches use_subworkflow steps to a SubworkflowRunner
(see ``workflows_core.subworkflow_runner`` init parameter).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, Optional

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
    """In-process nested execution via the v1 workflow executor."""

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

        from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
        from inference.core.workflows.execution_engine.v1.executor.core import (
            run_workflow,
        )

        pc = parent_context or {}

        return run_workflow(
            workflow=compiled_child,
            runtime_parameters=runtime_parameters,
            max_concurrent_steps=pc.get("max_concurrent_steps", WORKFLOWS_MAX_CONCURRENT_STEPS),
            kinds_serializers=pc.get("kinds_serializers"),
            serialize_results=False,
            profiler=pc.get("profiler"),
            executor=pc.get("executor"),
            step_error_handler=pc.get("step_error_handler"),
        )
