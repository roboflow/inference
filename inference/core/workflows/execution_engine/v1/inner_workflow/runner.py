"""
Pluggable execution backends for nested (inner) workflows.

The execution engine dispatches ``roboflow_core/inner_workflow@v1`` steps to an InnerWorkflowRunner
(see ``workflows_core.inner_workflow_runner`` init parameter).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from inference.core.workflows.execution_engine.v1.compiler.entities import (
        CompiledWorkflow,
    )


class InnerWorkflowExecutionMode(str, Enum):
    """How a nested workflow run should be carried out."""

    LOCAL = "local"
    REMOTE_SYNC = "remote_sync"
    REMOTE_ASYNC = "remote_async"


class InnerWorkflowRunner(ABC):
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
        mode: InnerWorkflowExecutionMode,
        parent_context: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Execute the child workflow and return a result in the shape the parent step expects.

        For ``REMOTE_ASYNC``, a future design may return a handle or partial result;
        not implemented in the base OSS stub.
        """
        raise NotImplementedError


class LocalInnerWorkflowRunner(InnerWorkflowRunner):
    """In-process nested execution via the v1 workflow executor."""

    def run(
        self,
        *,
        compiled_child: "CompiledWorkflow",
        runtime_parameters: Dict[str, Any],
        mode: InnerWorkflowExecutionMode,
        parent_context: Optional[Dict[str, Any]] = None,
    ) -> Any:
        if mode is not InnerWorkflowExecutionMode.LOCAL:
            raise NotImplementedError(
                f"{self.__class__.__name__} only supports "
                f"{InnerWorkflowExecutionMode.LOCAL.value!r} for now."
            )

        from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
        from inference.core.workflows.execution_engine.profiling.core import (
            NullWorkflowsProfiler,
        )
        from inference.core.workflows.execution_engine.v1.executor.core import (
            run_workflow,
        )
        from inference.core.workflows.execution_engine.v1.executor.runtime_input_assembler import (
            assemble_runtime_parameters,
        )
        from inference.core.workflows.execution_engine.v1.executor.runtime_input_validator import (
            validate_runtime_input,
        )

        pc = parent_context or {}
        profiler = pc.get("profiler") or NullWorkflowsProfiler.init()
        assembled = assemble_runtime_parameters(
            runtime_parameters=dict(runtime_parameters),
            defined_inputs=compiled_child.workflow_definition.inputs,
            kinds_deserializers=compiled_child.kinds_deserializers,
            prevent_local_images_loading=pc.get("prevent_local_images_loading", False),
            profiler=profiler,
        )
        validate_runtime_input(
            runtime_parameters=assembled,
            input_substitutions=compiled_child.input_substitutions,
            profiler=profiler,
        )

        return run_workflow(
            workflow=compiled_child,
            runtime_parameters=assembled,
            max_concurrent_steps=pc.get(
                "max_concurrent_steps", WORKFLOWS_MAX_CONCURRENT_STEPS
            ),
            kinds_serializers=pc.get("kinds_serializers"),
            serialize_results=False,
            profiler=profiler,
            executor=pc.get("executor"),
            step_error_handler=pc.get("step_error_handler"),
        )
