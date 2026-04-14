"""
Execution-engine handling for use_subworkflow steps (delegates to SubworkflowRunner).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from inference.core.workflows.errors import ExecutionEngineRuntimeError
from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.workflows.execution_engine.profiling.core import WorkflowsProfiler
from inference.core.workflows.execution_engine.v1.compiler.entities import CompiledWorkflow
from inference.core.workflows.execution_engine.v1.compiler.utils import (
    get_last_chunk_of_selector,
)
from inference.core.workflows.execution_engine.v1.executor.execution_data_manager.manager import (
    ExecutionDataManager,
)
from inference.core.workflows.execution_engine.v1.subworkflow.constants import (
    USE_SUBWORKFLOW_BLOCK_TYPE,
)
from inference.core.workflows.execution_engine.v1.subworkflow.runner import (
    LocalSubworkflowRunner,
    SubworkflowExecutionMode,
    SubworkflowRunner,
)


def is_use_subworkflow_step(workflow: CompiledWorkflow, step_name: str) -> bool:
    step = workflow.steps.get(step_name)
    if step is None:
        return False
    return step.manifest.type == USE_SUBWORKFLOW_BLOCK_TYPE


def _child_runtime_parameters(
    assembled_step_parameters: Dict[str, Any],
) -> Dict[str, Any]:
    bindings = assembled_step_parameters.get("parameter_bindings")
    if not isinstance(bindings, dict):
        raise RuntimeError(
            "use_subworkflow expected parameter_bindings dict in assembled step parameters."
        )
    return dict(bindings)


def _pick_runner(workflow: CompiledWorkflow) -> SubworkflowRunner:
    runner = workflow.init_parameters.get("workflows_core.subworkflow_runner")
    if runner is None:
        return LocalSubworkflowRunner()
    return runner


def _parent_run_context(
    workflow: CompiledWorkflow,
    kinds_serializers: Optional[Dict[str, Callable[[Any], Any]]],
    profiler: Optional[WorkflowsProfiler],
    executor: Any,
    step_error_handler: Optional[Callable[[str, Exception], None]],
) -> Dict[str, Any]:
    nested_max = min(WORKFLOWS_MAX_CONCURRENT_STEPS, 8)
    return {
        "max_concurrent_steps": nested_max,
        "kinds_serializers": kinds_serializers,
        "profiler": profiler,
        "executor": executor,
        "step_error_handler": step_error_handler,
    }


def _shape_use_subworkflow_child_result_for_simd_registration(
    collapsed: Dict[str, Any],
    *,
    nested_output_dimensionality_lift: int,
) -> Any:
    """
    When the compiler recorded a positive ``nested_output_dimensionality_lift`` (child
    exposes an expanded batch axis, e.g. ``dynamic_crop``), reshape a single dict whose
    values are lists of ``sv.Detections`` into a list of per-slice dicts for
    ``flatten_nested_output``. Otherwise return the collapsed dict unchanged.
    """
    import supervision as sv

    if nested_output_dimensionality_lift <= 0:
        return collapsed
    if len(collapsed) != 1:
        return [collapsed]
    key, value = next(iter(collapsed.items()))
    if (
        isinstance(value, list)
        and value
        and all(isinstance(v, sv.Detections) for v in value)
    ):
        return [{key: v} for v in value]
    return [collapsed]


def _collapse_child_run_result(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, list):
        if len(raw) == 0:
            raise ExecutionEngineRuntimeError(
                public_message="Nested workflow returned no output batches.",
                context="workflow_execution | use_subworkflow",
            )
        if len(raw) != 1:
            raise ExecutionEngineRuntimeError(
                public_message=(
                    "Nested workflow returned multiple output batches for a single parent slice; "
                    "ensure the child is invoked with a single batch per parent element."
                ),
                context="workflow_execution | use_subworkflow",
            )
        return raw[0]
    if isinstance(raw, dict):
        return raw
    raise ExecutionEngineRuntimeError(
        public_message=f"Unexpected nested workflow result type: {type(raw)}",
        context="workflow_execution | use_subworkflow",
    )


def run_use_subworkflow_simd(
    *,
    step_selector: str,
    workflow: CompiledWorkflow,
    execution_data_manager: ExecutionDataManager,
    profiler: WorkflowsProfiler,
    kinds_serializers: Optional[Dict[str, Callable[[Any], Any]]],
    executor: Any,
    step_error_handler: Optional[Callable[[str, Exception], None]],
) -> None:
    step_name = get_last_chunk_of_selector(selector=step_selector)
    child = workflow.nested_workflows.get(step_name)
    if child is None:
        raise RuntimeError(
            f"Missing nested workflow compilation for use_subworkflow step `{step_name}`."
        )
    runner = _pick_runner(workflow)
    parent_ctx = _parent_run_context(
        workflow,
        kinds_serializers=kinds_serializers,
        profiler=profiler,
        executor=executor,
        step_error_handler=step_error_handler,
    )
    indices: List = []
    results: List = []
    nested_lift = int(
        getattr(workflow.steps[step_name].manifest, "nested_output_dimensionality_lift", 0)
        or 0
    )
    with profiler.profile_execution_phase(
        name="iterative_step_code_execution",
        categories=["workflow_execution_operation", "workflow_block_operation"],
        metadata={"step": step_selector},
    ):
        for input_definition in execution_data_manager.iterate_over_simd_step_input(
            step_selector=step_selector
        ):
            with profiler.profile_execution_phase(
                name="step_code_execution",
                categories=["workflow_block_operation"],
                metadata={"step": step_selector},
            ):
                child_runtime = _child_runtime_parameters(input_definition.parameters)
                batch_result = runner.run(
                    compiled_child=child,
                    runtime_parameters=child_runtime,
                    mode=SubworkflowExecutionMode.LOCAL,
                    parent_context=parent_ctx,
                )
            collapsed = _collapse_child_run_result(batch_result)
            shaped = _shape_use_subworkflow_child_result_for_simd_registration(
                collapsed,
                nested_output_dimensionality_lift=nested_lift,
            )
            results.append(shaped)
            indices.append(input_definition.index)
    with profiler.profile_execution_phase(
        name="step_output_registration",
        categories=["workflow_execution_operation"],
        metadata={"step": step_selector},
    ):
        execution_data_manager.register_simd_step_output(
            step_selector=step_selector,
            indices=indices,
            outputs=results,
        )


def run_use_subworkflow_non_simd(
    *,
    step_selector: str,
    workflow: CompiledWorkflow,
    execution_data_manager: ExecutionDataManager,
    step_input: Dict[str, Any],
    profiler: WorkflowsProfiler,
    kinds_serializers: Optional[Dict[str, Callable[[Any], Any]]],
    executor: Any,
    step_error_handler: Optional[Callable[[str, Exception], None]],
) -> None:
    step_name = get_last_chunk_of_selector(selector=step_selector)
    child = workflow.nested_workflows.get(step_name)
    if child is None:
        raise RuntimeError(
            f"Missing nested workflow compilation for use_subworkflow step `{step_name}`."
        )
    runner = _pick_runner(workflow)
    parent_ctx = _parent_run_context(
        workflow,
        kinds_serializers=kinds_serializers,
        profiler=profiler,
        executor=executor,
        step_error_handler=step_error_handler,
    )
    with profiler.profile_execution_phase(
        name="step_code_execution",
        categories=["workflow_block_operation"],
        metadata={"step": step_selector},
    ):
        child_runtime = _child_runtime_parameters(step_input)
        batch_result = runner.run(
            compiled_child=child,
            runtime_parameters=child_runtime,
            mode=SubworkflowExecutionMode.LOCAL,
            parent_context=parent_ctx,
        )
    with profiler.profile_execution_phase(
        name="step_output_registration",
        categories=["workflow_execution_operation"],
        metadata={"step": step_selector},
    ):
        execution_data_manager.register_non_simd_step_output(
            step_selector=step_selector,
            output=_collapse_child_run_result(batch_result),
        )
