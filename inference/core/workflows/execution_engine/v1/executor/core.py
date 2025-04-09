from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Set

from inference.core import logger
from inference.core.workflows.errors import (
    ExecutionEngineRuntimeError,
    StepExecutionError,
    WorkflowError,
)
from inference.core.workflows.execution_engine.profiling.core import (
    NullWorkflowsProfiler,
    WorkflowsProfiler,
    execution_phase,
)
from inference.core.workflows.execution_engine.v1.compiler.entities import (
    CompiledWorkflow,
)
from inference.core.workflows.execution_engine.v1.compiler.utils import (
    get_last_chunk_of_selector,
)
from inference.core.workflows.execution_engine.v1.executor.execution_data_manager.manager import (
    ExecutionDataManager,
)
from inference.core.workflows.execution_engine.v1.executor.flow_coordinator import (
    ParallelStepExecutionCoordinator,
)
from inference.core.workflows.execution_engine.v1.executor.output_constructor import (
    construct_workflow_output,
)
from inference.core.workflows.execution_engine.v1.executor.utils import (
    run_steps_in_parallel,
)
from inference.core.workflows.prototypes.block import WorkflowBlock
from inference.usage_tracking.collector import usage_collector


@usage_collector("workflows")
@execution_phase(
    name="workflow_execution",
    categories=["execution_engine_operation"],
)
def run_workflow(
    workflow: CompiledWorkflow,
    runtime_parameters: Dict[str, Any],
    max_concurrent_steps: int,
    kinds_serializers: Optional[Dict[str, Callable[[Any], Any]]],
    serialize_results: bool = False,
    profiler: Optional[WorkflowsProfiler] = None,
    executor: Optional[ThreadPoolExecutor] = None,
) -> List[Dict[str, Any]]:
    execution_data_manager = ExecutionDataManager.init(
        execution_graph=workflow.execution_graph,
        runtime_parameters=runtime_parameters,
    )
    execution_coordinator = ParallelStepExecutionCoordinator.init(
        execution_graph=workflow.execution_graph,
    )
    next_steps = execution_coordinator.get_steps_to_execute_next(profiler=profiler)
    while next_steps is not None:
        execute_steps(
            next_steps=next_steps,
            workflow=workflow,
            execution_data_manager=execution_data_manager,
            max_concurrent_steps=max_concurrent_steps,
            profiler=profiler,
            executor=executor,
        )
        next_steps = execution_coordinator.get_steps_to_execute_next(profiler=profiler)
    with profiler.profile_execution_phase(
        name="outputs_construction",
        categories=["execution_engine_operation"],
    ):
        return construct_workflow_output(
            workflow_outputs=workflow.workflow_definition.outputs,
            execution_graph=workflow.execution_graph,
            execution_data_manager=execution_data_manager,
            serialize_results=serialize_results,
            kinds_serializers=kinds_serializers,
        )


@execution_phase(
    name="group_of_steps_execution",
    categories=["execution_engine_operation"],
    runtime_metadata=["next_steps", "max_concurrent_steps"],
)
def execute_steps(
    next_steps: List[str],
    workflow: CompiledWorkflow,
    execution_data_manager: ExecutionDataManager,
    max_concurrent_steps: int,
    profiler: Optional[WorkflowsProfiler] = None,
    executor: Optional[ThreadPoolExecutor] = None,
) -> None:
    logger.info(f"Executing steps: {next_steps}.")
    steps_functions = [
        partial(
            safe_execute_step,
            step_selector=step_selector,
            workflow=workflow,
            execution_data_manager=execution_data_manager,
            profiler=profiler,
        )
        for step_selector in next_steps
    ]
    _ = run_steps_in_parallel(
        steps=steps_functions, max_workers=max_concurrent_steps, executor=executor
    )


@execution_phase(
    name="step_execution",
    categories=["execution_engine_operation"],
    runtime_metadata=["step_selector"],
)
def safe_execute_step(
    step_selector: str,
    workflow: CompiledWorkflow,
    execution_data_manager: ExecutionDataManager,
    profiler: Optional[WorkflowsProfiler] = None,
) -> None:
    if profiler is None:
        profiler = NullWorkflowsProfiler.init()
    try:
        logger.info(
            f"started execution of: {step_selector} - {datetime.now().isoformat()}"
        )
        run_step(
            step_selector=step_selector,
            workflow=workflow,
            execution_data_manager=execution_data_manager,
            profiler=profiler,
        )
        logger.info(
            f"finished execution of: {step_selector} - {datetime.now().isoformat()}"
        )
    except WorkflowError as error:
        raise error
    except Exception as error:
        logger.exception(f"Execution of step {step_selector} encountered error.")
        step_name = get_last_chunk_of_selector(selector=step_selector)
        raise StepExecutionError(
            block_id=step_name,
            block_type=workflow.steps[step_name].manifest.type,
            public_message=str(error),
            context="workflow_execution | step_execution",
            inner_error=str(error),
        ) from error


def run_step(
    step_selector: str,
    workflow: CompiledWorkflow,
    execution_data_manager: ExecutionDataManager,
    profiler: WorkflowsProfiler,
) -> None:
    if execution_data_manager.is_step_simd(step_selector=step_selector):
        return run_simd_step(
            step_selector=step_selector,
            workflow=workflow,
            execution_data_manager=execution_data_manager,
            profiler=profiler,
        )
    return run_non_simd_step(
        step_selector=step_selector,
        workflow=workflow,
        execution_data_manager=execution_data_manager,
        profiler=profiler,
    )


def run_simd_step(
    step_selector: str,
    workflow: CompiledWorkflow,
    execution_data_manager: ExecutionDataManager,
    profiler: Optional[WorkflowsProfiler] = None,
) -> None:
    step_name = get_last_chunk_of_selector(selector=step_selector)
    step_instance = workflow.steps[step_name].step
    step_manifest = workflow.steps[step_name].manifest
    if step_manifest.accepts_batch_input():
        return run_simd_step_in_batch_mode(
            step_selector=step_selector,
            step_instance=step_instance,
            execution_data_manager=execution_data_manager,
            profiler=profiler,
        )
    return run_simd_step_in_non_batch_mode(
        step_selector=step_selector,
        step_instance=step_instance,
        execution_data_manager=execution_data_manager,
        profiler=profiler,
    )


def run_simd_step_in_batch_mode(
    step_selector: str,
    step_instance: WorkflowBlock,
    execution_data_manager: ExecutionDataManager,
    profiler: Optional[WorkflowsProfiler] = None,
) -> None:
    with profiler.profile_execution_phase(
        name="step_input_assembly",
        categories=["execution_engine_operation"],
        metadata={"step": step_selector},
    ):
        step_input = execution_data_manager.get_simd_step_input(
            step_selector=step_selector,
        )
    with profiler.profile_execution_phase(
        name="step_code_execution",
        categories=["workflow_block_operation"],
        metadata={
            "step": step_selector,
            "data_size": len(step_input.indices),
        },
    ):
        if not step_input.indices:
            # no inputs - discarded either by conditional exec or by not accepting empty
            outputs = []
        else:
            outputs = step_instance.run(**step_input.parameters)
    with profiler.profile_execution_phase(
        name="step_output_registration",
        categories=["execution_engine_operation"],
        metadata={"step": step_selector},
    ):
        execution_data_manager.register_simd_step_output(
            step_selector=step_selector,
            indices=step_input.indices,
            outputs=outputs,
        )


def run_simd_step_in_non_batch_mode(
    step_selector: str,
    step_instance: WorkflowBlock,
    execution_data_manager: ExecutionDataManager,
    profiler: Optional[WorkflowsProfiler] = None,
) -> None:
    indices, results = [], []
    with profiler.profile_execution_phase(
        name="iterative_step_code_execution",
        categories=["execution_engine_operation", "workflow_block_operation"],
        metadata={
            "step": step_selector,
        },
    ):
        for input_definition in execution_data_manager.iterate_over_simd_step_input(
            step_selector=step_selector
        ):
            with profiler.profile_execution_phase(
                name="step_code_execution",
                categories=["workflow_block_operation"],
                metadata={
                    "step": step_selector,
                },
            ):
                result = step_instance.run(**input_definition.parameters)
            results.append(result)
            indices.append(input_definition.index)
    with profiler.profile_execution_phase(
        name="step_output_registration",
        categories=["execution_engine_operation"],
        metadata={"step": step_selector},
    ):
        execution_data_manager.register_simd_step_output(
            step_selector=step_selector,
            indices=indices,
            outputs=results,
        )


def run_non_simd_step(
    step_selector: str,
    workflow: CompiledWorkflow,
    execution_data_manager: ExecutionDataManager,
    profiler: Optional[WorkflowsProfiler] = None,
) -> None:
    with profiler.profile_execution_phase(
        name="step_input_assembly",
        categories=["execution_engine_operation"],
        metadata={"step": step_selector},
    ):
        step_input = execution_data_manager.get_non_simd_step_input(
            step_selector=step_selector
        )
    if step_input is None:
        # discarded by conditional execution or empty value from upstream step
        return None
    step_name = get_last_chunk_of_selector(selector=step_selector)
    step_instance = workflow.steps[step_name].step
    with profiler.profile_execution_phase(
        name="step_code_execution",
        categories=["workflow_block_operation"],
        metadata={
            "step": step_selector,
        },
    ):
        step_result = step_instance.run(**step_input)
    if isinstance(step_result, list):
        raise ExecutionEngineRuntimeError(
            public_message=f"Error in execution engine. Non-SIMD step {step_name} "
            f"produced list of results which is not expected. This is most likely bug. "
            f"Contact Roboflow team through github issues "
            f"(https://github.com/roboflow/inference/issues) providing full context of"
            f"the problem - including workflow definition you use.",
            context="workflow_execution | step_output_registration",
        )
    with profiler.profile_execution_phase(
        name="step_output_registration",
        categories=["execution_engine_operation"],
        metadata={"step": step_selector},
    ):
        execution_data_manager.register_non_simd_step_output(
            step_selector=step_selector,
            output=step_result,
        )
