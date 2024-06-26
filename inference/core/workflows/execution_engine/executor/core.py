import asyncio
from datetime import datetime
from typing import Any, Dict, List

from inference.core import logger
from inference.core.workflows.errors import (
    ExecutionEngineRuntimeError,
    StepExecutionError,
    WorkflowError,
)
from inference.core.workflows.execution_engine.compiler.entities import CompiledWorkflow
from inference.core.workflows.execution_engine.compiler.utils import (
    get_last_chunk_of_selector,
)
from inference.core.workflows.execution_engine.executor.execution_data_manager.manager import (
    ExecutionDataManager,
)
from inference.core.workflows.execution_engine.executor.flow_coordinator import (
    ParallelStepExecutionCoordinator,
)
from inference.core.workflows.execution_engine.executor.output_constructor import (
    construct_workflow_output,
)
from inference.core.workflows.prototypes.block import WorkflowBlock
from inference_sdk.http.utils.iterables import make_batches


async def run_workflow(
    workflow: CompiledWorkflow,
    runtime_parameters: Dict[str, Any],
    max_concurrent_steps: int,
) -> List[Dict[str, Any]]:
    execution_data_manager = ExecutionDataManager.init(
        execution_graph=workflow.execution_graph,
        runtime_parameters=runtime_parameters,
    )
    execution_coordinator = ParallelStepExecutionCoordinator.init(
        execution_graph=workflow.execution_graph,
    )
    next_steps = execution_coordinator.get_steps_to_execute_next()
    while next_steps is not None:
        await execute_steps(
            next_steps=next_steps,
            workflow=workflow,
            execution_data_manager=execution_data_manager,
            max_concurrent_steps=max_concurrent_steps,
        )
        next_steps = execution_coordinator.get_steps_to_execute_next()
    return construct_workflow_output(
        workflow_outputs=workflow.workflow_definition.outputs,
        execution_graph=workflow.execution_graph,
        execution_data_manager=execution_data_manager,
    )


async def execute_steps(
    next_steps: List[str],
    workflow: CompiledWorkflow,
    execution_data_manager: ExecutionDataManager,
    max_concurrent_steps: int,
) -> None:
    logger.info(f"Executing steps: {next_steps}.")
    steps_batches = list(
        make_batches(iterable=next_steps, batch_size=max_concurrent_steps)
    )
    for steps_batch in steps_batches:
        logger.info(f"Steps batch: {steps_batch}")
        coroutines = [
            safe_execute_step(
                step_selector=step_selector,
                workflow=workflow,
                execution_data_manager=execution_data_manager,
            )
            for step_selector in steps_batch
        ]
        await asyncio.gather(*coroutines)


async def safe_execute_step(
    step_selector: str,
    workflow: CompiledWorkflow,
    execution_data_manager: ExecutionDataManager,
) -> None:
    try:
        logger.info(
            f"started execution of: {step_selector} - {datetime.now().isoformat()}"
        )
        await run_step(
            step_selector=step_selector,
            workflow=workflow,
            execution_data_manager=execution_data_manager,
        )
        logger.info(
            f"finished execution of: {step_selector} - {datetime.now().isoformat()}"
        )
    except WorkflowError as error:
        raise error
    except Exception as error:
        logger.exception(f"Execution of step {step_selector} encountered error.")
        raise StepExecutionError(
            public_message=f"Error during execution of step: {step_selector}. Details: {error}",
            context="workflow_execution | step_execution",
            inner_error=error,
        ) from error


async def run_step(
    step_selector: str,
    workflow: CompiledWorkflow,
    execution_data_manager: ExecutionDataManager,
) -> None:
    if execution_data_manager.is_step_simd(step_selector=step_selector):
        return await run_simd_step(
            step_selector=step_selector,
            workflow=workflow,
            execution_data_manager=execution_data_manager,
        )
    return await run_non_simd_step(
        step_selector=step_selector,
        workflow=workflow,
        execution_data_manager=execution_data_manager,
    )


async def run_simd_step(
    step_selector: str,
    workflow: CompiledWorkflow,
    execution_data_manager: ExecutionDataManager,
) -> None:
    step_name = get_last_chunk_of_selector(selector=step_selector)
    step_instance = workflow.steps[step_name].step
    step_manifest = workflow.steps[step_name].manifest
    if step_manifest.accepts_batch_input():
        return await run_simd_step_in_batch_mode(
            step_selector=step_selector,
            step_instance=step_instance,
            execution_data_manager=execution_data_manager,
        )
    return await run_simd_step_in_non_batch_mode(
        step_selector=step_selector,
        step_instance=step_instance,
        execution_data_manager=execution_data_manager,
    )


async def run_simd_step_in_batch_mode(
    step_selector: str,
    step_instance: WorkflowBlock,
    execution_data_manager: ExecutionDataManager,
) -> None:
    step_input = execution_data_manager.get_simd_step_input(step_selector=step_selector)
    if not step_input.indices:
        # no inputs - discarded either by conditional exec or by not accepting empty
        return None
    outputs = await step_instance.run(**step_input.parameters)
    execution_data_manager.register_simd_step_output(
        step_selector=step_selector,
        indices=step_input.indices,
        outputs=outputs,
    )


async def run_simd_step_in_non_batch_mode(
    step_selector: str,
    step_instance: WorkflowBlock,
    execution_data_manager: ExecutionDataManager,
) -> None:
    indices, results = [], []
    for input_definition in execution_data_manager.iterate_over_simd_step_input(
        step_selector=step_selector
    ):
        result = await step_instance.run(**input_definition.parameters)
        results.append(result)
        indices.append(input_definition.index)
    if not indices:
        return None
    execution_data_manager.register_simd_step_output(
        step_selector=step_selector,
        indices=indices,
        outputs=results,
    )


async def run_non_simd_step(
    step_selector: str,
    workflow: CompiledWorkflow,
    execution_data_manager: ExecutionDataManager,
) -> None:
    step_input = execution_data_manager.get_non_simd_step_input(
        step_selector=step_selector
    )
    if not step_input:
        # discarded by conditional execution
        return None
    step_name = get_last_chunk_of_selector(selector=step_selector)
    step_instance = workflow.steps[step_name].step
    step_result = await step_instance.run(**step_input)
    if isinstance(step_result, list):
        raise ExecutionEngineRuntimeError(
            public_message=f"Error in execution engine. Non-SIMD step {step_name} "
            f"produced list of results which is not expected. This is most likely bug. "
            f"Contact Roboflow team through github issues "
            f"(https://github.com/roboflow/inference/issues) providing full context of"
            f"the problem - including workflow definition you use.",
            context="workflow_execution | step_output_registration",
        )
    execution_data_manager.register_non_simd_step_output(
        step_selector=step_selector,
        output=step_result,
    )
