import asyncio
from datetime import datetime
from typing import Any, Dict, List, Set

from inference.core import logger
from inference.enterprise.workflows.complier.entities import StepExecutionMode
from inference.enterprise.workflows.complier.steps_executors.utils import make_batches
from inference.enterprise.workflows.entities.types import FlowControl
from inference.enterprise.workflows.errors import ExecutionEngineError
from inference.enterprise.workflows.execution_engine.compiler.entities import (
    CompiledWorkflow,
)
from inference.enterprise.workflows.execution_engine.compiler.utils import (
    get_last_chunk_of_selector,
)
from inference.enterprise.workflows.execution_engine.executor.execution_cache import (
    ExecutionCache,
)
from inference.enterprise.workflows.execution_engine.executor.flow_coordinator import (
    ParallelStepExecutionCoordinator,
    handle_flow_control,
)
from inference.enterprise.workflows.execution_engine.executor.output_constructor import (
    construct_workflow_output,
)
from inference.enterprise.workflows.execution_engine.executor.parameters_assembler import (
    assembly_step_parameters,
)


async def run_workflow(
    workflow: CompiledWorkflow,
    runtime_parameters: Dict[str, Any],
    max_concurrent_steps: int,
    step_execution_mode: StepExecutionMode,
) -> Dict[str, List[Any]]:
    execution_cache = ExecutionCache.init()
    execution_coordinator = ParallelStepExecutionCoordinator.init(
        execution_graph=workflow.execution_graph,
    )
    steps_to_discard = set()
    next_steps = execution_coordinator.get_steps_to_execute_next(
        steps_to_discard=steps_to_discard
    )
    while next_steps is not None:
        steps_to_discard = await execute_steps(
            next_steps=next_steps,
            workflow=workflow,
            runtime_parameters=runtime_parameters,
            execution_cache=execution_cache,
            max_concurrent_steps=max_concurrent_steps,
            step_execution_mode=step_execution_mode,
        )
        next_steps = execution_coordinator.get_steps_to_execute_next(
            steps_to_discard=steps_to_discard
        )
    return construct_workflow_output(
        workflow_definition=workflow.workflow_definition,
        execution_cache=execution_cache,
    )


async def execute_steps(
    next_steps: List[str],
    workflow: CompiledWorkflow,
    runtime_parameters: Dict[str, Any],
    execution_cache: ExecutionCache,
    max_concurrent_steps: int,
    step_execution_mode: StepExecutionMode,
) -> Set[str]:
    logger.info(f"Executing steps: {next_steps}. Execution mode: {step_execution_mode}")
    nodes_to_discard = set()
    steps_batches = list(
        make_batches(iterable=next_steps, batch_size=max_concurrent_steps)
    )
    for steps_batch in steps_batches:
        logger.info(f"Steps batch: {steps_batch}")
        coroutines = [
            safe_execute_step(
                step=step,
                workflow=workflow,
                runtime_parameters=runtime_parameters,
                execution_cache=execution_cache,
                step_execution_mode=step_execution_mode,
            )
            for step in steps_batch
        ]
        results = await asyncio.gather(*coroutines)
        for result in results:
            nodes_to_discard.update(result)
    return nodes_to_discard


async def safe_execute_step(
    step: str,
    workflow: CompiledWorkflow,
    runtime_parameters: Dict[str, Any],
    execution_cache: ExecutionCache,
    step_execution_mode: StepExecutionMode,
) -> Set[str]:
    try:
        return await execute_step(
            step=step,
            workflow=workflow,
            runtime_parameters=runtime_parameters,
            execution_cache=execution_cache,
            step_execution_mode=step_execution_mode,
        )
    except Exception as error:
        logger.exception(f"Execution of step {step} encountered error.")
        raise ExecutionEngineError(
            f"Error during execution of step: {step}."
        ) from error


async def execute_step(
    step: str,
    workflow: CompiledWorkflow,
    runtime_parameters: Dict[str, Any],
    execution_cache: ExecutionCache,
    step_execution_mode: StepExecutionMode,
) -> Set[str]:
    logger.info(f"started execution of: {step} - {datetime.now().isoformat()}")
    step_name = get_last_chunk_of_selector(selector=step)
    step_instance = workflow.steps[step_name].step
    step_manifest = workflow.steps[step_name].manifest
    step_outputs = step_instance.get_actual_outputs(step_manifest)
    execution_cache.register_step(
        step_name=step_name,
        output_definitions=step_outputs,
    )
    step_parameters = assembly_step_parameters(
        step_manifest=step_manifest,
        runtime_parameters=runtime_parameters,
        execution_cache=execution_cache,
        accepts_batch_input=step_instance.accepts_batch_input(),
    )
    step_run_method = (
        step_instance.run_locally
        if step_execution_mode is StepExecutionMode.LOCAL
        else step_instance.run_remotely
    )
    step_result = await step_run_method(**step_parameters)
    if issubclass(type(step_result), tuple):
        step_outputs, flow_control = step_result
    else:
        step_outputs, flow_control = step_result, FlowControl(mode="pass")
    execution_cache.register_step_outputs(
        step_name=step_name,
        outputs=step_outputs,
    )
    nodes_to_discard = handle_flow_control(
        current_step_selector=step,
        flow_control=flow_control,
        execution_graph=workflow.execution_graph,
    )
    logger.info(f"finished execution of: {step} - {datetime.now().isoformat()}")
    return nodes_to_discard
