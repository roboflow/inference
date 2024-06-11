import asyncio
from datetime import datetime
from typing import Any, Dict, List, Set

from inference.core import logger
from inference.core.workflows.entities.types import FlowControl
from inference.core.workflows.errors import StepExecutionError, WorkflowError
from inference.core.workflows.execution_engine.compiler.entities import CompiledWorkflow
from inference.core.workflows.execution_engine.compiler.utils import (
    get_last_chunk_of_selector,
    is_flow_control_step,
)
from inference.core.workflows.execution_engine.executor.flow_coordinator import (
    ParallelStepExecutionCoordinator,
    handle_flow_control,
)
from inference.core.workflows.execution_engine.executor.new_execution_cache import (
    DynamicBatchesManager,
    ExecutionBranchesManager,
    ExecutionCache,
)
from inference.core.workflows.execution_engine.executor.output_constructor import (
    construct_workflow_output,
)
from inference.core.workflows.execution_engine.executor.parameters_assembler import (
    assembly_step_parameters,
)
from inference_sdk.http.utils.iterables import make_batches


async def run_workflow(
    workflow: CompiledWorkflow,
    runtime_parameters: Dict[str, Any],
    max_concurrent_steps: int,
) -> Dict[str, List[Any]]:
    execution_cache = ExecutionCache.init()
    branches_manager = ExecutionBranchesManager.init(
        workflow_inputs=workflow.workflow_definition.inputs,
        runtime_parameters=runtime_parameters,
    )
    dynamic_batches_manager = DynamicBatchesManager.init(
        workflow_inputs=workflow.workflow_definition.inputs,
        runtime_parameters=runtime_parameters,
    )
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
            branches_manager=branches_manager,
            dynamic_batches_manager=dynamic_batches_manager,
        )
        next_steps = execution_coordinator.get_steps_to_execute_next(
            steps_to_discard=steps_to_discard
        )
    return construct_workflow_output(
        workflow_outputs=workflow.workflow_definition.outputs,
        execution_cache=execution_cache,
        runtime_parameters=runtime_parameters,
    )


async def execute_steps(
    next_steps: List[str],
    workflow: CompiledWorkflow,
    runtime_parameters: Dict[str, Any],
    execution_cache: ExecutionCache,
    max_concurrent_steps: int,
    branches_manager: ExecutionBranchesManager,
    dynamic_batches_manager: DynamicBatchesManager,
) -> Set[str]:
    logger.info(f"Executing steps: {next_steps}.")
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
                branches_manager=branches_manager,
                dynamic_batches_manager=dynamic_batches_manager,
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
    branches_manager: ExecutionBranchesManager,
    dynamic_batches_manager: DynamicBatchesManager,
) -> Set[str]:
    try:
        return await execute_step(
            step=step,
            workflow=workflow,
            runtime_parameters=runtime_parameters,
            execution_cache=execution_cache,
            branches_manager=branches_manager,
            dynamic_batches_manager=dynamic_batches_manager,
        )
    except WorkflowError as error:
        raise error
    except Exception as error:
        logger.exception(f"Execution of step {step} encountered error.")
        raise StepExecutionError(
            public_message=f"Error during execution of step: {step}. Details: {error}",
            context="workflow_execution | step_execution",
            inner_error=error,
        ) from error


async def execute_step(
    step: str,
    workflow: CompiledWorkflow,
    runtime_parameters: Dict[str, Any],
    execution_cache: ExecutionCache,
    branches_manager: ExecutionBranchesManager,
    dynamic_batches_manager: DynamicBatchesManager,
) -> Set[str]:
    logger.info(f"started execution of: {step} - {datetime.now().isoformat()}")
    step_name = get_last_chunk_of_selector(selector=step)
    step_instance = workflow.steps[step_name].step
    step_manifest = workflow.steps[step_name].manifest
    execution_cache.register_step(
        step_name=step_name,
        compatible_with_batches=step_instance.produces_batch_output(),
    )
    print(workflow.execution_graph.nodes[step])
    print(
        "branch_mask",
        branches_manager.retrieve_branch_mask(
            branch_name=workflow.execution_graph.nodes[step][
                "execution_branches_stack"
            ][-1]
        ),
    )
    print(
        "batch_element_indices",
        dynamic_batches_manager.get_batch_element_indices(
            data_lineage=workflow.execution_graph.nodes[step]["dimensionality_lineage"][
                -1
            ]
        ),
    )
    step_parameters = assembly_step_parameters(
        step_manifest=step_manifest,
        runtime_parameters=runtime_parameters,
        execution_cache=execution_cache,
        accepts_batch_input=step_instance.accepts_batch_input(),
    )
    step_result = await step_instance.run(**step_parameters)
    if is_flow_control_step(execution_graph=workflow.execution_graph, node=step):
        nodes_to_discard = handle_flow_control(
            current_step_selector=step,
            flow_control=step_result,
            execution_graph=workflow.execution_graph,
        )
    else:
        execution_cache.register_step_outputs(
            step_name=step_name,
            indices=dynamic_batches_manager.get_batch_element_indices(
                data_lineage=workflow.execution_graph.nodes[step][
                    "dimensionality_lineage"
                ][-1]
            ),
            outputs=step_result,
        )
        nodes_to_discard = set()
    logger.info(f"finished execution of: {step} - {datetime.now().isoformat()}")
    return nodes_to_discard
