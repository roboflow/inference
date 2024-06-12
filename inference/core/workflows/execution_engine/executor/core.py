import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from networkx import DiGraph

from inference.core import logger
from inference.core.workflows.constants import (
    DIMENSIONALITY_LINEAGE_PROPERTY,
    FLOW_CONTROL_EXECUTION_BRANCHES_STACK_PROPERTY,
    STEP_DEFINITION_PROPERTY,
)
from inference.core.workflows.entities.base import Batch, WorkflowImageData
from inference.core.workflows.entities.types import FlowControl
from inference.core.workflows.errors import (
    ExecutionEngineNotImplementedError,
    ExecutionEngineRuntimeError,
    StepExecutionError,
    WorkflowError,
)
from inference.core.workflows.execution_engine.compiler.entities import CompiledWorkflow
from inference.core.workflows.execution_engine.compiler.utils import (
    construct_step_selector,
    get_last_chunk_of_selector,
    get_step_selector_from_its_output,
    is_flow_control_step,
    is_input_selector,
    is_step_output_selector,
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
    get_manifest_fields_values,
)
from inference.core.workflows.prototypes.block import (
    WorkflowBlock,
    WorkflowBlockManifest,
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
    execution_cache.register_step(
        step_name=step_name,
        compatible_with_batches=step_instance.produces_batch_output(),
    )
    step_parameters = assembly_step_execution_parameters(
        step=step,
        workflow=workflow,
        runtime_parameters=runtime_parameters,
        execution_cache=execution_cache,
        branches_manager=branches_manager,
        dynamic_batches_manager=dynamic_batches_manager,
    )
    step_result = await run_step(
        step_instance=step_instance,
        parameters=step_parameters,
        dynamic_batches_manager=dynamic_batches_manager,
        execution_cache=execution_cache,
        step_controls_flow=is_flow_control_step(
            execution_graph=workflow.execution_graph, node=step
        ),
    )
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


def assembly_step_execution_parameters(
    step: str,
    workflow: CompiledWorkflow,
    runtime_parameters: Dict[str, Any],
    execution_cache: ExecutionCache,
    branches_manager: ExecutionBranchesManager,
    dynamic_batches_manager: DynamicBatchesManager,
) -> Dict[str, Union[Batch[Any], Any]]:
    step_name = get_last_chunk_of_selector(selector=step)
    step_manifest = workflow.steps[step_name].manifest
    execution_branches_impacting_step = grab_execution_branches_impacting_step(
        step=step,
        execution_graph=workflow.execution_graph,
    )
    batch_masks = [
        branches_manager.retrieve_branch_mask_for_batch(branch_name=branch)
        for branch in execution_branches_impacting_step
        if branches_manager.is_batch_compatible_branch(branch_name=branch)
    ]
    non_batch_masks = [
        branches_manager.retrieve_branch_mask_for_non_batch(branch_name=branch)
        for branch in execution_branches_impacting_step
        if not branches_manager.is_batch_compatible_branch(branch_name=branch)
    ]
    intersection_of_masks = prepare_intersection_of_masks(
        batch_masks=batch_masks,
        non_batch_masks=set(non_batch_masks),
    )
    return prepare_parameters(
        step_manifest=step_manifest,
        runtime_parameters=runtime_parameters,
        execution_cache=execution_cache,
        intersection_of_masks=intersection_of_masks,
        dynamic_batches_manager=dynamic_batches_manager,
        execution_graph=workflow.execution_graph,
    )


def grab_execution_branches_impacting_step(
    step: str,
    execution_graph: DiGraph,
) -> List[str]:
    predecessors_of_step = execution_graph.predecessors(step)
    flow_control_predecessors = [
        step
        for step in predecessors_of_step
        if is_flow_control_step(execution_graph=execution_graph, node=step)
    ]
    return [
        execution_graph.nodes[predecessor][
            FLOW_CONTROL_EXECUTION_BRANCHES_STACK_PROPERTY
        ][step]
        for predecessor in flow_control_predecessors
    ]


def prepare_intersection_of_masks(
    batch_masks: List[Set[Tuple[int, ...]]],
    non_batch_masks: Set[bool],
) -> Dict[int, Set[Tuple[int, ...]]]:
    masks_dimensions = enumerate_masks_dimensions(batch_masks=batch_masks)
    if False in non_batch_masks:
        return {
            dimension: set() for dimension in masks_dimensions
        }  # empty set means nothing
    return {
        dimension: get_masks_intersection_up_to_dimension(
            batch_masks=batch_masks,
            dimension=dimension,
        )
        for dimension in masks_dimensions
    }


def enumerate_masks_dimensions(batch_masks: List[Set[Tuple[int, ...]]]) -> List[int]:
    dimensions_spotted = set()
    for batch_mask in batch_masks:
        for mask_element in batch_mask:
            dimensions_spotted.add(len(mask_element))
    return sorted(list(dimensions_spotted))


def get_masks_intersection_up_to_dimension(
    batch_masks: List[Set[Tuple[int, ...]]],
    dimension: int,
) -> Set[Tuple[int, ...]]:
    batch_masks_in_dimension = [
        {mask_element[:dimension] for mask_element in batch_mask}
        for batch_mask in batch_masks
    ]
    return set.intersection(*batch_masks_in_dimension)


def prepare_parameters(
    step_manifest: WorkflowBlockManifest,
    runtime_parameters: Dict[str, Any],
    execution_cache: ExecutionCache,
    intersection_of_masks: Dict[int, Set[Tuple[int, ...]]],
    dynamic_batches_manager: DynamicBatchesManager,
    execution_graph: DiGraph,
) -> Dict[str, Any]:
    manifest_dict = get_manifest_fields_values(step_manifest=step_manifest)
    result = {}
    for key, value in manifest_dict.items():
        if isinstance(value, list):
            value = [
                retrieve_value(
                    value=v,
                    step_name=step_manifest.name,
                    runtime_parameters=runtime_parameters,
                    execution_cache=execution_cache,
                    intersection_of_masks=intersection_of_masks,
                    dynamic_batches_manager=dynamic_batches_manager,
                    execution_graph=execution_graph,
                )
                for v in value
            ]
        elif isinstance(value, dict):
            value = {
                k: retrieve_value(
                    value=v,
                    step_name=step_manifest.name,
                    runtime_parameters=runtime_parameters,
                    execution_cache=execution_cache,
                    intersection_of_masks=intersection_of_masks,
                    dynamic_batches_manager=dynamic_batches_manager,
                    execution_graph=execution_graph,
                )
                for k, v in value.items()
            }
        else:
            value = retrieve_value(
                value=value,
                step_name=step_manifest.name,
                runtime_parameters=runtime_parameters,
                execution_cache=execution_cache,
                intersection_of_masks=intersection_of_masks,
                dynamic_batches_manager=dynamic_batches_manager,
                execution_graph=execution_graph,
            )
        result[key] = value
    return result


def retrieve_value(
    value: Any,
    step_name: str,
    runtime_parameters: Dict[str, Any],
    execution_cache: ExecutionCache,
    execution_graph: DiGraph,
    intersection_of_masks: Dict[int, Set[Tuple[int, ...]]],
    dynamic_batches_manager: DynamicBatchesManager,
) -> Any:
    if is_step_output_selector(selector_or_value=value):
        this_step_selector = construct_step_selector(step_name=step_name)
        predecessors = execution_graph.predecessors(this_step_selector)
        predecessors_batch_dims = [
            len(execution_graph.nodes[predecessor][DIMENSIONALITY_LINEAGE_PROPERTY])
            for predecessor in predecessors
            if len(execution_graph.nodes[predecessor][DIMENSIONALITY_LINEAGE_PROPERTY])
            > 0
        ]
        if len(predecessors_batch_dims) > 0:
            requested_dimensionality = min(predecessors_batch_dims)
        else:
            requested_dimensionality = None
        value = retrieve_step_output(
            selector=value,
            step_name=step_name,
            execution_cache=execution_cache,
            execution_graph=execution_graph,
            intersection_of_masks=intersection_of_masks,
            dynamic_batches_manager=dynamic_batches_manager,
            requested_dimensionality=requested_dimensionality,
        )
    elif is_input_selector(selector_or_value=value):
        value = retrieve_value_from_runtime_input(
            selector=value,
            runtime_parameters=runtime_parameters,
            step_name=step_name,
            execution_graph=execution_graph,
            intersection_of_masks=intersection_of_masks,
        )
    return value


def retrieve_step_output(
    selector: str,
    step_name: str,
    execution_cache: ExecutionCache,
    execution_graph: DiGraph,
    intersection_of_masks: Dict[int, Set[Tuple[int, ...]]],
    dynamic_batches_manager: DynamicBatchesManager,
    requested_dimensionality: Optional[int],
) -> Any:
    step_selector = get_step_selector_from_its_output(step_output_selector=selector)
    step_node_data = execution_graph.nodes[step_selector]
    input_dimensionality = len(step_node_data[DIMENSIONALITY_LINEAGE_PROPERTY])
    if input_dimensionality == 0:
        return execution_cache.get_non_batch_output(selector=selector)
    input_mask = None
    if input_dimensionality in intersection_of_masks:
        input_mask = intersection_of_masks[input_dimensionality]
    all_input_indices = dynamic_batches_manager.get_batch_element_indices(
        data_lineage=step_node_data[DIMENSIONALITY_LINEAGE_PROPERTY][-1]
    )
    if (
        requested_dimensionality == input_dimensionality
    ) or requested_dimensionality is None:
        data = execution_cache.get_batch_output(
            selector=selector,
            batch_elements_indices=all_input_indices,
            mask=input_mask,
        )
        return Batch(data, all_input_indices)
    if requested_dimensionality > input_dimensionality:
        raise ValueError(
            f"Requested dimensionality {requested_dimensionality} is higher than selected "
            f"data dimensionality: {input_dimensionality} for step {step_name} and selector {selector}."
        )
    indices_prefixes, aggregated_indices = group_indices_by_dimension(
        batch_indices=all_input_indices,
        dimension=requested_dimensionality,
    )
    result = [
        Batch(
            execution_cache.get_batch_output(
                selector=selector,
                batch_elements_indices=indices_group,
                mask=input_mask,
            ),
            indices_group,
        )
        for indices_group in aggregated_indices
    ]
    return Batch(result, indices_prefixes)


def group_indices_by_dimension(
    batch_indices: List[Tuple[int, ...]],
    dimension: int,
) -> Tuple[List[Tuple[int, ...]], List[List[Tuple[int, ...]]]]:
    ordered_prefixes = []
    grouped_indices = {}
    for batch_index in batch_indices:
        index_prefix = batch_index[:dimension]
        if index_prefix not in grouped_indices:
            ordered_prefixes.append(index_prefix)
            grouped_indices[index_prefix] = []
        grouped_indices[index_prefix].append(batch_index)
    return ordered_prefixes, [
        grouped_indices[index_prefix] for index_prefix in ordered_prefixes
    ]


def retrieve_value_from_runtime_input(
    selector: str,
    runtime_parameters: Dict[str, Any],
    step_name: str,
    execution_graph: DiGraph,
    intersection_of_masks: Dict[int, Set[Tuple[int, ...]]],
) -> Any:
    try:
        parameter_name = get_last_chunk_of_selector(selector=selector)
        value = runtime_parameters[parameter_name]
    except KeyError as e:
        raise ExecutionEngineRuntimeError(
            public_message=f"Attempted to retrieve runtime parameter of step {step_name} using selector {selector} "
            f"discovering miss in runtime parameters. This should have been detected "
            f"by execution engine at the earlier stage. "
            f"Contact Roboflow team through github issues "
            f"(https://github.com/roboflow/inference/issues) providing full context of"
            f"the problem - including workflow definition you use.",
            context="workflow_execution | steps_parameters_assembling",
            inner_error=e,
        ) from e
    if not execution_graph.nodes[selector][
        STEP_DEFINITION_PROPERTY
    ].is_batch_oriented():
        return value
    if not isinstance(value, list):
        raise ValueError(f"Value under {selector} should be batch oriented")
    indices = [(i,) for i in range(len(value))]
    if 1 in intersection_of_masks:
        mask = intersection_of_masks[1]
        value = [v if i in mask else None for i, v in zip(indices, value)]
    return Batch(content=value, indices=indices)


async def run_step(
    step_instance: WorkflowBlock,
    parameters: Dict[str, Any],
    execution_cache: ExecutionCache,
    dynamic_batches_manager: DynamicBatchesManager,
    step_controls_flow: bool,
) -> Optional[Batch[FlowControl]]:
    if step_instance.accepts_batch_input():
        return await run_step_in_batch_mode(
            step_instance=step_instance,
            parameters=parameters,
            execution_cache=execution_cache,
            dynamic_batches_manager=dynamic_batches_manager,
            step_controls_flow=step_controls_flow,
        )
    return await run_step_in_non_batch_mode(
        step_instance=step_instance,
        parameters=parameters,
        execution_cache=execution_cache,
        dynamic_batches_manager=dynamic_batches_manager,
        step_controls_flow=step_controls_flow,
    )


async def run_step_in_batch_mode(
    step_name: str,
    step_instance: WorkflowBlock,
    parameters: Dict[str, Any],
    execution_cache: ExecutionCache,
    dynamic_batches_manager: DynamicBatchesManager,
    step_controls_flow: bool,
) -> Optional[Batch[FlowControl]]:
    if not step_instance.accepts_empty_datapoints():
        non_empty_indices = get_all_non_empty_indices(value=parameters)
        parameters = filter_parameters(
            value=parameters,
            non_empty_indices=non_empty_indices,
        )
    indices_of_parameters = retrieve_indices(
        parameters=parameters,
        reference_parameter=step_instance.get_data_dimensionality_property(),
    )
    results = await step_instance.run(**parameters)
    if step_controls_flow:
        return results
    if not step_instance.produces_batch_output():
        # TODO
        return None
    if step_instance.get_impact_on_data_dimensionality() == "decreases":
        reduced_indices_of_parameters = []
        already_spotted_indices = set()
        for index in indices_of_parameters:
            decreased_index = index[:-1]
            if decreased_index not in already_spotted_indices:
                reduced_indices_of_parameters.append(decreased_index)
                already_spotted_indices.add(decreased_index)
        indices_of_parameters = reduced_indices_of_parameters
    if indices_of_parameters == [()]:
        # we have singular output
        return None
    if len(results) != len(indices_of_parameters):
        raise ValueError("Missmatch in step result dimension")
    if step_instance.get_impact_on_data_dimensionality() == "increases":
        increased_indices, increased_data = [], []
        for index, result in zip(indices_of_parameters, results):
            for nested_idx, result_element in enumerate(result):
                increased_indices.append(index + (nested_idx,))
                increased_data.append(result_element)
        indices_of_parameters = increased_indices
        results = increased_data
        dynamic_batches_manager.register_batch_sizes(data_lineage=..., sizes=...)
    execution_cache.register_batch_of_step_outputs(
        step_name=step_name, indices=indices_of_parameters, outputs=results
    )

    # register results
    return ...


def get_all_non_empty_indices(value: Any) -> Set[Tuple[int, ...]]:
    result = set()
    if isinstance(value, dict):
        for v in value.values():
            value_result = get_all_non_empty_indices(v)
            result = result.union(value_result)
    if isinstance(value, list):
        for v in value:
            value_result = get_all_non_empty_indices(v)
            result = result.union(value_result)
    if isinstance(value, Batch):
        indices = value._indices
        for i, v in zip(indices, value):
            if isinstance(v, Batch):
                value_result = get_all_non_empty_indices(v)
                result = result.union(value_result)
            elif v is not None:
                result = result.union(generate_all_index_prefixes(index=i))
    return result


def generate_all_index_prefixes(index: Tuple[int, ...]) -> Set[Tuple[int, ...]]:
    result = set()
    for i in range(1, len(index) + 1):
        result.add(index[:i])
    return result


def filter_parameters(value: Any, non_empty_indices: Set[Tuple[int, ...]]) -> Any:
    if isinstance(value, dict):
        return {
            k: filter_parameters(value=v, non_empty_indices=non_empty_indices)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [
            filter_parameters(value=v, non_empty_indices=non_empty_indices)
            for v in value
        ]
    if isinstance(value, Batch):
        return value.filter_by_indices(non_empty_indices)
    return value


def retrieve_indices(
    parameters: Dict[str, Any], reference_parameter: Optional[str]
) -> List[Tuple[int, ...]]:
    indices_of_all_parameters = retrieve_indices_of_all_parameters(
        parameters=parameters
    )
    if reference_parameter is not None:
        return indices_of_all_parameters[reference_parameter]
    for name, values in indices_of_all_parameters.items():
        if values:
            return values
    return []


def retrieve_indices_of_all_parameters(
    parameters: Dict[str, Any]
) -> Dict[str, List[Tuple[int, ...]]]:
    return {
        name: retrieve_indices_of_parameter(value=value)
        for name, value in parameters.items()
    }


def retrieve_indices_of_parameter(value: Any) -> List[Tuple[int, ...]]:
    if isinstance(value, dict):
        previous_result = None
        for k, v in value.items():
            current_result = retrieve_indices_of_parameter(v)
            if current_result:
                if previous_result:
                    if previous_result != current_result:
                        raise ValueError("Missmatch in parameters dims")
                else:
                    previous_result = current_result
        if previous_result:
            return previous_result
        return []
    if isinstance(value, list):
        previous_result = None
        for v in value:
            current_result = retrieve_indices_of_parameter(v)
            if current_result:
                if previous_result:
                    if previous_result != current_result:
                        raise ValueError("Missmatch in parameters dims")
                else:
                    previous_result = current_result
        if previous_result:
            return previous_result
        return []
    if isinstance(value, Batch):
        if any(isinstance(v, Batch) for v in value):
            result = []
            for v in value:
                v_result = retrieve_indices_of_parameter(v)
                result.extend(v_result)
            return result
        return value._indices
    return []


async def run_step_in_non_batch_mode(
    step_instance: WorkflowBlock,
    parameters: Dict[str, Any],
    execution_cache: ExecutionCache,
    dynamic_batches_manager: DynamicBatchesManager,
    step_controls_flow: bool,
) -> Optional[Batch[FlowControl]]:
    pass
