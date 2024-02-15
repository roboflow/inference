import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

import networkx as nx
from fastapi import BackgroundTasks
from networkx import DiGraph

from inference.core import logger
from inference.core.managers.base import ModelManager
from inference.enterprise.workflows.complier.entities import StepExecutionMode
from inference.enterprise.workflows.complier.flow_coordinator import (
    ParallelStepExecutionCoordinator,
    SerialExecutionCoordinator,
)
from inference.enterprise.workflows.complier.runtime_input_validator import (
    prepare_runtime_parameters,
)
from inference.enterprise.workflows.complier.steps_executors.active_learning_middlewares import (
    WorkflowsActiveLearningMiddleware,
)
from inference.enterprise.workflows.complier.steps_executors.auxiliary import (
    run_active_learning_data_collector,
    run_condition_step,
    run_crop_step,
    run_detection_filter,
    run_detection_offset_step,
    run_detections_consensus_step,
    run_static_crop_step,
)
from inference.enterprise.workflows.complier.steps_executors.constants import (
    PARENT_COORDINATES_SUFFIX,
)
from inference.enterprise.workflows.complier.steps_executors.models import (
    run_clip_comparison_step,
    run_ocr_model_step,
    run_roboflow_model_step,
)
from inference.enterprise.workflows.complier.steps_executors.types import OutputsLookup
from inference.enterprise.workflows.complier.steps_executors.utils import make_batches
from inference.enterprise.workflows.complier.utils import (
    get_nodes_of_specific_kind,
    get_step_selector_from_its_output,
    is_condition_step,
)
from inference.enterprise.workflows.constants import OUTPUT_NODE_KIND
from inference.enterprise.workflows.entities.outputs import CoordinatesSystem
from inference.enterprise.workflows.entities.validators import get_last_selector_chunk
from inference.enterprise.workflows.errors import (
    ExecutionEngineError,
    WorkflowsCompilerRuntimeError,
)

STEP_TYPE2EXECUTOR_MAPPING = {
    "ClassificationModel": run_roboflow_model_step,
    "MultiLabelClassificationModel": run_roboflow_model_step,
    "ObjectDetectionModel": run_roboflow_model_step,
    "KeypointsDetectionModel": run_roboflow_model_step,
    "InstanceSegmentationModel": run_roboflow_model_step,
    "OCRModel": run_ocr_model_step,
    "Crop": run_crop_step,
    "Condition": run_condition_step,
    "DetectionFilter": run_detection_filter,
    "DetectionOffset": run_detection_offset_step,
    "AbsoluteStaticCrop": run_static_crop_step,
    "RelativeStaticCrop": run_static_crop_step,
    "ClipComparison": run_clip_comparison_step,
    "DetectionsConsensus": run_detections_consensus_step,
    "ActiveLearningDataCollector": run_active_learning_data_collector,
}


async def execute_graph(
    execution_graph: DiGraph,
    runtime_parameters: Dict[str, Any],
    model_manager: ModelManager,
    active_learning_middleware: WorkflowsActiveLearningMiddleware,
    background_tasks: Optional[BackgroundTasks] = None,
    api_key: Optional[str] = None,
    max_concurrent_steps: int = 1,
    step_execution_mode: StepExecutionMode = StepExecutionMode.LOCAL,
) -> dict:
    runtime_parameters = prepare_runtime_parameters(
        execution_graph=execution_graph,
        runtime_parameters=runtime_parameters,
    )
    outputs_lookup = {}
    steps_to_discard = set()
    if max_concurrent_steps > 1:
        execution_coordinator = ParallelStepExecutionCoordinator.init(
            execution_graph=execution_graph
        )
    else:
        execution_coordinator = SerialExecutionCoordinator.init(
            execution_graph=execution_graph
        )
    while True:
        next_steps = execution_coordinator.get_steps_to_execute_next(
            steps_to_discard=steps_to_discard
        )
        if next_steps is None:
            break
        steps_to_discard = await execute_steps(
            steps=next_steps,
            max_concurrent_steps=max_concurrent_steps,
            execution_graph=execution_graph,
            runtime_parameters=runtime_parameters,
            outputs_lookup=outputs_lookup,
            model_manager=model_manager,
            api_key=api_key,
            step_execution_mode=step_execution_mode,
            active_learning_middleware=active_learning_middleware,
            background_tasks=background_tasks,
        )
    return construct_response(
        execution_graph=execution_graph, outputs_lookup=outputs_lookup
    )


async def execute_steps(
    steps: List[str],
    max_concurrent_steps: int,
    execution_graph: DiGraph,
    runtime_parameters: Dict[str, Any],
    outputs_lookup: OutputsLookup,
    model_manager: ModelManager,
    api_key: Optional[str],
    step_execution_mode: StepExecutionMode,
    active_learning_middleware: WorkflowsActiveLearningMiddleware,
    background_tasks: Optional[BackgroundTasks],
) -> Set[str]:
    """outputs_lookup is mutated while execution, only independent steps may be run together"""
    logger.info(f"Executing steps: {steps}. Execution mode: {step_execution_mode}")
    nodes_to_discard = set()
    steps_batches = list(make_batches(iterable=steps, batch_size=max_concurrent_steps))
    for steps_batch in steps_batches:
        logger.info(f"Steps batch: {steps_batch}")
        coroutines = [
            safe_execute_step(
                step=step,
                execution_graph=execution_graph,
                runtime_parameters=runtime_parameters,
                outputs_lookup=outputs_lookup,
                model_manager=model_manager,
                api_key=api_key,
                step_execution_mode=step_execution_mode,
                active_learning_middleware=active_learning_middleware,
                background_tasks=background_tasks,
            )
            for step in steps_batch
        ]
        results = await asyncio.gather(*coroutines)
        for result in results:
            nodes_to_discard.update(result)
    return nodes_to_discard


async def safe_execute_step(
    step: str,
    execution_graph: DiGraph,
    runtime_parameters: Dict[str, Any],
    outputs_lookup: OutputsLookup,
    model_manager: ModelManager,
    api_key: Optional[str],
    step_execution_mode: StepExecutionMode,
    active_learning_middleware: WorkflowsActiveLearningMiddleware,
    background_tasks: Optional[BackgroundTasks],
) -> Set[str]:
    try:
        return await execute_step(
            step=step,
            execution_graph=execution_graph,
            runtime_parameters=runtime_parameters,
            outputs_lookup=outputs_lookup,
            model_manager=model_manager,
            api_key=api_key,
            step_execution_mode=step_execution_mode,
            active_learning_middleware=active_learning_middleware,
            background_tasks=background_tasks,
        )
    except Exception as error:
        raise ExecutionEngineError(
            f"Error during execution of step: {step}. "
            f"Type of error: {type(error).__name__}. "
            f"Cause: {error}"
        ) from error


async def execute_step(
    step: str,
    execution_graph: DiGraph,
    runtime_parameters: Dict[str, Any],
    outputs_lookup: OutputsLookup,
    model_manager: ModelManager,
    api_key: Optional[str],
    step_execution_mode: StepExecutionMode,
    active_learning_middleware: WorkflowsActiveLearningMiddleware,
    background_tasks: Optional[BackgroundTasks],
) -> Set[str]:
    logger.info(f"started execution of: {step} - {datetime.now().isoformat()}")
    nodes_to_discard = set()
    step_definition = execution_graph.nodes[step]["definition"]
    executor = STEP_TYPE2EXECUTOR_MAPPING[step_definition.type]
    additional_args = {}
    if step_definition.type == "ActiveLearningDataCollector":
        additional_args["active_learning_middleware"] = active_learning_middleware
        additional_args["background_tasks"] = background_tasks
    next_step, outputs_lookup = await executor(
        step=step_definition,
        runtime_parameters=runtime_parameters,
        outputs_lookup=outputs_lookup,
        model_manager=model_manager,
        api_key=api_key,
        step_execution_mode=step_execution_mode,
        **additional_args,
    )
    if is_condition_step(execution_graph=execution_graph, node=step):
        if execution_graph.nodes[step]["definition"].step_if_true == next_step:
            nodes_to_discard = get_all_nodes_in_execution_path(
                execution_graph=execution_graph,
                source=execution_graph.nodes[step]["definition"].step_if_false,
            )
        else:
            nodes_to_discard = get_all_nodes_in_execution_path(
                execution_graph=execution_graph,
                source=execution_graph.nodes[step]["definition"].step_if_true,
            )
    logger.info(f"finished execution of: {step} - {datetime.now().isoformat()}")
    return nodes_to_discard


def get_all_nodes_in_execution_path(
    execution_graph: DiGraph,
    source: str,
) -> Set[str]:
    nodes = set(nx.descendants(execution_graph, source))
    nodes.add(source)
    return nodes


def construct_response(
    execution_graph: nx.DiGraph,
    outputs_lookup: Dict[str, Any],
) -> Dict[str, Any]:
    output_nodes = get_nodes_of_specific_kind(
        execution_graph=execution_graph, kind=OUTPUT_NODE_KIND
    )
    result = {}
    for node in output_nodes:
        node_definition = execution_graph.nodes[node]["definition"]
        fallback_selector = None
        node_selector = node_definition.selector
        if node_definition.coordinates_system is CoordinatesSystem.PARENT:
            fallback_selector = node_selector
            node_selector = f"{node_selector}{PARENT_COORDINATES_SUFFIX}"
        step_selector = get_step_selector_from_its_output(
            step_output_selector=node_selector
        )
        step_field = get_last_selector_chunk(selector=node_selector)
        fallback_step_field = (
            None
            if fallback_selector is None
            else get_last_selector_chunk(selector=fallback_selector)
        )
        step_result = outputs_lookup.get(step_selector)
        if step_result is not None:
            if issubclass(type(step_result), list):
                step_result = extract_step_result_from_list(
                    result=step_result,
                    step_field=step_field,
                    fallback_step_field=fallback_step_field,
                    step_selector=step_selector,
                )
            else:
                step_result = extract_step_result_from_dict(
                    result=step_result,
                    step_field=step_field,
                    fallback_step_field=fallback_step_field,
                    step_selector=step_selector,
                )
        result[execution_graph.nodes[node]["definition"].name] = step_result
    return result


def extract_step_result_from_list(
    result: List[Dict[str, Any]],
    step_field: str,
    fallback_step_field: Optional[str],
    step_selector: str,
) -> List[Any]:
    return [
        extract_step_result_from_dict(
            result=element,
            step_field=step_field,
            fallback_step_field=fallback_step_field,
            step_selector=step_selector,
        )
        for element in result
    ]


def extract_step_result_from_dict(
    result: Dict[str, Any],
    step_field: str,
    fallback_step_field: Optional[str],
    step_selector: str,
) -> Any:
    step_result = result.get(step_field, result.get(fallback_step_field))
    if step_result is None:
        raise WorkflowsCompilerRuntimeError(
            f"Cannot find neither field {step_field} nor {fallback_step_field} in result of step {step_selector}"
        )
    return step_result
