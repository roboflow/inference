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
    run_barcode_detection_step,
    run_clip_comparison_step,
    run_lmm_for_classification_step,
    run_lmm_step,
    run_ocr_model_step,
    run_qr_code_detection_step,
    run_roboflow_model_step,
    run_yolo_world_model_step,
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
    "YoloWorld": run_yolo_world_model_step,
    "LMM": run_lmm_step,
    "LMMForClassification": run_lmm_for_classification_step,
    "QRCodeDetection": run_qr_code_detection_step,
    "BarcodeDetection": run_barcode_detection_step,
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
        logger.exception(f"Execution of step {step} encountered error.")
        raise ExecutionEngineError(
            f"Error during execution of step: {step}."
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
        node_selector = node_definition.selector
        step_selector = get_step_selector_from_its_output(
            step_output_selector=node_selector
        )
        step_field = get_last_selector_chunk(selector=node_selector)
        step_result = outputs_lookup.get(step_selector)
        if step_result is not None:
            if issubclass(type(step_result), list):
                step_result = extract_step_result_from_list(
                    result=step_result,
                    step_field=step_field,
                    coordinates_system=node_definition.coordinates_system,
                    step_selector=step_selector,
                )
            else:
                step_result = extract_step_result_from_dict(
                    result=step_result,
                    step_field=step_field,
                    coordinates_system=node_definition.coordinates_system,
                    step_selector=step_selector,
                )
        result[execution_graph.nodes[node]["definition"].name] = step_result
    return result


def extract_step_result_from_list(
    result: List[Dict[str, Any]],
    step_field: str,
    coordinates_system: CoordinatesSystem,
    step_selector: str,
) -> List[Any]:
    return [
        extract_step_result_from_dict(
            result=element,
            step_field=step_field,
            coordinates_system=coordinates_system,
            step_selector=step_selector,
        )
        for element in result
    ]


def extract_step_result_from_dict(
    result: Dict[str, Any],
    step_field: str,
    coordinates_system: CoordinatesSystem,
    step_selector: str,
) -> Any:
    if step_field == "*":
        return extract_step_result_from_dict_using_wildcard(
            result=result,
            coordinates_system=coordinates_system,
        )
    key_in_parents_coordinates = get_key_in_parents_coordinates(key=step_field)
    if (
        coordinates_system is CoordinatesSystem.PARENT
        and key_in_parents_coordinates in result
    ):
        step_field = key_in_parents_coordinates
    step_result = result.get(step_field)
    if step_result is None:
        raise WorkflowsCompilerRuntimeError(
            f"Cannot find field {step_field} in result of step {step_selector}"
        )
    return step_result


def extract_step_result_from_dict_using_wildcard(
    result: Dict[str, Any],
    coordinates_system: CoordinatesSystem,
) -> Dict[str, Any]:
    all_keys_without_parent_suffix = {
        key for key in result.keys() if not key.endswith(PARENT_COORDINATES_SUFFIX)
    }
    keys_to_be_extracted = {
        (
            (key, key)
            if (
                get_key_in_parents_coordinates(key=key) not in result
                or coordinates_system is CoordinatesSystem.OWN
            )
            else (key, get_key_in_parents_coordinates(key=key))
        )
        for key in all_keys_without_parent_suffix
    }
    return {
        key_alias: result[result_key] for key_alias, result_key in keys_to_be_extracted
    }


def get_key_in_parents_coordinates(key: str) -> str:
    return f"{key}{PARENT_COORDINATES_SUFFIX}"
