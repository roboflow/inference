from typing import Any, Dict, Optional, Set

import networkx as nx
from networkx import DiGraph

from inference.core.managers.base import ModelManager
from inference.enterprise.deployments.complier.runtime_input_validator import (
    prepare_runtime_parameters,
)
from inference.enterprise.deployments.complier.steps_executors.auxiliary import (
    run_condition_step,
    run_crop_step,
    run_detection_filter,
    run_detection_offset_step,
    run_static_crop_step,
)
from inference.enterprise.deployments.complier.steps_executors.models import (
    run_clip_comparison_step,
    run_ocr_model_step,
    run_roboflow_model_step,
)
from inference.enterprise.deployments.complier.utils import (
    get_nodes_of_specific_kind,
    get_step_selector_from_its_output,
    is_condition_step,
    is_step_output_selector,
)
from inference.enterprise.deployments.constants import OUTPUT_NODE_KIND, STEP_NODE_KIND
from inference.enterprise.deployments.entities.steps import get_last_selector_chunk

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
}


async def execute_graph(
    execution_graph: DiGraph,
    runtime_parameters: Dict[str, Any],
    model_manager: ModelManager,
    api_key: Optional[str] = None,
) -> dict:
    runtime_parameters = prepare_runtime_parameters(
        execution_graph=execution_graph,
        runtime_parameters=runtime_parameters,
    )
    outputs_lookup = {}
    step_nodes = get_nodes_of_specific_kind(
        execution_graph=execution_graph, kind=STEP_NODE_KIND
    )
    steps_topological_order = [
        n for n in nx.topological_sort(execution_graph) if n in step_nodes
    ]
    nodes_excluded_by_conditional_execution = set()
    for step in steps_topological_order:
        if step in nodes_excluded_by_conditional_execution:
            continue
        step_definition = execution_graph.nodes[step]["definition"]
        executor = STEP_TYPE2EXECUTOR_MAPPING[step_definition.type]
        next_step, outputs_lookup = await executor(
            step=step_definition,
            runtime_parameters=runtime_parameters,
            outputs_lookup=outputs_lookup,
            model_manager=model_manager,
            api_key=api_key,
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
            nodes_excluded_by_conditional_execution.update(nodes_to_discard)
    output_nodes = get_nodes_of_specific_kind(
        execution_graph=execution_graph, kind=OUTPUT_NODE_KIND
    )
    result = {}
    for node in output_nodes:
        node_selector = execution_graph.nodes[node]["definition"].selector
        if not is_step_output_selector(selector_or_value=node_selector):
            raise RuntimeError("TODO CHECK IF OUTPUTS ARE DEFINED ONLY AMONG STEPS!")
        step_selector = get_step_selector_from_its_output(
            step_output_selector=node_selector
        )
        step_field = get_last_selector_chunk(selector=node_selector)
        step_result = outputs_lookup.get(step_selector)
        if step_result is not None:
            if issubclass(type(step_result), list):
                step_result = [e[step_field] for e in step_result]
            else:
                step_result = step_result[step_field]
        result[execution_graph.nodes[node]["definition"].name] = step_result
    return result


def get_all_nodes_in_execution_path(
    execution_graph: DiGraph,
    source: str,
) -> Set[str]:
    nodes = set(execution_graph.successors(source))
    nodes.add(source)
    return nodes
