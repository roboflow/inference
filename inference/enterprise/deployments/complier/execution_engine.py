import random
from typing import Any, Dict, Optional, Set, Tuple

import networkx as nx
from networkx import DiGraph

from inference.core.managers.base import ModelManager
from inference.enterprise.deployments.complier.runtime_input_validator import (
    validate_runtime_input,
)
from inference.enterprise.deployments.complier.utils import (
    construct_step_selector,
    get_nodes_of_specific_kind,
    get_selector_chunks,
    is_condition_step,
    is_crop_step,
    is_cv_model_step,
    is_selector,
)
from inference.enterprise.deployments.constants import OUTPUT_NODE_KIND, STEP_NODE_KIND
from inference.enterprise.deployments.entities.steps import (
    Condition,
    ConditionSpecs,
    Crop,
    CVModel,
    Operator,
)

OPERATORS = {
    Operator.EQUAL: lambda a, b: a == b,
    Operator.NOT_EQUAL: lambda a, b: a != b,
    Operator.LOWER_THAN: lambda a, b: a < b,
    Operator.GREATER_THAN: lambda a, b: a > b,
    Operator.LOWER_OR_EQUAL_THAN: lambda a, b: a <= b,
    Operator.GREATER_OR_EQUAL_THAN: lambda a, b: a >= b,
}


def execute_graph(
    execution_graph: DiGraph,
    runtime_parameters: Dict[str, Any],
    model_manager: ModelManager,
    api_key: Optional[str] = None,
) -> dict:
    validate_runtime_input(
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
    print(f"Base topological order of execution: {steps_topological_order}")
    nodes_excluded_by_conditional_execution = set()
    for step in steps_topological_order:
        if step in nodes_excluded_by_conditional_execution:
            continue
        if is_cv_model_step(execution_graph=execution_graph, node=step):
            outputs_lookup = execute_cv_model_step(
                step=execution_graph.nodes[step]["definition"],
                model_manager=model_manager,
                runtime_parameters=runtime_parameters,
                outputs_lookup=outputs_lookup,
            )
        elif is_crop_step(execution_graph=execution_graph, node=step):
            outputs_lookup = execute_crop_step(
                step=execution_graph.nodes[step]["definition"],
                model_manager=model_manager,
                runtime_parameters=runtime_parameters,
                outputs_lookup=outputs_lookup,
            )
        elif is_condition_step(execution_graph=execution_graph, node=step):
            next_step, outputs_lookup = execute_condition_step_step(
                step=execution_graph.nodes[step]["definition"],
                model_manager=model_manager,
                runtime_parameters=runtime_parameters,
                outputs_lookup=outputs_lookup,
            )
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
            print(
                f"As a result of condition evaluation in step {step} - discarding: {nodes_to_discard}"
            )
            nodes_excluded_by_conditional_execution.update(nodes_to_discard)
        else:
            raise NotImplementedError(f"Unknown step type for step: {step}")
    output_nodes = get_nodes_of_specific_kind(
        execution_graph=execution_graph, kind=OUTPUT_NODE_KIND
    )
    result = {}
    print(outputs_lookup)
    for node in output_nodes:
        node_selector = execution_graph.nodes[node]["definition"].selector
        result[execution_graph.nodes[node]["definition"].name] = outputs_lookup.get(
            node_selector
        )
    return result


def get_all_nodes_in_execution_path(
    execution_graph: DiGraph,
    source: str,
) -> Set[str]:
    nodes = set(execution_graph.successors(source))
    nodes.add(source)
    return nodes


def execute_cv_model_step(
    step: CVModel,
    model_manager: ModelManager,
    runtime_parameters: Dict[str, Any],
    outputs_lookup: Dict[str, Any],
) -> Dict[str, Any]:
    step_selector = construct_step_selector(step_name=step.name)
    print(f"Executing CVModel step: {step_selector}")
    outputs_lookup[f"{step_selector}.top"] = "cat"
    outputs_lookup[f"{step_selector}.predictions"] = {
        "predictions": step.inputs["model_id"]
    }
    return outputs_lookup


def execute_crop_step(
    step: Crop,
    model_manager: ModelManager,
    runtime_parameters: Dict[str, Any],
    outputs_lookup: Dict[str, Any],
) -> Dict[str, Any]:
    step_selector = construct_step_selector(step_name=step.name)
    print(f"Executing Crop step: {step_selector}")
    outputs_lookup[f"{step_selector}.predictions"] = {"predictions": "predictions_crop"}
    outputs_lookup[f"{step_selector}.crops"] = ["list", "of", "crops"]
    return outputs_lookup


def execute_condition_step_step(
    step: Condition,
    model_manager: ModelManager,
    runtime_parameters: Dict[str, Any],
    outputs_lookup: Dict[str, Any],
) -> Tuple[str, Dict[str, Any]]:
    step_selector = construct_step_selector(step_name=step.name)
    print(f"Executing Condition step: {step_selector}")
    # evaluation_result = evaluate_condition(
    #     condition=step.condition,
    #     runtime_parameters=runtime_parameters,
    #     outputs_lookup=outputs_lookup,
    # )
    evaluation_result = random.random() > 0.5
    next_step = step.step_if_true if evaluation_result else step.step_if_false
    return next_step, outputs_lookup


def evaluate_condition(
    condition: ConditionSpecs,
    runtime_parameters: Dict[str, Any],
    outputs_lookup: Dict[str, Any],
) -> bool:
    left_value = condition.left
    if is_selector(left_value):
        left_value = resolve_selector(
            selector=left_value,
            runtime_parameters=runtime_parameters,
            outputs_lookup=outputs_lookup,
        )
    right_value = condition.right
    if is_selector(right_value):
        right_value = resolve_selector(
            selector=right_value,
            runtime_parameters=runtime_parameters,
            outputs_lookup=outputs_lookup,
        )
    return OPERATORS[condition.operator](left_value, right_value)


def resolve_selector(
    selector: str,
    runtime_parameters: Dict[str, Any],
    outputs_lookup: Dict[str, Any],
) -> Any:
    if selector.startswith("$inputs"):
        selector_chunks = get_selector_chunks(selector=selector)
        return runtime_parameters[selector_chunks[0]]
    if selector.startswith("$steps"):
        return outputs_lookup[selector]
    raise NotImplementedError(f"Not implemented resumption for selector {selector}")
