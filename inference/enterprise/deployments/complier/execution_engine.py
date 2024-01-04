from typing import Dict, Any, Optional, Tuple

from networkx import DiGraph

from inference.core.managers.base import ModelManager
from inference.enterprise.deployments.complier.utils import construct_step_selector, is_selector, get_selector_chunks
from inference.enterprise.deployments.entities.steps import CVModel, Crop, Condition, ConditionSpecs, Operator

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
) -> None:
    pass


def execute_cv_model_step(
    step: CVModel,
    model_manager: ModelManager,
    runtime_parameters: Dict[str, Any],
    outputs_lookup: Dict[str, Any],
) -> Dict[str, Any]:
    step_selector = construct_step_selector(step_name=step.name)
    print(f"Executing CVModel step: {step_selector}")
    outputs_lookup[f"{step_selector}.top"] = "cat"
    outputs_lookup[f"{step_selector}.predictions"] = {"predictions": step.inputs["model_id"]}
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
    evaluation_result = evaluate_condition(
        condition=step.condition,
        runtime_parameters=runtime_parameters,
        outputs_lookup=outputs_lookup,
    )
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
