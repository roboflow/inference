from typing import Any, Dict, List, Optional

from inference.enterprise.workflows.complier.steps_executors.constants import (
    PARENT_COORDINATES_SUFFIX,
)
from inference.enterprise.workflows.entities.outputs import CoordinatesSystem
from inference.enterprise.workflows.entities.validators import get_last_selector_chunk
from inference.enterprise.workflows.execution_engine.compiler.entities import (
    ParsedWorkflowDefinition,
)
from inference.enterprise.workflows.execution_engine.executor.execution_cache import (
    ExecutionCache,
)


def construct_workflow_output(
    workflow_definition: ParsedWorkflowDefinition,
    execution_cache: ExecutionCache,
) -> Dict[str, List[Any]]:
    result = {}
    for node in workflow_definition.outputs:
        step_name = get_last_selector_chunk(selector=node.selector)
        cache_contains_step = execution_cache.contains_step(step_name=step_name)
        if not cache_contains_step:
            result[node.name] = []
            continue
        if node.selector.endswith(".*"):
            result[node.name] = construct_wildcard_output(
                step_name=step_name,
                execution_cache=execution_cache,
                use_parents_coordinates=node.coordinates_system
                is CoordinatesSystem.PARENT,
            )
            continue
        result[node.name] = construct_specific_property_output(
            selector=node.selector,
            execution_cache=execution_cache,
            coordinates_system=node.coordinates_system,
        )
    return result


def construct_wildcard_output(
    step_name: str,
    execution_cache: ExecutionCache,
    use_parents_coordinates: bool,
) -> Optional[List[Dict[str, Any]]]:
    step_outputs = execution_cache.get_all_step_outputs(step_name=step_name)
    result = []
    for element in step_outputs:
        element_result = {}
        for key, value in element.items():
            if key.endswith(PARENT_COORDINATES_SUFFIX):
                if use_parents_coordinates:
                    element_result[key[: len(PARENT_COORDINATES_SUFFIX)]] = value
                else:
                    continue
            else:
                if (
                    f"{key}{PARENT_COORDINATES_SUFFIX}" in step_outputs
                    and use_parents_coordinates
                ):
                    continue
                else:
                    element_result[key] = value
        result.append(element_result)
    return result


def construct_specific_property_output(
    selector: str,
    execution_cache: ExecutionCache,
    coordinates_system: CoordinatesSystem,
) -> Optional[List[Any]]:
    cache_contains_selector = execution_cache.contains_value(selector=selector)
    if coordinates_system is CoordinatesSystem.OWN:
        if cache_contains_selector:
            return execution_cache.get_output(selector=selector)
        else:
            return None
    parent_selector = f"{selector}{PARENT_COORDINATES_SUFFIX}"
    if execution_cache.contains_value(selector=parent_selector):
        return execution_cache.get_output(selector=parent_selector)
    if cache_contains_selector:
        return execution_cache.get_output(selector=selector)
    return None
