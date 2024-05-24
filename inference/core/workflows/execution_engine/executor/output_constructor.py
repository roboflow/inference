from typing import Any, Dict, List

import supervision as sv

from inference.core.workflows.core_steps.common.utils import (
    sv_detections_to_root_coordinates,
)
from inference.core.workflows.entities.base import CoordinatesSystem, JsonField
from inference.core.workflows.execution_engine.compiler.utils import (
    get_last_chunk_of_selector,
    get_step_selector_from_its_output,
)
from inference.core.workflows.execution_engine.executor.execution_cache import (
    ExecutionCache,
)


def construct_workflow_output(
    workflow_outputs: List[JsonField],
    execution_cache: ExecutionCache,
) -> Dict[str, List[Any]]:
    # TODO: figure out if we needed generic "to-root coordinates" transformation?
    # Maybe output constructor should never touch this? Maybe it's better to
    # have steps transforming to specific coordinates systems?
    result = {}
    for node in workflow_outputs:
        step_selector = get_step_selector_from_its_output(
            step_output_selector=node.selector
        )
        step_name = get_last_chunk_of_selector(selector=step_selector)
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
) -> List[Dict[str, Any]]:
    step_outputs = execution_cache.get_all_step_outputs(step_name=step_name)
    result = []
    for element in step_outputs:
        element_result = {}
        for key, value in element.items():
            if isinstance(value, sv.Detections) and use_parents_coordinates:
                element_result[key] = sv_detections_to_root_coordinates(
                    detections=value
                )
            else:
                element_result[key] = value
        result.append(element_result)
    return result


def construct_specific_property_output(
    selector: str,
    execution_cache: ExecutionCache,
    coordinates_system: CoordinatesSystem,
) -> List[Any]:
    cache_contains_selector = execution_cache.is_value_registered(selector=selector)
    if not cache_contains_selector:
        return []
    value = execution_cache.get_output(selector=selector)
    if (
        not is_batch_of_sv_detections(value=value)
        or coordinates_system is CoordinatesSystem.OWN
    ):
        return value
    return [
        sv_detections_to_root_coordinates(detections=v) if v is not None else None
        for v in value
    ]


def is_batch_of_sv_detections(value: Any) -> bool:
    if not isinstance(value, list):
        return False
    return all(isinstance(v, sv.Detections) or v is None for v in value)
