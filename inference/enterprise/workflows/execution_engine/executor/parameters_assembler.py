from typing import Any, Dict

from inference.enterprise.workflows.complier.utils import (
    get_last_chunk_of_selector,
    is_input_selector,
    is_step_output_selector,
)
from inference.enterprise.workflows.execution_engine.executor.execution_cache import (
    ExecutionCache,
)
from inference.enterprise.workflows.prototypes.block import WorkflowBlockManifest


def assembly_step_parameters(
    step_manifest: WorkflowBlockManifest,
    runtime_parameters: Dict[str, Any],
    execution_cache: ExecutionCache,
    accepts_batch_input: bool,
) -> Dict[str, Any]:
    manifest_dict = step_manifest.dict(exclude={"type", "name"})
    result = {}
    for key, value in manifest_dict.items():
        if is_step_output_selector(selector_or_value=value):
            value = retrieve_step_output(
                selector=value,
                execution_cache=execution_cache,
                accepts_batch_input=accepts_batch_input,
            )
        elif is_input_selector(selector_or_value=value):
            value = retrieve_value_from_runtime_input(
                selector=value, runtime_parameters=runtime_parameters
            )
        result[key] = value
    return result


def retrieve_step_output(
    selector: str, execution_cache: ExecutionCache, accepts_batch_input: bool
) -> Any:
    value = execution_cache.get_output(selector=selector)
    if accepts_batch_input:
        return value
    if issubclass(type(value), list):
        if len(value) > 1:
            raise RuntimeError(
                f"{selector} points to batch input which is not accepted"
            )
        return value[0]
    return value


def retrieve_value_from_runtime_input(
    selector: str,
    runtime_parameters: Dict[str, Any],
) -> Any:
    try:
        parameter_name = get_last_chunk_of_selector(selector=selector)
        return runtime_parameters[parameter_name]
    except KeyError as e:
        raise e
