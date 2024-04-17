from typing import Any, Dict

from inference.enterprise.workflows.errors import ExecutionEngineNotImplementedError
from inference.enterprise.workflows.execution_engine.compiler.utils import (
    get_last_chunk_of_selector,
    is_input_selector,
    is_step_output_selector,
)
from inference.enterprise.workflows.execution_engine.executor.execution_cache import (
    ExecutionCache,
)
from inference.enterprise.workflows.prototypes.block import WorkflowBlockManifest

EXCLUDED_FIELDS = {"type", "name"}


def assembly_step_parameters(
    step_manifest: WorkflowBlockManifest,
    runtime_parameters: Dict[str, Any],
    execution_cache: ExecutionCache,
    accepts_batch_input: bool,
) -> Dict[str, Any]:
    manifest_dict = get_manifest_fields_values(step_manifest=step_manifest)
    result = {}
    for key, value in manifest_dict.items():
        if is_step_output_selector(selector_or_value=value):
            value = retrieve_step_output(
                selector=value,
                execution_cache=execution_cache,
                accepts_batch_input=accepts_batch_input,
                step_name=step_manifest.name,
            )
        elif is_input_selector(selector_or_value=value):
            value = retrieve_value_from_runtime_input(
                selector=value, runtime_parameters=runtime_parameters
            )
        result[key] = value
    return result


def get_manifest_fields_values(step_manifest: WorkflowBlockManifest) -> Dict[str, Any]:
    result = {}
    for field in step_manifest.__fields__:
        if field in EXCLUDED_FIELDS:
            continue
        result[field] = getattr(step_manifest, field)
    return result


def retrieve_step_output(
    selector: str,
    execution_cache: ExecutionCache,
    accepts_batch_input: bool,
    step_name: str,
) -> Any:
    value = execution_cache.get_output(selector=selector)
    if not execution_cache.output_represent_batch(selector=selector):
        value = value[0]
    if accepts_batch_input:
        return value
    if issubclass(type(value), list):
        if len(value) > 1:
            raise ExecutionEngineNotImplementedError(
                public_message=f"Step `{step_name}` defines input pointing to {selector} which "
                f"ships batch input of size larger than one, but at the same time workflow block "
                f"used to implement the step does not accept batch input. That may be "
                f"for instance the case for steps with flow-control, as workflows execution engine "
                f"does not support yet branching when control-flow decision is made element-wise.",
                context="workflow_execution | steps_parameters_assembling",
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
