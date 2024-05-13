from typing import Any, Dict

from inference.core.workflows.errors import (
    ExecutionEngineNotImplementedError,
    ExecutionEngineRuntimeError,
)
from inference.core.workflows.execution_engine.compiler.utils import (
    get_last_chunk_of_selector,
    is_input_selector,
    is_step_output_selector,
)
from inference.core.workflows.execution_engine.executor.execution_cache import (
    ExecutionCache,
)
from inference.core.workflows.prototypes.block import WorkflowBlockManifest

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
        if isinstance(value, list):
            value = [
                retrieve_value(
                    value=v,
                    step_name=step_manifest.name,
                    runtime_parameters=runtime_parameters,
                    execution_cache=execution_cache,
                    accepts_batch_input=accepts_batch_input,
                )
                for v in value
            ]
        else:
            value = retrieve_value(
                value=value,
                step_name=step_manifest.name,
                runtime_parameters=runtime_parameters,
                execution_cache=execution_cache,
                accepts_batch_input=accepts_batch_input,
            )
        result[key] = value
    return result


def retrieve_value(
    value: Any,
    step_name: str,
    runtime_parameters: Dict[str, Any],
    execution_cache: ExecutionCache,
    accepts_batch_input: bool,
) -> Any:
    if is_step_output_selector(selector_or_value=value):
        value = retrieve_step_output(
            selector=value,
            execution_cache=execution_cache,
            accepts_batch_input=accepts_batch_input,
            step_name=step_name,
        )
    elif is_input_selector(selector_or_value=value):
        value = retrieve_value_from_runtime_input(
            selector=value,
            runtime_parameters=runtime_parameters,
            accepts_batch_input=accepts_batch_input,
            step_name=step_name,
        )
    return value


def get_manifest_fields_values(step_manifest: WorkflowBlockManifest) -> Dict[str, Any]:
    result = {}
    for field in step_manifest.model_fields:
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
    if isinstance(value, list):
        if len(value) > 1:
            raise ExecutionEngineNotImplementedError(
                public_message=f"Step `{step_name}` defines input pointing to {selector} which "
                f"ships batch input of size larger than one, but at the same time workflow block "
                f"used to implement the step does not accept batch input. That may be "
                f"for instance the case for steps with flow-control, as workflows execution engine "
                f"does not yet support branching when control-flow decision is made element-wise.",
                context="workflow_execution | steps_parameters_assembling",
            )
        return value[0]
    return value


def retrieve_value_from_runtime_input(
    selector: str,
    runtime_parameters: Dict[str, Any],
    accepts_batch_input: bool,
    step_name: str,
) -> Any:
    try:
        parameter_name = get_last_chunk_of_selector(selector=selector)
        value = runtime_parameters[parameter_name]
        if not _retrieved_inference_image(value=value) or accepts_batch_input:
            return value
        if len(value) > 1:
            raise ExecutionEngineNotImplementedError(
                public_message=f"Step `{step_name}` defines input pointing to {selector} which "
                f"ships batch input of size larger than one, but at the same time workflow block "
                f"used to implement the step does not accept batch input. That may be "
                f"for instance the case for steps with flow-control, as workflows execution engine "
                f"does not yet support branching when control-flow decision is made element-wise.",
                context="workflow_execution | steps_parameters_assembling",
            )
        return value[0]
    except KeyError as e:
        raise ExecutionEngineRuntimeError(
            public_message=f"Attempted to retrieve runtime parameter using selector {selector} "
            f"discovering miss in runtime parameters. This should have been detected "
            f"by execution engine at the earlier stage. "
            f"Contact Roboflow team through github issues "
            f"(https://github.com/roboflow/inference/issues) providing full context of"
            f"the problem - including workflow definition you use.",
            context="workflow_execution | steps_parameters_assembling",
            inner_error=e,
        ) from e


def _retrieved_inference_image(value: Any) -> bool:
    if not isinstance(value, list):
        return False
    if len(value) < 1:
        return False
    if not isinstance(value[0], dict):
        return False
    if "type" in value[0] and "value" in value[0]:
        return True
    return False
