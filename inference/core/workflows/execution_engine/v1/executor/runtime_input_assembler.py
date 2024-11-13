from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from inference.core.workflows.errors import AssumptionError, RuntimeInputError
from inference.core.workflows.execution_engine.entities.base import InputType
from inference.core.workflows.execution_engine.entities.types import Kind
from inference.core.workflows.execution_engine.profiling.core import (
    WorkflowsProfiler,
    execution_phase,
)


@execution_phase(
    name="workflow_input_assembly",
    categories=["execution_engine_operation"],
)
def assemble_runtime_parameters(
    runtime_parameters: Dict[str, Any],
    defined_inputs: List[InputType],
    kinds_deserializers: Dict[str, Callable[[str, Any], Any]],
    prevent_local_images_loading: bool = False,
    profiler: Optional[WorkflowsProfiler] = None,
) -> Dict[str, Any]:
    input_batch_size = determine_input_batch_size(
        runtime_parameters=runtime_parameters,
        defined_inputs=defined_inputs,
    )
    for defined_input in defined_inputs:
        if defined_input.is_batch_oriented():
            runtime_parameters[defined_input.name] = assemble_batch_oriented_input(
                defined_input=defined_input,
                value=runtime_parameters.get(defined_input.name),
                kinds_deserializers=kinds_deserializers,
                input_batch_size=input_batch_size,
                prevent_local_images_loading=prevent_local_images_loading,
            )
        else:
            runtime_parameters[defined_input.name] = assemble_inference_parameter(
                parameter=defined_input.name,
                runtime_parameters=runtime_parameters,
                default_value=defined_input.default_value,
            )
    return runtime_parameters


def determine_input_batch_size(
    runtime_parameters: Dict[str, Any], defined_inputs: List[InputType]
) -> int:
    for defined_input in defined_inputs:
        if not defined_input.is_batch_oriented():
            continue
        parameter_value = runtime_parameters.get(defined_input.name)
        if isinstance(parameter_value, list) and len(parameter_value) > 1:
            return len(parameter_value)
    return 1


def assemble_batch_oriented_input(
    defined_input: InputType,
    value: Any,
    kinds_deserializers: Dict[str, Callable[[str, Any], Any]],
    input_batch_size: int,
    prevent_local_images_loading: bool,
) -> List[Any]:
    if value is None:
        raise RuntimeInputError(
            public_message=f"Detected runtime parameter `{defined_input.name}` defined as "
            f"`{defined_input.type}` (of kind `{[_get_kind_name(k) for k in defined_input.kind]}`), "
            f"but value is not provided.",
            context="workflow_execution | runtime_input_validation",
        )
    if not isinstance(value, list):
        result = [
            assemble_single_element_of_batch_oriented_input(
                defined_input=defined_input,
                value=value,
                kinds_deserializers=kinds_deserializers,
                prevent_local_images_loading=prevent_local_images_loading,
            )
        ] * input_batch_size
    else:
        result = [
            assemble_nested_batch_oriented_input(
                current_depth=1,
                defined_input=defined_input,
                value=element,
                kinds_deserializers=kinds_deserializers,
                prevent_local_images_loading=prevent_local_images_loading,
                identifier=f"{defined_input.name}.[{identifier}]",
            )
            for identifier, element in enumerate(value)
        ]
        if len(result) == 1 and len(result) != input_batch_size:
            result = result * input_batch_size
    if len(result) != input_batch_size:
        raise RuntimeInputError(
            public_message="Expected all batch-oriented workflow inputs be the same length, or of length 1 - "
            f"but parameter: {defined_input.name} provided with batch size {len(result)}, where expected "
            f"batch size based on remaining parameters is: {input_batch_size}.",
            context="workflow_execution | runtime_input_validation",
        )
    return result


def assemble_nested_batch_oriented_input(
    current_depth: int,
    defined_input: InputType,
    value: Any,
    kinds_deserializers: Dict[str, Callable[[str, Any], Any]],
    prevent_local_images_loading: bool,
    identifier: Optional[str] = None,
) -> Union[list, Any]:
    if current_depth > defined_input.dimensionality:
        raise AssumptionError(
            public_message=f"While constructing input `{defined_input.name}`, Execution Engine encountered the state "
            f"in which it is not possible to construct nested batch-oriented input. "
            f"This is most likely the bug. Contact Roboflow team "
            f"through github issues (https://github.com/roboflow/inference/issues) providing full "
            f"context of the problem - including workflow definition you use.",
            context="workflow_execution | step_input_assembling",
        )
    if current_depth == defined_input.dimensionality:
        return assemble_single_element_of_batch_oriented_input(
            defined_input=defined_input,
            value=value,
            kinds_deserializers=kinds_deserializers,
            prevent_local_images_loading=prevent_local_images_loading,
            identifier=identifier,
        )
    if not isinstance(value, list):
        raise RuntimeInputError(
            public_message=f"Workflow input `{defined_input.name}` is declared to be nested batch with dimensionality "
            f"`{defined_input.dimensionality}`. Input data does not define batch at the {current_depth} "
            f"dimensionality level.",
            context="workflow_execution | runtime_input_validation",
        )
    return [
        assemble_nested_batch_oriented_input(
            current_depth=current_depth + 1,
            defined_input=defined_input,
            value=element,
            kinds_deserializers=kinds_deserializers,
            prevent_local_images_loading=prevent_local_images_loading,
            identifier=f"{identifier}.[{idx}]",
        )
        for idx, element in enumerate(value)
    ]


def assemble_single_element_of_batch_oriented_input(
    defined_input: InputType,
    value: Any,
    kinds_deserializers: Dict[str, Callable[[str, Any], Any]],
    prevent_local_images_loading: bool,
    identifier: Optional[str] = None,
) -> Any:
    if value is None:
        return None
    matching_deserializers = _get_matching_deserializers(
        defined_input=defined_input,
        kinds_deserializers=kinds_deserializers,
    )
    if not matching_deserializers:
        return value
    parameter_identifier = defined_input.name
    if identifier is not None:
        parameter_identifier = identifier
    errors = []
    for kind, deserializer in matching_deserializers:
        try:
            if kind == "image":
                # this is left-over of bad design decision with adding `prevent_local_images_loading`
                # flag at the level of execution engine. To avoid BC we need to
                # be aware of special treatment for  image kind.
                # TODO: deprecate in v2 of Execution Engine
                return deserializer(
                    parameter_identifier, value, prevent_local_images_loading
                )
            return deserializer(parameter_identifier, value)
        except Exception as error:
            errors.append((kind, error))
    error_message = (
        f"Failed to assemble `{parameter_identifier}`. "
        f"Could not successfully use any deserializer for declared kinds. Details: "
    )
    for kind, error in errors:
        error_message = f"{error_message}\nKind: `{kind}` - Error: {error}"
    raise RuntimeInputError(
        public_message=error_message,
        context="workflow_execution | runtime_input_validation",
    )


def _get_matching_deserializers(
    defined_input: InputType,
    kinds_deserializers: Dict[str, Callable[[str, Any], Any]],
) -> List[Tuple[str, Callable[[str, Any], Any]]]:
    matching_deserializers = []
    for kind in defined_input.kind:
        kind_name = _get_kind_name(kind=kind)
        if kind_name not in kinds_deserializers:
            continue
        matching_deserializers.append((kind_name, kinds_deserializers[kind_name]))
    return matching_deserializers


def _get_kind_name(kind: Union[Kind, str]) -> str:
    if isinstance(kind, Kind):
        return kind.name
    return kind


def assemble_inference_parameter(
    parameter: str,
    runtime_parameters: Dict[str, Any],
    default_value: Any,
) -> Any:
    if parameter in runtime_parameters:
        return runtime_parameters[parameter]
    return default_value
