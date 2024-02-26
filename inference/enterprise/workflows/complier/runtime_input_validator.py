from typing import Any, Dict, Optional, Set, Union

import numpy as np
from networkx import DiGraph

from inference.core.utils.image_utils import ImageType
from inference.enterprise.workflows.complier.steps_executors.constants import (
    IMAGE_TYPE_KEY,
    IMAGE_VALUE_KEY,
    PARENT_ID_KEY,
)
from inference.enterprise.workflows.complier.utils import (
    get_nodes_of_specific_kind,
    is_input_selector,
)
from inference.enterprise.workflows.constants import INPUT_NODE_KIND, STEP_NODE_KIND
from inference.enterprise.workflows.entities.validators import get_last_selector_chunk
from inference.enterprise.workflows.errors import (
    InvalidStepInputDetected,
    RuntimeParameterMissingError,
)


def prepare_runtime_parameters(
    execution_graph: DiGraph,
    runtime_parameters: Dict[str, Any],
) -> Dict[str, Any]:
    ensure_all_parameters_filled(
        execution_graph=execution_graph,
        runtime_parameters=runtime_parameters,
    )
    runtime_parameters = fill_runtime_parameters_with_defaults(
        execution_graph=execution_graph,
        runtime_parameters=runtime_parameters,
    )
    runtime_parameters = assembly_input_images(
        execution_graph=execution_graph,
        runtime_parameters=runtime_parameters,
    )
    validate_inputs_binding(
        execution_graph=execution_graph,
        runtime_parameters=runtime_parameters,
    )
    return runtime_parameters


def ensure_all_parameters_filled(
    execution_graph: DiGraph,
    runtime_parameters: Dict[str, Any],
) -> None:
    parameters_without_default_values = get_input_parameters_without_default_values(
        execution_graph=execution_graph,
    )
    missing_parameters = []
    for name in parameters_without_default_values:
        if name not in runtime_parameters:
            missing_parameters.append(name)
    if len(missing_parameters) > 0:
        raise RuntimeParameterMissingError(
            f"Parameters passed to execution runtime do not define required inputs: {missing_parameters}"
        )


def get_input_parameters_without_default_values(execution_graph: DiGraph) -> Set[str]:
    input_nodes = get_nodes_of_specific_kind(
        execution_graph=execution_graph,
        kind=INPUT_NODE_KIND,
    )
    result = set()
    for input_node in input_nodes:
        definition = execution_graph.nodes[input_node]["definition"]
        if definition.type == "InferenceImage":
            result.add(definition.name)
            continue
        if definition.type == "InferenceParameter" and definition.default_value is None:
            result.add(definition.name)
            continue
    return result


def fill_runtime_parameters_with_defaults(
    execution_graph: DiGraph,
    runtime_parameters: Dict[str, Any],
) -> Dict[str, Any]:
    default_values_parameters = get_input_parameters_default_values(
        execution_graph=execution_graph
    )
    default_values_parameters.update(runtime_parameters)
    return default_values_parameters


def get_input_parameters_default_values(execution_graph: DiGraph) -> Dict[str, Any]:
    input_nodes = get_nodes_of_specific_kind(
        execution_graph=execution_graph,
        kind=INPUT_NODE_KIND,
    )
    result = {}
    for input_node in input_nodes:
        definition = execution_graph.nodes[input_node]["definition"]
        if (
            definition.type == "InferenceParameter"
            and definition.default_value is not None
        ):
            result[definition.name] = definition.default_value
    return result


def assembly_input_images(
    execution_graph: DiGraph,
    runtime_parameters: Dict[str, Any],
) -> Dict[str, Any]:
    input_nodes = get_nodes_of_specific_kind(
        execution_graph=execution_graph,
        kind=INPUT_NODE_KIND,
    )
    for input_node in input_nodes:
        definition = execution_graph.nodes[input_node]["definition"]
        if definition.type != "InferenceImage":
            continue
        if issubclass(type(runtime_parameters[definition.name]), list):
            runtime_parameters[definition.name] = [
                assembly_input_image(
                    parameter=input_node,
                    image=image,
                    identifier=i,
                )
                for i, image in enumerate(runtime_parameters[definition.name])
            ]
        else:
            runtime_parameters[definition.name] = [
                assembly_input_image(
                    parameter=input_node, image=runtime_parameters[definition.name]
                )
            ]
    return runtime_parameters


def assembly_input_image(
    parameter: str, image: Any, identifier: Optional[int] = None
) -> Dict[str, Union[str, np.ndarray]]:
    parent = parameter
    if identifier is not None:
        parent = f"{parent}.[{identifier}]"
    if issubclass(type(image), dict):
        image[PARENT_ID_KEY] = parent
        return image
    if issubclass(type(image), np.ndarray):
        return {
            IMAGE_TYPE_KEY: ImageType.NUMPY_OBJECT.value,
            IMAGE_VALUE_KEY: image,
            PARENT_ID_KEY: parent,
        }
    raise InvalidStepInputDetected(
        f"Detected runtime parameter `{parameter}` defined as `InferenceImage` with type {type(image)} that is invalid."
    )


def validate_inputs_binding(
    execution_graph: DiGraph,
    runtime_parameters: Dict[str, Any],
) -> None:
    step_nodes = get_nodes_of_specific_kind(
        execution_graph=execution_graph,
        kind=STEP_NODE_KIND,
    )
    for step in step_nodes:
        validate_step_input_bindings(
            step=step,
            execution_graph=execution_graph,
            runtime_parameters=runtime_parameters,
        )


def validate_step_input_bindings(
    step: str,
    execution_graph: DiGraph,
    runtime_parameters: Dict[str, Any],
) -> None:
    step_definition = execution_graph.nodes[step]["definition"]
    for input_name in step_definition.get_input_names():
        selector_or_value = getattr(step_definition, input_name)
        if not is_input_selector(selector_or_value=selector_or_value):
            continue
        input_parameter_name = get_last_selector_chunk(selector=selector_or_value)
        parameter_value = runtime_parameters[input_parameter_name]
        step_definition.validate_field_binding(
            field_name=input_name, value=parameter_value
        )
