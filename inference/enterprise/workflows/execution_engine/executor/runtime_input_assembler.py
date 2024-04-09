from typing import Any, Dict, List, Optional, Union

import numpy as np

from inference.core.utils.image_utils import ImageType
from inference.enterprise.workflows.complier.steps_executors.constants import (
    IMAGE_TYPE_KEY,
    IMAGE_VALUE_KEY,
    PARENT_ID_KEY,
)
from inference.enterprise.workflows.entities.inputs import InferenceImage
from inference.enterprise.workflows.entities.workflows_specification import InputType
from inference.enterprise.workflows.errors import InvalidStepInputDetected


def assembly_runtime_parameters(
    runtime_parameters: Dict[str, Any],
    defined_inputs: List[InputType],
) -> Dict[str, Any]:
    for defined_input in defined_inputs:
        if issubclass(type(defined_input), InferenceImage):
            runtime_parameters[defined_input.name] = assembly_input_image(
                parameter=defined_input.name,
                image=runtime_parameters.get(defined_input.name),
            )
        else:
            runtime_parameters[defined_input.name] = assembly_inference_parameter(
                parameter=defined_input.name,
                runtime_parameters=runtime_parameters,
                default_value=defined_input.default_value,
            )
    return runtime_parameters


def assembly_input_image(
    parameter: str, image: Any
) -> List[Dict[str, Union[str, np.ndarray]]]:
    if image is None:
        raise InvalidStepInputDetected(
            f"Detected runtime parameter `{parameter}` defined as `InferenceImage` which is not provided."
        )
    if not issubclass(type(image), list):
        return [_assembly_input_image(parameter=parameter, image=image)]
    return [
        _assembly_input_image(parameter=parameter, image=element, identifier=idx)
        for idx, element in enumerate(image)
    ]


def _assembly_input_image(
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


def assembly_inference_parameter(
    parameter: str,
    runtime_parameters: Dict[str, Any],
    default_value: Any,
) -> Any:
    if parameter in runtime_parameters:
        return runtime_parameters[parameter]
    return default_value
