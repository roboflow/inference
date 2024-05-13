from typing import Any, Dict, List, Optional, Union

import numpy as np

from inference.core.utils.image_utils import ImageType
from inference.core.workflows.constants import (
    IMAGE_TYPE_KEY,
    IMAGE_VALUE_KEY,
    PARENT_ID_KEY,
)
from inference.core.workflows.entities.base import InputType, WorkflowImage
from inference.core.workflows.errors import RuntimeInputError


def assembly_runtime_parameters(
    runtime_parameters: Dict[str, Any],
    defined_inputs: List[InputType],
) -> Dict[str, Any]:
    for defined_input in defined_inputs:
        if isinstance(defined_input, WorkflowImage):
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
        raise RuntimeInputError(
            public_message=f"Detected runtime parameter `{parameter}` defined as "
            f"`InferenceImage`, but value is not provided.",
            context="workflow_execution | runtime_input_validation",
        )
    if not isinstance(image, list):
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
    if isinstance(image, dict):
        image[PARENT_ID_KEY] = parent
        return image
    if isinstance(image, np.ndarray):
        return {
            IMAGE_TYPE_KEY: ImageType.NUMPY_OBJECT.value,
            IMAGE_VALUE_KEY: image,
            PARENT_ID_KEY: parent,
        }
    raise RuntimeInputError(
        public_message=f"Detected runtime parameter `{parameter}` defined as `InferenceImage` "
        f"with type {type(image)} that is invalid. Workflows accept only np.arrays "
        f"and dicts with keys `type` and `value` compatible with `inference` (or list of them).",
        context="workflow_execution | runtime_input_validation",
    )


def assembly_inference_parameter(
    parameter: str,
    runtime_parameters: Dict[str, Any],
    default_value: Any,
) -> Any:
    if parameter in runtime_parameters:
        return runtime_parameters[parameter]
    return default_value
