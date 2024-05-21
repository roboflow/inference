import os.path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from inference.core.utils.image_utils import (
    attempt_loading_image_from_string,
    load_image_from_url,
)
from inference.core.workflows.entities.base import (
    InputType,
    OriginCoordinatesSystem,
    ParentImageMetadata,
    WorkflowImage,
    WorkflowImageData,
)
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


def assembly_input_image(parameter: str, image: Any) -> List[WorkflowImageData]:
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
) -> WorkflowImageData:
    parent_id = parameter
    if identifier is not None:
        parent_id = f"{parent_id}.[{identifier}]"
    if isinstance(image, dict) and isinstance(image.get("value"), np.ndarray):
        image = image["value"]
    if isinstance(image, np.ndarray):
        parent_metadata = ParentImageMetadata(
            parent_id=parent_id,
            origin_coordinates=OriginCoordinatesSystem(
                left_top_x=0,
                left_top_y=0,
                origin_width=image.shape[1],
                origin_height=image.shape[0],
            ),
        )
        return WorkflowImageData(
            parent_metadata=parent_metadata,
            numpy_image=image,
        )
    try:
        if isinstance(image, dict):
            image = image["value"]
        if isinstance(image, str):
            base64_image = None
            image_reference = None
            if image.startswith("http://") or image.startswith("https://"):
                image_reference = image
                image = load_image_from_url(value=image)
            elif os.path.exists(image):
                image_reference = image
                image = cv2.imread(image)
            else:
                base64_image = image
                image = attempt_loading_image_from_string(image)[0]
            parent_metadata = ParentImageMetadata(
                parent_id=parent_id,
                origin_coordinates=OriginCoordinatesSystem(
                    left_top_x=0,
                    left_top_y=0,
                    origin_width=image.shape[1],
                    origin_height=image.shape[0],
                ),
            )
            return WorkflowImageData(
                parent_metadata=parent_metadata,
                numpy_image=image,
                base64_image=base64_image,
                image_reference=image_reference,
            )
    except Exception as error:
        raise RuntimeInputError(
            public_message=f"Detected runtime parameter `{parameter}` defined as `WorkflowImage` "
            f"that is invalid. Failed on input validation. Details: {error}",
            context="workflow_execution | runtime_input_validation",
        ) from error
    raise RuntimeInputError(
        public_message=f"Detected runtime parameter `{parameter}` defined as `WorkflowImage` "
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
