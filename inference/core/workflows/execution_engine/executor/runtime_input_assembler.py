import os.path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from inference.core.utils.image_utils import (
    attempt_loading_image_from_string,
    load_image_from_url,
)
from inference.core.workflows.entities.base import (
    ImageParentMetadata,
    InputType,
    WorkflowImage,
    WorkflowImageData,
)
from inference.core.workflows.errors import RuntimeInputError


def assembly_runtime_parameters(
    runtime_parameters: Dict[str, Any],
    defined_inputs: List[InputType],
    prevent_local_images_loading: bool = False,
) -> Dict[str, Any]:
    batch_input_size: Optional[int] = None
    for defined_input in defined_inputs:
        if isinstance(defined_input, WorkflowImage):
            runtime_parameters[defined_input.name] = assembly_input_image(
                parameter=defined_input.name,
                image=runtime_parameters.get(defined_input.name),
                batch_input_size=batch_input_size,
                prevent_local_images_loading=prevent_local_images_loading,
            )
            batch_input_size = len(runtime_parameters[defined_input.name])
        else:
            runtime_parameters[defined_input.name] = assembly_inference_parameter(
                parameter=defined_input.name,
                runtime_parameters=runtime_parameters,
                default_value=defined_input.default_value,
            )
    return runtime_parameters


def assembly_input_image(
    parameter: str,
    image: Any,
    batch_input_size: Optional[int],
    prevent_local_images_loading: bool = False,
) -> List[WorkflowImageData]:
    if image is None:
        raise RuntimeInputError(
            public_message=f"Detected runtime parameter `{parameter}` defined as "
            f"`InferenceImage`, but value is not provided.",
            context="workflow_execution | runtime_input_validation",
        )
    if not isinstance(image, list):
        expected_input_length = 1 if batch_input_size is None else batch_input_size
        return [
            _assembly_input_image(
                parameter=parameter,
                image=image,
                prevent_local_images_loading=prevent_local_images_loading,
            )
        ] * expected_input_length
    result = [
        _assembly_input_image(
            parameter=parameter,
            image=element,
            identifier=idx,
            prevent_local_images_loading=prevent_local_images_loading,
        )
        for idx, element in enumerate(image)
    ]
    expected_input_length = (
        len(result) if batch_input_size is None else batch_input_size
    )
    if len(result) != expected_input_length:
        raise RuntimeInputError(
            public_message="Expected all batch-oriented workflow inputs be the same length, or of length 1 - "
            f"but parameter: {parameter} provided with batch size {len(result)}, where expected "
            f"batch size based on remaining parameters is: {expected_input_length}.",
            context="workflow_execution | runtime_input_validation",
        )
    return result


def _assembly_input_image(
    parameter: str,
    image: Any,
    identifier: Optional[int] = None,
    prevent_local_images_loading: bool = False,
) -> WorkflowImageData:
    parent_id = parameter
    if identifier is not None:
        parent_id = f"{parent_id}.[{identifier}]"
    if isinstance(image, dict) and isinstance(image.get("value"), np.ndarray):
        image = image["value"]
    if isinstance(image, np.ndarray):
        parent_metadata = ImageParentMetadata(parent_id=parent_id)
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
            elif not prevent_local_images_loading and os.path.exists(image):
                # prevent_local_images_loading is introduced to eliminate
                # server vulnerability - namely it prevents local server
                # file system from being exploited.
                image_reference = image
                image = cv2.imread(image)
            else:
                base64_image = image
                image = attempt_loading_image_from_string(image)[0]
            parent_metadata = ImageParentMetadata(parent_id=parent_id)
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
