import os.path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from pydantic import ValidationError

from inference.core.utils.image_utils import (
    attempt_loading_image_from_string,
    load_image_from_url,
)
from inference.core.workflows.errors import RuntimeInputError
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    InputType,
    VideoMetadata,
    WorkflowImage,
    WorkflowImageData,
    WorkflowVideoMetadata,
)
from inference.core.workflows.execution_engine.profiling.core import (
    WorkflowsProfiler,
    execution_phase,
)

BATCH_ORIENTED_PARAMETER_TYPES = {WorkflowImage, WorkflowVideoMetadata}


@execution_phase(
    name="workflow_input_assembly",
    categories=["execution_engine_operation"],
)
def assemble_runtime_parameters(
    runtime_parameters: Dict[str, Any],
    defined_inputs: List[InputType],
    prevent_local_images_loading: bool = False,
    profiler: Optional[WorkflowsProfiler] = None,
) -> Dict[str, Any]:
    input_batch_size = determine_input_batch_size(
        runtime_parameters=runtime_parameters,
        defined_inputs=defined_inputs,
    )
    for defined_input in defined_inputs:
        if isinstance(defined_input, WorkflowImage):
            runtime_parameters[defined_input.name] = assemble_input_image(
                parameter=defined_input.name,
                image=runtime_parameters.get(defined_input.name),
                input_batch_size=input_batch_size,
                prevent_local_images_loading=prevent_local_images_loading,
            )
        elif isinstance(defined_input, WorkflowVideoMetadata):
            runtime_parameters[defined_input.name] = assemble_video_metadata(
                parameter=defined_input.name,
                video_metadata=runtime_parameters.get(defined_input.name),
                input_batch_size=input_batch_size,
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
        if type(defined_input) not in BATCH_ORIENTED_PARAMETER_TYPES:
            continue
        parameter_value = runtime_parameters.get(defined_input.name)
        if isinstance(parameter_value, list) and len(parameter_value) > 1:
            return len(parameter_value)
    return 1


def assemble_input_image(
    parameter: str,
    image: Any,
    input_batch_size: int,
    prevent_local_images_loading: bool = False,
) -> List[WorkflowImageData]:
    if image is None:
        raise RuntimeInputError(
            public_message=f"Detected runtime parameter `{parameter}` defined as "
            f"`WorkflowImage`, but value is not provided.",
            context="workflow_execution | runtime_input_validation",
        )
    if not isinstance(image, list):
        return [
            _assemble_input_image(
                parameter=parameter,
                image=image,
                prevent_local_images_loading=prevent_local_images_loading,
            )
        ] * input_batch_size
    result = [
        _assemble_input_image(
            parameter=parameter,
            image=element,
            identifier=idx,
            prevent_local_images_loading=prevent_local_images_loading,
        )
        for idx, element in enumerate(image)
    ]
    if len(result) != input_batch_size:
        raise RuntimeInputError(
            public_message="Expected all batch-oriented workflow inputs be the same length, or of length 1 - "
            f"but parameter: {parameter} provided with batch size {len(result)}, where expected "
            f"batch size based on remaining parameters is: {input_batch_size}.",
            context="workflow_execution | runtime_input_validation",
        )
    return result


def _assemble_input_image(
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


def assemble_video_metadata(
    parameter: str,
    video_metadata: Any,
    input_batch_size: int,
) -> List[VideoMetadata]:
    if video_metadata is None:
        raise RuntimeInputError(
            public_message=f"Detected runtime parameter `{parameter}` defined as "
            f"`WorkflowVideoMetadata`, but value is not provided.",
            context="workflow_execution | runtime_input_validation",
        )
    if not isinstance(video_metadata, list):
        return [
            _assemble_video_metadata(
                parameter=parameter,
                video_metadata=video_metadata,
            )
        ] * input_batch_size
    result = [
        _assemble_video_metadata(
            parameter=parameter,
            video_metadata=element,
        )
        for element in video_metadata
    ]
    if len(result) != input_batch_size:
        raise RuntimeInputError(
            public_message="Expected all batch-oriented workflow inputs be the same length, or of length 1 - "
            f"but parameter: {parameter} provided with batch size {len(result)}, where expected "
            f"batch size based on remaining parameters is: {input_batch_size}.",
            context="workflow_execution | runtime_input_validation",
        )
    return result


def _assemble_video_metadata(
    parameter: str,
    video_metadata: Any,
) -> VideoMetadata:
    if isinstance(video_metadata, VideoMetadata):
        return video_metadata
    if not isinstance(video_metadata, dict):
        raise RuntimeInputError(
            public_message=f"Detected runtime parameter `{parameter}` defined as "
            f"`WorkflowVideoMetadata`, but provided value is not a dict.",
            context="workflow_execution | runtime_input_validation",
        )
    try:
        return VideoMetadata.model_validate(video_metadata)
    except ValidationError as error:
        raise RuntimeInputError(
            public_message=f"Detected runtime parameter `{parameter}` defined as "
            f"`WorkflowVideoMetadata`, but provided value is malformed. "
            f"See details in inner error.",
            context="workflow_execution | runtime_input_validation",
            inner_error=error,
        )


def assemble_inference_parameter(
    parameter: str,
    runtime_parameters: Dict[str, Any],
    default_value: Any,
) -> Any:
    if parameter in runtime_parameters:
        return runtime_parameters[parameter]
    return default_value
