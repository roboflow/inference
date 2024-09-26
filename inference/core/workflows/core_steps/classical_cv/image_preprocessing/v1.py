from typing import List, Literal, Optional, Type, Union

import cv2
import numpy as np
from pydantic import AliasChoices, ConfigDict, Field

from inference.core.workflows.execution_engine.entities.base import (
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    IMAGE_KIND,
    INTEGER_KIND,
    STRING_KIND,
    StepOutputImageSelector,
    WorkflowImageSelector,
    WorkflowParameterSelector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

SHORT_DESCRIPTION = "Resize, flip, or rotate an image."
LONG_DESCRIPTION = """
Apply a resize, flip, or rotation step to an image. 

Width and height are required for resizing. Degrees are required for rotating. Flip type is required for flipping.
"""


class ImagePreprocessingManifest(WorkflowBlockManifest):
    type: Literal["roboflow_core/image_preprocessing@v1"]
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Image Preprocessing",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "classical_computer_vision",
            "ui_manifest": {
                "section": "classical_cv",
                "icon": "far fa-image",
                "blockPriority": 0.1,
                "opencv": True,
            },
        }
    )
    image: Union[WorkflowImageSelector, StepOutputImageSelector] = Field(
        title="Input Image",
        description="The input image for this step.",
        examples=["$inputs.image", "$steps.cropping.crops"],
        validation_alias=AliasChoices("image", "images"),
    )
    task_type: Literal["resize", "rotate", "flip"] = Field(
        description="Preprocessing task to be applied to the image.",
    )
    width: Union[int, WorkflowParameterSelector(kind=[INTEGER_KIND])] = Field(  # type: ignore
        title="Width",
        default=640,
        description="Width of the image to be resized to.",
        examples=[640, "$inputs.resize_width"],
        gt=0,
        json_schema_extra={
            "relevant_for": {
                "task_type": {
                    "values": ["resize"],
                    "required": True,
                },
            },
        },
    )
    height: Union[int, WorkflowParameterSelector(kind=[INTEGER_KIND])] = Field(  # type: ignore
        title="Height",
        default=640,
        description="Height of the image to be resized to.",
        examples=[640, "$inputs.resize_height"],
        gt=0,
        json_schema_extra={
            "relevant_for": {
                "task_type": {
                    "values": ["resize"],
                    "required": True,
                },
            },
        },
    )
    rotation_degrees: Union[int, WorkflowParameterSelector(kind=[INTEGER_KIND])] = Field(  # type: ignore
        title="Degrees of Rotation",
        description="Positive value to rotate clockwise, negative value to rotate counterclockwise",
        default=90,
        examples=[90, "$inputs.rotation_degrees"],
        gte=-360,
        le=360,
        json_schema_extra={
            "relevant_for": {
                "task_type": {
                    "values": ["rotate"],
                    "required": True,
                },
            }
        },
    )
    flip_type: Union[WorkflowParameterSelector(kind=[STRING_KIND]), Literal["vertical", "horizontal", "both"]] = Field(  # type: ignore
        title="Flip Type",
        description="Type of flip to be applied to the image.",
        default="vertical",
        examples=["vertical", "horizontal", "$inputs.flip_type"],
        json_schema_extra={
            "relevant_for": {
                "task_type": {
                    "values": ["flip"],
                    "required": True,
                },
            }
        },
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="image", kind=[IMAGE_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.0.0,<2.0.0"


class ImagePreprocessingBlockV1(WorkflowBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def get_manifest(cls) -> Type[ImagePreprocessingManifest]:
        return ImagePreprocessingManifest

    def run(
        self,
        image: WorkflowImageData,
        task_type: str,
        width: Optional[int],
        height: Optional[int],
        rotation_degrees: Optional[int],
        flip_type: Optional[str],
        *args,
        **kwargs,
    ) -> BlockResult:

        response_image = None

        if task_type == "resize":
            response_image = apply_resize_image(image.numpy_image, width, height)
        elif task_type == "rotate":
            response_image = apply_rotate_image(image.numpy_image, rotation_degrees)
        elif task_type == "flip":
            response_image = apply_flip_image(image.numpy_image, flip_type)
        else:
            raise ValueError(f"Invalid task type: {task_type}")

        output_image = WorkflowImageData(
            parent_metadata=image.parent_metadata,
            workflow_root_ancestor_metadata=image.workflow_root_ancestor_metadata,
            numpy_image=response_image,
        )
        return {"image": output_image}


def apply_resize_image(
    np_image: np.ndarray, width: Optional[int], height: Optional[int]
) -> np.ndarray:
    if width is None and height is None:
        return np_image.copy()

    current_height, current_width = np_image.shape[:2]

    if width is None:
        aspect_ratio = height / current_height
        width = int(current_width * aspect_ratio)
    elif height is None:
        aspect_ratio = width / current_width
        height = int(current_height * aspect_ratio)

    resized_image = cv2.resize(np_image, (width, height), interpolation=cv2.INTER_AREA)
    return resized_image


def apply_rotate_image(np_image: np.ndarray, rotation_degrees: Optional[int]):
    if rotation_degrees is None or rotation_degrees == 0:
        return np_image.copy()
    existing_height, existing_width = np_image.shape[:2]
    center = (existing_width // 2, existing_height // 2)  # Corrected order
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation_degrees, 1.0)

    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])
    new_width = int((existing_height * sin) + (existing_width * cos))
    new_height = int((existing_height * cos) + (existing_width * sin))

    rotation_matrix[0, 2] += (new_width / 2) - center[0]
    rotation_matrix[1, 2] += (new_height / 2) - center[1]

    rotated_image = cv2.warpAffine(np_image, rotation_matrix, (new_width, new_height))
    return rotated_image


def apply_flip_image(np_image: np.ndarray, flip_type: Optional[str]):
    if flip_type == "vertical":
        return cv2.flip(np_image, 0)
    elif flip_type == "horizontal":
        return cv2.flip(np_image, 1)
    elif flip_type == "both":
        return cv2.flip(np_image, -1)
    else:
        return np_image
