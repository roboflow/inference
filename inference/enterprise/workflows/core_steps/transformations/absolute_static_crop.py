import itertools
from typing import Any, Dict, List, Literal, Tuple, Type, Union
from uuid import uuid4

import numpy as np
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, PositiveInt

from inference.core.utils.image_utils import ImageType, load_image
from inference.enterprise.workflows.complier.steps_executors.constants import (
    CENTER_X_KEY,
    CENTER_Y_KEY,
    DETECTION_ID_KEY,
    HEIGHT_KEY,
    IMAGE_TYPE_KEY,
    IMAGE_VALUE_KEY,
    ORIGIN_COORDINATES_KEY,
    ORIGIN_SIZE_KEY,
    PARENT_ID_KEY,
    WIDTH_KEY,
)
from inference.enterprise.workflows.core_steps.common.utils import (
    detection_to_xyxy,
    extract_origin_size_from_images,
)
from inference.enterprise.workflows.entities.steps import OutputDefinition
from inference.enterprise.workflows.entities.types import (
    IMAGE_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    INTEGER_KIND,
    KEYPOINT_DETECTION_PREDICTION_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    PARENT_ID_KIND,
    FlowControl,
    InferenceImageSelector,
    InferenceParameterSelector,
    OutputStepImageSelector,
    StepOutputSelector,
)
from inference.enterprise.workflows.prototypes.block import (
    WorkflowBlock,
    WorkflowBlockManifest,
)


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "description": "Responsible for cropping RoIs from images - using absolute coordinates.",
            "docs": "https://inference.roboflow.com/workflows/absolute_static_crop/",
            "block_type": "transformation",
        }
    )
    type: Literal["AbsoluteStaticCrop"]
    image: Union[InferenceImageSelector, OutputStepImageSelector] = Field(
        description="Reference at image to be used as input for step processing",
        examples=["$inputs.image", "$steps.cropping.crops"],
    )
    x_center: Union[PositiveInt, InferenceParameterSelector(kind=[INTEGER_KIND])] = (
        Field(
            description="Center X of static crop (absolute coordinate)",
            examples=[40, "$inputs.center_x"],
        )
    )
    y_center: Union[PositiveInt, InferenceParameterSelector(kind=[INTEGER_KIND])] = (
        Field(
            description="Center Y of static crop (absolute coordinate)",
            examples=[40, "$inputs.center_y"],
        )
    )
    width: Union[PositiveInt, InferenceParameterSelector(kind=[INTEGER_KIND])] = Field(
        description="Width of static crop (absolute value)",
        examples=[40, "$inputs.width"],
    )
    height: Union[PositiveInt, InferenceParameterSelector(kind=[INTEGER_KIND])] = Field(
        description="Height of static crop (absolute value)",
        examples=[40, "$inputs.height"],
    )


class AbsoluteStaticCropBlock(WorkflowBlock):

    @classmethod
    def get_input_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="crops", kind=[IMAGE_KIND]),
            OutputDefinition(name="parent_id", kind=[PARENT_ID_KIND]),
        ]

    async def run_locally(
        self,
        image: List[dict],
        x_center: int,
        y_center: int,
        width: int,
        height: int,
    ) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], FlowControl]]:
        decoded_images = [load_image(e) for e in image]
        decoded_images = [
            i[0] if i[1] is True else i[0][:, :, ::-1] for i in decoded_images
        ]
        origin_image_shape = extract_origin_size_from_images(
            input_images=image,
            decoded_images=decoded_images,
        )
        crops = [
            take_static_crop(
                image=i,
                x_center=x_center,
                y_center=y_center,
                width=width,
                height=height,
                origin_size=size,
            )
            for i, size in zip(decoded_images, origin_image_shape)
        ]
        return [{"crops": c, PARENT_ID_KEY: c["PARENT_ID_KEY"]} for c in crops]


def take_static_crop(
    image: np.ndarray,
    x_center: int,
    y_center: int,
    width: int,
    height: int,
    origin_size: dict,
) -> Dict[str, Union[str, np.ndarray]]:
    x_min = round(x_center - width / 2)
    y_min = round(y_center - height / 2)
    x_max = round(x_min + width)
    y_max = round(y_min + height)
    cropped_image = image[y_min:y_max, x_min:x_max]
    return {
        IMAGE_TYPE_KEY: ImageType.NUMPY_OBJECT.value,
        IMAGE_VALUE_KEY: cropped_image,
        PARENT_ID_KEY: f"absolute_static_crop.{uuid4()}",
        ORIGIN_COORDINATES_KEY: {
            CENTER_X_KEY: x_center,
            CENTER_Y_KEY: y_center,
            ORIGIN_SIZE_KEY: origin_size,
        },
    }
