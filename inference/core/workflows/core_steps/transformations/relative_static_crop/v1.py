from typing import List, Literal, Optional, Type, Union
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    FLOAT_ZERO_TO_ONE_KIND,
    IMAGE_KIND,
    FloatZeroToOne,
    ImageInputField,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

LONG_DESCRIPTION = """
Crop a Region of Interest (RoI) from an image, using relative coordinates.

This is useful when placed after an ObjectDetection block as part of a multi-stage 
workflow. For example, you could use an ObjectDetection block to detect objects, then 
the RelativeStaticCrop block to crop objects, then an OCR block to run character 
recognition on each of the individual cropped regions.
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Relative Static Crop",
            "version": "v1",
            "short_description": "Crop an image proportional (%) to its dimensions.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "transformation",
            "ui_manifest": {
                "section": "transformation",
                "icon": "far fa-crop-alt",
                "blockPriority": 2,
            },
        }
    )
    type: Literal["roboflow_core/relative_statoic_crop@v1", "RelativeStaticCrop"]
    images: Selector(kind=[IMAGE_KIND]) = ImageInputField
    x_center: Union[FloatZeroToOne, Selector(kind=[FLOAT_ZERO_TO_ONE_KIND])] = Field(
        description="Center X of static crop (relative coordinate 0.0-1.0)",
        examples=[0.3, "$inputs.center_x"],
    )
    y_center: Union[FloatZeroToOne, Selector(kind=[FLOAT_ZERO_TO_ONE_KIND])] = Field(
        description="Center Y of static crop (relative coordinate 0.0-1.0)",
        examples=[0.3, "$inputs.center_y"],
    )
    width: Union[FloatZeroToOne, Selector(kind=[FLOAT_ZERO_TO_ONE_KIND])] = Field(
        description="Width of static crop (relative value 0.0-1.0)",
        examples=[0.3, "$inputs.width"],
    )
    height: Union[FloatZeroToOne, Selector(kind=[FLOAT_ZERO_TO_ONE_KIND])] = Field(
        description="Height of static crop (relative value 0.0-1.0)",
        examples=[0.3, "$inputs.height"],
    )

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["images"]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="crops", kind=[IMAGE_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class RelativeStaticCropBlockV1(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[BaseModel]:
        return BlockManifest

    def run(
        self,
        images: Batch[WorkflowImageData],
        x_center: float,
        y_center: float,
        width: float,
        height: float,
    ) -> BlockResult:
        return [
            {
                "crops": take_static_crop(
                    image=image,
                    x_center=x_center,
                    y_center=y_center,
                    width=width,
                    height=height,
                )
            }
            for image in images
        ]


def take_static_crop(
    image: WorkflowImageData,
    x_center: float,
    y_center: float,
    width: float,
    height: float,
) -> Optional[WorkflowImageData]:
    x_center = round(image.numpy_image.shape[1] * x_center)
    y_center = round(image.numpy_image.shape[0] * y_center)
    width = round(image.numpy_image.shape[1] * width)
    height = round(image.numpy_image.shape[0] * height)
    x_min = round(x_center - width / 2)
    y_min = round(y_center - height / 2)
    x_max = round(x_min + width)
    y_max = round(y_min + height)
    cropped_image = image.numpy_image[y_min:y_max, x_min:x_max]
    if not cropped_image.size:
        return None
    return WorkflowImageData.create_crop(
        origin_image_data=image,
        crop_identifier=f"relative_static_crop.{uuid4()}",
        cropped_image=cropped_image,
        offset_x=x_min,
        offset_y=y_min,
        preserve_video_metadata=True,
    )
