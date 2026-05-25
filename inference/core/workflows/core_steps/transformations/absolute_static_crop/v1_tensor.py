from typing import List, Literal, Optional, Type, Union
from uuid import uuid4

from pydantic import ConfigDict, Field, PositiveInt

from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    IMAGE_KIND,
    INTEGER_KIND,
    ImageInputField,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

LONG_DESCRIPTION = """Tensor-native sibling of AbsoluteStaticCrop — slices
WorkflowImageData.tensor_image directly and creates the cropped child via
WorkflowImageData.create_crop_from_tensor. No numpy materialisation on the
hot path."""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Absolute Static Crop",
            "version": "v1",
            "short_description": "Crop an image using fixed pixel coordinates.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "transformation",
            "ui_manifest": {
                "section": "transformation",
                "icon": "far fa-crop-alt",
                "blockPriority": 1,
            },
        }
    )
    type: Literal["roboflow_core/absolute_static_crop@v1", "AbsoluteStaticCrop"]
    images: Selector(kind=[IMAGE_KIND]) = ImageInputField
    x_center: Union[PositiveInt, Selector(kind=[INTEGER_KIND])] = Field(
        examples=[40, "$inputs.center_x"]
    )
    y_center: Union[PositiveInt, Selector(kind=[INTEGER_KIND])] = Field(
        examples=[40, "$inputs.center_y"]
    )
    width: Union[PositiveInt, Selector(kind=[INTEGER_KIND])] = Field(
        examples=[40, "$inputs.width"]
    )
    height: Union[PositiveInt, Selector(kind=[INTEGER_KIND])] = Field(
        examples=[40, "$inputs.height"]
    )

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["images"]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="crops", kind=[IMAGE_KIND])]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class AbsoluteStaticCropBlockV1(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        images: Batch[WorkflowImageData],
        x_center: int,
        y_center: int,
        width: int,
        height: int,
    ) -> BlockResult:
        return [
            {
                "crops": _take_static_crop_tensor(
                    image=image,
                    x_center=x_center,
                    y_center=y_center,
                    width=width,
                    height=height,
                )
            }
            for image in images
        ]


def _take_static_crop_tensor(
    image: WorkflowImageData,
    x_center: int,
    y_center: int,
    width: int,
    height: int,
) -> Optional[WorkflowImageData]:
    x_min = round(x_center - width / 2)
    y_min = round(y_center - height / 2)
    x_max = round(x_min + width)
    y_max = round(y_min + height)
    cropped = image.tensor_image[y_min:y_max, x_min:x_max, :]
    if cropped.numel() == 0:
        return None
    return WorkflowImageData.create_crop_from_tensor(
        origin_image_data=image,
        crop_identifier=f"absolute_static_crop.{uuid4()}",
        cropped_tensor_image=cropped.contiguous(),
        offset_x=x_min,
        offset_y=y_min,
        preserve_video_metadata=True,
    )
