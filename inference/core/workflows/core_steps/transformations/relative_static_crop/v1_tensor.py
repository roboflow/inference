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

LONG_DESCRIPTION = """Tensor-native sibling of RelativeStaticCrop — reads
image dims via `_read_shape_without_materialization` then slices
WorkflowImageData.tensor_image directly. No numpy materialisation."""


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
        examples=[0.3, "$inputs.center_x"]
    )
    y_center: Union[FloatZeroToOne, Selector(kind=[FLOAT_ZERO_TO_ONE_KIND])] = Field(
        examples=[0.3, "$inputs.center_y"]
    )
    width: Union[FloatZeroToOne, Selector(kind=[FLOAT_ZERO_TO_ONE_KIND])] = Field(
        examples=[0.3, "$inputs.width"]
    )
    height: Union[FloatZeroToOne, Selector(kind=[FLOAT_ZERO_TO_ONE_KIND])] = Field(
        examples=[0.3, "$inputs.height"]
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
    x_center: float,
    y_center: float,
    width: float,
    height: float,
) -> Optional[WorkflowImageData]:
    h, w = image._read_shape_without_materialization()
    x_center_px = round(w * x_center)
    y_center_px = round(h * y_center)
    width_px = round(w * width)
    height_px = round(h * height)
    x_min = round(x_center_px - width_px / 2)
    y_min = round(y_center_px - height_px / 2)
    x_max = round(x_min + width_px)
    y_max = round(y_min + height_px)
    cropped = image.tensor_image[y_min:y_max, x_min:x_max, :]
    if cropped.numel() == 0:
        return None
    return WorkflowImageData.create_crop_from_tensor(
        origin_image_data=image,
        crop_identifier=f"relative_static_crop.{uuid4()}",
        cropped_tensor_image=cropped.contiguous(),
        offset_x=x_min,
        offset_y=y_min,
        preserve_video_metadata=True,
    )
