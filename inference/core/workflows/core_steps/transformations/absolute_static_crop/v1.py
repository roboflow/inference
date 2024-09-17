from dataclasses import replace
from typing import List, Literal, Optional, Type, Union
from uuid import uuid4

from pydantic import ConfigDict, Field, PositiveInt

from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    ImageParentMetadata,
    OriginCoordinatesSystem,
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    IMAGE_KIND,
    INTEGER_KIND,
    ImageInputField,
    StepOutputImageSelector,
    WorkflowImageSelector,
    WorkflowParameterSelector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

LONG_DESCRIPTION = """
Crop a Region of Interest (RoI) from an image, using absolute coordinates.

This is useful when placed after an ObjectDetection block as part of a multi-stage 
workflow. For example, you could use an ObjectDetection block to detect objects, then 
the AbsoluteStaticCrop block to crop objects, then an OCR block to run character 
recognition on each of the individual cropped regions.
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Absolute Static Crop",
            "version": "v1",
            "short_description": "Crop an image using fixed pixel coordinates.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "transformation",
        }
    )
    type: Literal["roboflow_core/absolute_static_crop@v1", "AbsoluteStaticCrop"]
    images: Union[WorkflowImageSelector, StepOutputImageSelector] = ImageInputField
    x_center: Union[PositiveInt, WorkflowParameterSelector(kind=[INTEGER_KIND])] = (
        Field(
            description="Center X of static crop (absolute coordinate)",
            examples=[40, "$inputs.center_x"],
        )
    )
    y_center: Union[PositiveInt, WorkflowParameterSelector(kind=[INTEGER_KIND])] = (
        Field(
            description="Center Y of static crop (absolute coordinate)",
            examples=[40, "$inputs.center_y"],
        )
    )
    width: Union[PositiveInt, WorkflowParameterSelector(kind=[INTEGER_KIND])] = Field(
        description="Width of static crop (absolute value)",
        examples=[40, "$inputs.width"],
    )
    height: Union[PositiveInt, WorkflowParameterSelector(kind=[INTEGER_KIND])] = Field(
        description="Height of static crop (absolute value)",
        examples=[40, "$inputs.height"],
    )

    @classmethod
    def accepts_batch_input(cls) -> bool:
        return True

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="crops", kind=[IMAGE_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.0.0,<2.0.0"


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
    x_center: int,
    y_center: int,
    width: int,
    height: int,
) -> Optional[WorkflowImageData]:
    x_min = round(x_center - width / 2)
    y_min = round(y_center - height / 2)
    x_max = round(x_min + width)
    y_max = round(y_min + height)
    cropped_image = image.numpy_image[y_min:y_max, x_min:x_max]
    if not cropped_image.size:
        return None
    workflow_root_ancestor_coordinates = replace(
        image.workflow_root_ancestor_metadata.origin_coordinates,
        left_top_x=image.workflow_root_ancestor_metadata.origin_coordinates.left_top_x
        + x_min,
        left_top_y=image.workflow_root_ancestor_metadata.origin_coordinates.left_top_y
        + y_min,
    )
    workflow_root_ancestor_metadata = ImageParentMetadata(
        parent_id=image.workflow_root_ancestor_metadata.parent_id,
        origin_coordinates=workflow_root_ancestor_coordinates,
    )
    parent_metadata = ImageParentMetadata(
        parent_id=f"absolute_static_crop.{uuid4()}",
        origin_coordinates=OriginCoordinatesSystem(
            left_top_x=x_min,
            left_top_y=y_min,
            origin_width=image.numpy_image.shape[1],
            origin_height=image.numpy_image.shape[0],
        ),
    )
    return WorkflowImageData(
        parent_metadata=parent_metadata,
        workflow_root_ancestor_metadata=workflow_root_ancestor_metadata,
        numpy_image=cropped_image,
    )
