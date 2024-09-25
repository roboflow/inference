from dataclasses import replace
from typing import List, Literal, Optional, Tuple, Type, Union
from uuid import uuid4

import numpy as np
from pydantic import AliasChoices, ConfigDict, Field, PositiveInt
from supervision import crop_image
from typing_extensions import Annotated

from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    OriginCoordinatesSystem,
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    FLOAT_ZERO_TO_ONE_KIND,
    IMAGE_KIND,
    INTEGER_KIND,
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
This block enables [Slicing Adaptive Inference (SAHI)](https://ieeexplore.ieee.org/document/9897990) technique in 
Workflows providing implementation for first step of procedure - making slices out of input image.

To use the block effectively, it must be paired with detection model (object-detection or 
instance segmentation) running against output images from this block. At the end - 
Detections Stitch block must be applied on top of predictions to merge them as if 
the prediction was made against input image, not its slices.

We recommend adjusting the size of slices to match the model's input size and the scale of objects in the dataset 
the model was trained on. Models generally perform best on data that is similar to what they encountered during 
training. The default size of slices is 640, but this might not be optimal if the model's input size is 320, as each 
slice would be downsized by a factor of two during inference. Similarly, if the model's input size is 1280, each slice 
will be artificially up-scaled. The best setup should be determined experimentally based on the specific data and model 
you are using.

To learn more about SAHI please visit [Roboflow blog](https://blog.roboflow.com/how-to-use-sahi-to-detect-small-objects/)
which describes the technique in details, yet not in context of Roboflow workflows.
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Image Slicer",
            "version": "v1",
            "short_description": "Splits input image into series of smaller images to perform accurate prediction.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "transformation",
        }
    )
    type: Literal["roboflow_core/image_slicer@v1"]
    image: Union[WorkflowImageSelector, StepOutputImageSelector] = Field(
        title="Image to slice",
        description="The input image for this step.",
        examples=["$inputs.image", "$steps.cropping.crops"],
        validation_alias=AliasChoices("image", "images"),
    )
    slice_width: Union[PositiveInt, WorkflowParameterSelector(kind=[INTEGER_KIND])] = (
        Field(
            default=640,
            description="Width of each slice, in pixels",
            examples=[320, "$inputs.slice_width"],
        )
    )
    slice_height: Union[PositiveInt, WorkflowParameterSelector(kind=[INTEGER_KIND])] = (
        Field(
            default=640,
            description="Height of each slice, in pixels",
            examples=[320, "$inputs.slice_height"],
        )
    )
    overlap_ratio_width: Union[
        Annotated[float, Field(ge=0.0, lt=1.0)],
        WorkflowParameterSelector(kind=[FLOAT_ZERO_TO_ONE_KIND]),
    ] = Field(
        default=0.2,
        description="Overlap ratio between consecutive slices in the width dimension",
        examples=[0.2, "$inputs.overlap_ratio_width"],
    )
    overlap_ratio_height: Union[
        Annotated[float, Field(ge=0.0, lt=1.0)],
        WorkflowParameterSelector(kind=[FLOAT_ZERO_TO_ONE_KIND]),
    ] = Field(
        default=0.2,
        description="Overlap ratio between consecutive slices in the height dimension",
        examples=[0.2, "$inputs.overlap_ratio_height"],
    )

    @classmethod
    def get_output_dimensionality_offset(cls) -> int:
        return 1

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="slices", kind=[IMAGE_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.0.0,<2.0.0"


class ImageSlicerBlockV1(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        image: WorkflowImageData,
        slice_width: int,
        slice_height: int,
        overlap_ratio_width: float,
        overlap_ratio_height: float,
    ) -> BlockResult:
        image_numpy = image.numpy_image
        resolution_wh = (image_numpy.shape[1], image_numpy.shape[0])
        offsets = generate_offsets(
            resolution_wh=resolution_wh,
            slice_wh=(slice_width, slice_height),
            overlap_ratio_wh=(overlap_ratio_width, overlap_ratio_height),
        )
        slices = []
        for offset in offsets:
            x_min, y_min, _, _ = offset
            crop_numpy = crop_image(image=image_numpy, xyxy=offset)
            parent_metadata = ImageParentMetadata(
                parent_id=f"image_slicer.{uuid4()}",
                origin_coordinates=OriginCoordinatesSystem(
                    left_top_x=x_min.item(),
                    left_top_y=y_min.item(),
                    origin_width=image.numpy_image.shape[1],
                    origin_height=image.numpy_image.shape[0],
                ),
            )
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
            if crop_numpy.size:
                cropped_image = WorkflowImageData(
                    parent_metadata=parent_metadata,
                    workflow_root_ancestor_metadata=workflow_root_ancestor_metadata,
                    numpy_image=crop_numpy,
                )
                slices.append({"slices": cropped_image})
            else:
                slices.append({"slices": None})
        return slices


def generate_offsets(
    resolution_wh: Tuple[int, int],
    slice_wh: Tuple[int, int],
    overlap_ratio_wh: Optional[Tuple[float, float]],
) -> np.ndarray:
    """
    Original code: https://github.com/roboflow/supervision/blob/5123085037ec594524fc8f9d9b71b1cd9f487e8d/supervision/detection/tools/inference_slicer.py#L204-L203
    to avoid fragile contract with supervision, as this function is not element of public
    API.

    Generate offset coordinates for slicing an image based on the given resolution,
    slice dimensions, and overlap ratios.

    Args:
        resolution_wh (Tuple[int, int]): A tuple representing the width and height
            of the image to be sliced.
        slice_wh (Tuple[int, int]): Dimensions of each slice measured in pixels. The
        tuple should be in the format `(width, height)`.
        overlap_ratio_wh (Optional[Tuple[float, float]]): A tuple representing the
            desired overlap ratio for width and height between consecutive slices.
            Each value should be in the range [0, 1), where 0 means no overlap and
            a value close to 1 means high overlap.
    Returns:
        np.ndarray: An array of shape `(n, 4)` containing coordinates for each
            slice in the format `[xmin, ymin, xmax, ymax]`.

    Note:
        The function ensures that slices do not exceed the boundaries of the
            original image. As a result, the final slices in the row and column
            dimensions might be smaller than the specified slice dimensions if the
            image's width or height is not a multiple of the slice's width or
            height minus the overlap.
    """
    slice_width, slice_height = slice_wh
    image_width, image_height = resolution_wh
    overlap_width = int(overlap_ratio_wh[0] * slice_width)
    overlap_height = int(overlap_ratio_wh[1] * slice_height)
    width_stride = slice_width - overlap_width
    height_stride = slice_height - overlap_height
    ws = np.arange(0, image_width, width_stride)
    hs = np.arange(0, image_height, height_stride)
    xmin, ymin = np.meshgrid(ws, hs)
    xmax = np.clip(xmin + slice_width, 0, image_width)
    ymax = np.clip(ymin + slice_height, 0, image_height)
    return np.stack([xmin, ymin, xmax, ymax], axis=-1).reshape(-1, 4)
