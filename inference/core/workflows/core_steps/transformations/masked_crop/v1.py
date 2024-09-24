from dataclasses import replace
from typing import Dict, List, Literal, Optional, Tuple, Type, Union

import numpy as np
import supervision as sv
from pydantic import AliasChoices, ConfigDict, Field

from inference.core.workflows.execution_engine.constants import DETECTION_ID_KEY
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    ImageParentMetadata,
    OriginCoordinatesSystem,
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    IMAGE_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    KEYPOINT_DETECTION_PREDICTION_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    RGB_COLOR_KIND,
    STRING_KIND,
    StepOutputImageSelector,
    StepOutputSelector,
    WorkflowImageSelector,
    WorkflowParameterSelector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

LONG_DESCRIPTION = """
Create crops from an image based on predictions from instance segmentation model.

The block is useful when combined with instance segmentation model (for instance 
`roboflow_core/roboflow_instance_segmentation_model@v1`) which produces segmentation masks.
Each mask will be used to create the crop of the image - based on the mask shape, rectangular 
area of crop will be chosen and pixels outside of the instance mask will be replaced with
selected color.
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Masked Crop",
            "version": "v1",
            "short_description": "Crop an image using segmentation mask subtracting object background.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "transformation",
        }
    )
    type: Literal["roboflow_core/masked_crop@v1"]
    images: Union[WorkflowImageSelector, StepOutputImageSelector] = Field(
        title="Image to Crop",
        description="The input image for this step.",
        examples=["$inputs.image", "$steps.cropping.crops"],
        validation_alias=AliasChoices("images", "image"),
    )
    predictions: StepOutputSelector(kind=[INSTANCE_SEGMENTATION_PREDICTION_KIND]) = (
        Field(
            title="Segmentation Masks",
            description="The output of a detection model describing the bounding boxes that will be used to crop the image.",
            examples=["$steps.segmentation_model.predictions"],
        )
    )
    background_color: Union[
        WorkflowParameterSelector(kind=[STRING_KIND]),
        StepOutputSelector(kind=[RGB_COLOR_KIND]),
        str,
        Tuple[int, int, int],
    ] = Field(
        default=(127, 127, 127),
        description="Target color to count in the image. Can be a hex string "
        "(like '#431112') RGB string (like '(128, 32, 64)') or a RGB tuple "
        "(like (18, 17, 67)).",
        examples=["#431112", "$inputs.target_color", (18, 17, 67)],
    )

    @classmethod
    def accepts_batch_input(cls) -> bool:
        return True

    @classmethod
    def get_output_dimensionality_offset(cls) -> int:
        return 1

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="crops", kind=[IMAGE_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.0.0,<2.0.0"


class MaskedCropBlockV1(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        images: Batch[WorkflowImageData],
        predictions: Batch[sv.Detections],
        background_color: Union[str, Tuple[int, int, int]],
    ) -> BlockResult:
        return [
            mask_and_crop_image(
                image=image,
                detections=detections,
                background_color=background_color,
            )
            for image, detections in zip(images, predictions)
        ]


def mask_and_crop_image(
    image: WorkflowImageData,
    detections: sv.Detections,
    background_color: Union[str, Tuple[int, int, int]],
    detection_id_key: str = DETECTION_ID_KEY,
) -> List[Dict[str, WorkflowImageData]]:
    if len(detections) == 0:
        return []
    if detections.mask is None:
        raise ValueError(
            f"sv.Detections object passed to masked crop step do not fulfill contract - "
            f"lack of segmentation mask."
        )
    if detection_id_key not in detections.data:
        raise ValueError(
            f"sv.Detections object passed to masked crop step do not fulfill contract - "
            f"lack of {detection_id_key} key in data dictionary."
        )
    crops = []
    bgr_color = convert_color_to_bgr_tuple(color=background_color)
    for (x_min, y_min, x_max, y_max), mask, detection_id in zip(
        detections.xyxy.round().astype(dtype=int),
        detections.mask,
        detections[detection_id_key],
    ):
        cropped_image = image.numpy_image[y_min:y_max, x_min:x_max]
        if not cropped_image.size:
            crops.append({"crops": None})
            continue
        cropped_mask = np.stack([mask[y_min:y_max, x_min:x_max]] * 3, axis=-1)
        background = np.ones_like(cropped_image) * bgr_color
        blended_crop = np.where(cropped_mask > 0, cropped_image, background)
        parent_metadata = ImageParentMetadata(
            parent_id=detection_id,
            origin_coordinates=OriginCoordinatesSystem(
                left_top_x=x_min,
                left_top_y=y_min,
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
        result = WorkflowImageData(
            parent_metadata=parent_metadata,
            workflow_root_ancestor_metadata=workflow_root_ancestor_metadata,
            numpy_image=blended_crop,
        )
        crops.append({"crops": result})
    return crops


def convert_color_to_bgr_tuple(
    color: Union[str, Tuple[int, int, int]]
) -> Tuple[int, int, int]:
    if isinstance(color, str):
        return convert_string_color_to_bgr_tuple(color=color)
    if isinstance(color, tuple) and len(color) == 3:
        return color[::-1]
    raise ValueError(f"Invalid color format: {color}")


def convert_string_color_to_bgr_tuple(color: str) -> Tuple[int, int, int]:
    if color.startswith("#") and len(color) == 7:
        try:
            return tuple(int(color[i : i + 2], 16) for i in (5, 3, 1))
        except ValueError as e:
            raise ValueError(f"Invalid hex color format: {color}") from e
    if color.startswith("#") and len(color) == 4:
        try:
            return tuple(int(color[i] + color[i], 16) for i in (3, 2, 1))
        except ValueError as e:
            raise ValueError(f"Invalid hex color format: {color}") from e
    if color.startswith("(") and color.endswith(")"):
        try:
            return tuple(map(int, color[1:-1].split(",")))[::-1]
        except ValueError as e:
            raise ValueError(f"Invalid tuple color format: {color}") from e
    raise ValueError(f"Invalid hex color format: {color}")
