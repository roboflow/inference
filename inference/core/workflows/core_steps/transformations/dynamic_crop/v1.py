from dataclasses import replace
from typing import Dict, List, Literal, Optional, Tuple, Type, Union

import cv2
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
    FLOAT_ZERO_TO_ONE_KIND,
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
Create dynamic crops from an image based on detections from detections-based model.

This is useful when placed after an ObjectDetection block as part of a multi-stage 
workflow. For example, you could use an ObjectDetection block to detect objects, then 
the DynamicCropBlock block to crop objects, then an OCR block to run character recognition on 
each of the individual cropped regions.

In addition, for instance segmentation predictions (which provide segmentation mask for each 
bounding box) it is possible to remove background in the crops, outside of detected instances.
To enable that functionality, set `mask_opacity` to positive value and optionally tune 
`background_color`.
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Dynamic Crop",
            "version": "v1",
            "short_description": "Crop an image using bounding boxes from a detection model.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "transformation",
        }
    )
    type: Literal["roboflow_core/dynamic_crop@v1", "DynamicCrop", "Crop"]
    images: Union[WorkflowImageSelector, StepOutputImageSelector] = Field(
        title="Image to Crop",
        description="The input image for this step.",
        examples=["$inputs.image", "$steps.cropping.crops"],
        validation_alias=AliasChoices("images", "image"),
    )
    predictions: StepOutputSelector(
        kind=[
            OBJECT_DETECTION_PREDICTION_KIND,
            INSTANCE_SEGMENTATION_PREDICTION_KIND,
            KEYPOINT_DETECTION_PREDICTION_KIND,
        ]
    ) = Field(
        title="Regions of Interest",
        description="The output of a detection model describing the bounding boxes that will be used to crop the image.",
        examples=["$steps.my_object_detection_model.predictions"],
        validation_alias=AliasChoices("predictions", "detections"),
    )
    mask_opacity: Union[
        WorkflowParameterSelector(kind=[FLOAT_ZERO_TO_ONE_KIND]),
        float,
    ] = Field(
        default=0.0,
        le=1.0,
        ge=0.0,
        description="For instance segmentation, mask_opacity can be used to control background removal. "
        "Opacity 1.0 removes the background, while 0.0 leaves the crop unchanged.",
        json_schema_extra={
            "relevant_for": {
                "predictions": {
                    "kind": [INSTANCE_SEGMENTATION_PREDICTION_KIND.name],
                    "required": True,
                },
            }
        },
    )
    background_color: Union[
        WorkflowParameterSelector(kind=[STRING_KIND]),
        StepOutputSelector(kind=[RGB_COLOR_KIND]),
        str,
        Tuple[int, int, int],
    ] = Field(
        default=(0, 0, 0),
        description="For background removal based on segmentation mask, new background color can be selected. "
        "Can be a hex string (like '#431112') RGB string (like '(128, 32, 64)') or a RGB tuple "
        "(like (18, 17, 67)).",
        examples=["#431112", "$inputs.bg_color", (18, 17, 67)],
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


class DynamicCropBlockV1(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        images: Batch[WorkflowImageData],
        predictions: Batch[sv.Detections],
        mask_opacity: float,
        background_color: Union[str, Tuple[int, int, int]],
    ) -> BlockResult:
        return [
            crop_image(
                image=image,
                detections=detections,
                mask_opacity=mask_opacity,
                background_color=background_color,
            )
            for image, detections in zip(images, predictions)
        ]


def crop_image(
    image: WorkflowImageData,
    detections: sv.Detections,
    mask_opacity: float,
    background_color: Union[str, Tuple[int, int, int]],
    detection_id_key: str = DETECTION_ID_KEY,
) -> List[Dict[str, WorkflowImageData]]:
    if len(detections) == 0:
        return []
    if detection_id_key not in detections.data:
        raise ValueError(
            f"sv.Detections object passed to crop step do not fulfill contract - lack of {detection_id_key} key "
            f"in data dictionary."
        )
    crops = []
    for idx, ((x_min, y_min, x_max, y_max), detection_id) in enumerate(
        zip(detections.xyxy.round().astype(dtype=int), detections[detection_id_key])
    ):
        cropped_image = image.numpy_image[y_min:y_max, x_min:x_max]
        if not cropped_image.size:
            crops.append({"crops": None})
            continue
        if mask_opacity > 0 and detections.mask is not None:
            detection_mask = detections.mask[idx]
            cropped_mask = np.stack(
                [detection_mask[y_min:y_max, x_min:x_max]] * 3, axis=-1
            )
            cropped_image = overlay_crop_with_mask(
                crop=cropped_image,
                mask=cropped_mask,
                mask_opacity=mask_opacity,
                background_color=background_color,
            )
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
            numpy_image=cropped_image,
        )
        crops.append({"crops": result})
    return crops


def overlay_crop_with_mask(
    crop: np.ndarray,
    mask: np.ndarray,
    mask_opacity: float,
    background_color: Union[str, Tuple[int, int, int]],
) -> np.ndarray:
    bgr_color = convert_color_to_bgr_tuple(color=background_color)
    background = (np.ones_like(crop) * bgr_color).astype(np.uint8)
    blended_crop = np.where(mask > 0, crop, background)
    return cv2.addWeighted(blended_crop, mask_opacity, crop, 1.0 - mask_opacity, 0)


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
