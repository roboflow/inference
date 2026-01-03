from dataclasses import replace
from typing import Dict, List, Literal, Optional, Tuple, Type, Union

import cv2
import numpy as np
import supervision as sv
from pydantic import AliasChoices, ConfigDict, Field

from inference.core.workflows.execution_engine.constants import (
    DETECTION_ID_KEY,
    KEYPOINTS_XY_KEY_IN_SV_DETECTIONS,
    POLYGON_KEY_IN_SV_DETECTIONS,
)
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
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
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

LONG_DESCRIPTION = """
Extract cropped image regions from input images based on bounding boxes from detection model predictions, supporting object detection, instance segmentation, and keypoint detection models with optional background removal using segmentation masks for focused region extraction and multi-stage analysis workflows.

## How This Block Works

This block crops rectangular regions from input images using bounding boxes from detection model outputs, producing individual cropped images for each detected object. The block:

1. Receives input images and detection predictions (object detection, instance segmentation, or keypoint detection) containing bounding boxes
2. Validates that predictions contain detection IDs required for crop tracking
3. Extracts each bounding box from the predictions and crops the corresponding rectangular region from the input image
4. For instance segmentation predictions with `mask_opacity > 0`: Applies background removal by overlaying the segmentation mask, replacing background pixels outside the detected instance with the specified `background_color` and blending with the original crop based on mask opacity
5. Creates cropped image objects with metadata tracking the crop's origin (original image, offset coordinates, detection ID)
6. Translates prediction coordinates from the original image space to the cropped region space (adjusts bounding boxes, masks, keypoints, and polygons to be relative to the crop origin)
7. Returns a list of results for each detection, each containing the cropped image and the translated predictions

The block processes each detection's bounding box independently, creating separate crops for each detected object. For instance segmentation predictions, the optional background removal feature uses the segmentation mask to isolate the detected object from background pixels, useful for creating clean object-focused crops. All prediction coordinates (bounding boxes, keypoints, polygons, mask coordinates) are automatically translated to be relative to the cropped region's top-left corner, ensuring downstream blocks can process the crops correctly. The block increases output dimensionality by one (produces a list of crops per input image), enabling batch processing workflows where each crop can be processed independently.

## Common Use Cases

- **Multi-Stage Object Analysis**: Extract individual object crops from full images for detailed analysis (e.g., detect objects in a scene, crop each detected object, then run OCR or classification on individual crops), enabling focused analysis of specific regions without processing entire images
- **Background Removal for Object Focus**: Create clean object crops with background removed using segmentation masks (e.g., detect and segment objects, crop with background removal, create isolated object images for training or analysis), enabling focused object extraction and cleaner downstream processing
- **Region-Based Processing Pipelines**: Extract regions of interest for specialized processing (e.g., detect text regions, crop each text region, run OCR on crops; detect faces, crop each face, run face recognition), enabling efficient processing of specific image regions
- **Keypoint and Annotation Preservation**: Extract object crops while preserving detection annotations (e.g., detect objects with keypoints, crop objects maintaining keypoint coordinates, analyze keypoints in cropped context), enabling focused analysis with full annotation context
- **Batch Region Extraction**: Extract multiple regions from single images for parallel processing (e.g., detect all objects in image, crop each object separately, process crops in parallel for classification or analysis), enabling efficient batch processing of multiple regions
- **Training Data Preparation**: Create cropped object datasets from annotated images (e.g., detect objects with bounding boxes, crop each object individually, export crops for training data collection), enabling automated extraction of training samples from full images

## Connecting to Other Blocks

This block receives images and detection predictions, producing cropped images:

- **After detection blocks** (e.g., Object Detection, Instance Segmentation, Keypoint Detection) to extract individual object regions based on detected bounding boxes, enabling focused analysis of detected objects in isolation
- **Before classification or analysis blocks** that need object-focused inputs (e.g., OCR for text regions, fine-grained classification for cropped objects, detailed feature extraction), enabling specialized processing of individual regions rather than full images
- **In multi-stage detection workflows** where initial detections are used to extract regions for secondary analysis (e.g., detect vehicles, crop each vehicle, detect license plates in crops), enabling hierarchical detection and analysis pipelines
- **Before visualization blocks** that display individual objects (e.g., display cropped objects separately, create galleries of detected objects, show isolated object annotations), enabling focused visualization of extracted regions
- **After detection blocks with instance segmentation** to create clean object crops with background removal, enabling isolated object images for analysis, training, or presentation
- **In keypoint detection workflows** where keypoint coordinates need to be preserved in cropped contexts (e.g., detect people with keypoints, crop each person, analyze pose in cropped images), enabling keypoint analysis in focused image regions
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
            "ui_manifest": {
                "section": "transformation",
                "icon": "far fa-crop-alt",
                "blockPriority": 0,
                "popular": True,
            },
        }
    )
    type: Literal["roboflow_core/dynamic_crop@v1", "DynamicCrop", "Crop"]
    images: Selector(kind=[IMAGE_KIND]) = Field(
        title="Image to Crop",
        description="Input image(s) to extract cropped regions from. Can be a single image or batch of images. Each image will be processed with corresponding detection predictions to extract bounding box regions. Cropped regions are extracted based on bounding boxes in the predictions. Can also accept previously cropped images from another Dynamic Crop step for nested cropping workflows.",
        examples=["$inputs.image", "$steps.cropping.crops"],
        validation_alias=AliasChoices("images", "image"),
    )
    predictions: Selector(
        kind=[
            OBJECT_DETECTION_PREDICTION_KIND,
            INSTANCE_SEGMENTATION_PREDICTION_KIND,
            KEYPOINT_DETECTION_PREDICTION_KIND,
        ]
    ) = Field(
        title="Regions of Interest",
        description="Detection model predictions containing bounding boxes that define regions to crop from the images. Supports object detection (bounding boxes), instance segmentation (bounding boxes with segmentation masks), or keypoint detection (bounding boxes with keypoints) predictions. Each bounding box in the predictions defines a rectangular region to extract. Predictions must include detection IDs for crop tracking. Multiple detections per image result in multiple crops per image.",
        examples=["$steps.my_object_detection_model.predictions"],
        validation_alias=AliasChoices("predictions", "detections"),
    )
    mask_opacity: Union[
        Selector(kind=[FLOAT_ZERO_TO_ONE_KIND]),
        float,
    ] = Field(
        default=0.0,
        le=1.0,
        ge=0.0,
        description="Background removal opacity for instance segmentation crops (0.0 to 1.0). Only applies when predictions contain segmentation masks (instance segmentation predictions). Controls how aggressively background pixels outside the detected instance are removed: 0.0 leaves the crop unchanged (no background removal), 1.0 fully replaces background with background_color, values between blend the original crop with the background. Higher values create cleaner object-focused crops. Set to 0.0 to disable background removal. Requires instance segmentation predictions with masks.",
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
        Selector(kind=[STRING_KIND]),
        Selector(kind=[RGB_COLOR_KIND]),
        str,
        Tuple[int, int, int],
    ] = Field(
        default=(0, 0, 0),
        description="Background color to use when removing background from instance segmentation crops. Only applies when mask_opacity > 0 and predictions contain segmentation masks. Background pixels outside the detected instance mask are replaced with this color. Can be specified as: hex string (e.g., '#431112' or '#fff'), RGB string in parentheses (e.g., '(128, 32, 64)'), or RGB tuple (e.g., (18, 17, 67)). Defaults to black (0, 0, 0). Use white (255, 255, 255) or '#ffffff' for white backgrounds, or match your use case's background requirements. Color values are interpreted as RGB and converted to BGR for image processing.",
        examples=["#431112", "$inputs.bg_color", (18, 17, 67)],
        json_schema_extra={
            "relevant_for": {
                "predictions": {
                    "kind": [INSTANCE_SEGMENTATION_PREDICTION_KIND.name],
                    "required": True,
                },
            }
        },
    )

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["images", "predictions"]

    @classmethod
    def get_output_dimensionality_offset(cls) -> int:
        return 1

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="crops", kind=[IMAGE_KIND]),
            OutputDefinition(
                name="predictions",
                kind=[
                    OBJECT_DETECTION_PREDICTION_KIND,
                    INSTANCE_SEGMENTATION_PREDICTION_KIND,
                    KEYPOINT_DETECTION_PREDICTION_KIND,
                ],
            ),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


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
) -> List[Dict[str, any]]:
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
        result = WorkflowImageData.create_crop(
            origin_image_data=image,
            crop_identifier=detection_id,
            cropped_image=cropped_image,
            offset_x=x_min,
            offset_y=y_min,
        )

        selected_detection = detections[idx]

        translated_detection = replace(
            selected_detection,
            xyxy=sv.move_boxes(xyxy=selected_detection.xyxy, offset=(-x_min, -y_min)),
            mask=(
                selected_detection.mask[:, y_min:y_max, x_min:x_max]
                if selected_detection.mask is not None
                else None
            ),
        )

        if KEYPOINTS_XY_KEY_IN_SV_DETECTIONS in detections.data:
            translated_detection[KEYPOINTS_XY_KEY_IN_SV_DETECTIONS] = (
                selected_detection[KEYPOINTS_XY_KEY_IN_SV_DETECTIONS]
                - np.array([x_min, y_min])
            )
        if POLYGON_KEY_IN_SV_DETECTIONS in detections.data:
            translated_detection[POLYGON_KEY_IN_SV_DETECTIONS] = selected_detection[
                POLYGON_KEY_IN_SV_DETECTIONS
            ] - np.array([x_min, y_min])

        crops.append(
            {
                "crops": result,
                # preserve all masks, keypoints, and metadata if present
                "predictions": translated_detection,
            }
        )
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
    color: Union[str, Tuple[int, int, int]],
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
