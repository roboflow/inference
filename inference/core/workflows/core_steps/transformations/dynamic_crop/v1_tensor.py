"""Tensor-native sibling of ``roboflow_core/dynamic_crop@v1``.

Under ENABLE_TENSOR_DATA_REPRESENTATION this block CONSUMES a native detection
prediction (``inference_models.Detections`` / ``InstanceDetections`` / the
keypoint-detection tuple ``Tuple[KeyPoints, Optional[Detections]]``) and PRODUCES
a batch of image crops (one per detection) plus, per crop, the SAME-shaped native
prediction translated into the crop's coordinate frame.

The numpy sibling (``.../dynamic_crop/v1.py``) does sv-shaped work that breaks on
native inputs:
- ``detections.data`` (detection-id contract check, keypoints/polygon/obb keys),
- ``detections.xyxy.round().astype`` (numpy on a torch tensor),
- ``detections.mask[idx]`` indexing / ``detections[idx]`` row slicing,
- ``dataclasses.replace`` + ``sv.move_boxes`` to translate the slice.

Here every one of those is reimplemented natively:
- ``xyxy`` is read off the torch tensor (rounded to int via torch);
- the image is cropped from ``image.tensor_image`` (CHW) like the
  ``absolute_static_crop`` tensor sibling;
- each per-detection prediction is sliced via ``take_prediction_by_indices``
  (preserves masks / per-box metadata / keypoints metadata) and then translated
  by ``(-x_min, -y_min)`` across every geometry field: ``xyxy`` (torch), dense /
  RLE masks (cropped to the box), the ``KeyPoints.xy`` tensor, and the flattened
  ``keypoints_xy`` / ``polygon`` / ``xyxyxyxy`` (OBB) entries in
  ``bboxes_metadata``.

The detection-id contract is enforced against ``bboxes_metadata[i][detection_id]``
(there is no ``.data``); the crop child is built with
``WorkflowImageData.create_crop_from_tensor`` so it stays on-device.

Only this file is created. The manifest is identical to the numpy sibling except
the input/output detection kinds are the ``TENSOR_NATIVE_*`` equivalents; the
tensor serialiser already handles native ``Detections`` / ``InstanceDetections``.
"""

from copy import deepcopy
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

import numpy as np
import torch
from pydantic import AliasChoices, ConfigDict, Field
from supervision.config import ORIENTED_BOX_COORDINATES

from inference.core.workflows.core_steps.common.tensor_native import (
    take_prediction_by_indices,
)
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
from inference.core.workflows.execution_engine.entities.tensor_native_types import (
    TENSOR_NATIVE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
    TENSOR_NATIVE_KEYPOINT_DETECTION_PREDICTION_KIND,
    TENSOR_NATIVE_OBJECT_DETECTION_PREDICTION_KIND,
)
from inference.core.workflows.execution_engine.entities.types import (
    FLOAT_ZERO_TO_ONE_KIND,
    IMAGE_KIND,
    RGB_COLOR_KIND,
    STRING_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)
from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.keypoints_detection import KeyPoints
from inference_models.models.base.object_detection import Detections
from inference_models.models.base.types import InstancesRLEMasks
from inference_models.models.common.rle_utils import (
    coco_rle_masks_to_numpy_mask,
    torch_mask_to_coco_rle,
)

# Native tensor-data input/output shapes. The keypoint-detection kind arrives as a
# Tuple[KeyPoints, Optional[Detections]]; both the KeyPoints and the bbox Detections
# components are sliced/translated consistently per crop.
TensorNativeDetections = Union[Detections, InstanceDetections]
KeyPointPrediction = Tuple[KeyPoints, Optional[Detections]]
TensorNativePrediction = Union[Detections, InstanceDetections, KeyPointPrediction]

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
            TENSOR_NATIVE_OBJECT_DETECTION_PREDICTION_KIND,
            TENSOR_NATIVE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
            TENSOR_NATIVE_KEYPOINT_DETECTION_PREDICTION_KIND,
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
                    "kind": [TENSOR_NATIVE_INSTANCE_SEGMENTATION_PREDICTION_KIND.name],
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
                    "kind": [TENSOR_NATIVE_INSTANCE_SEGMENTATION_PREDICTION_KIND.name],
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
                    TENSOR_NATIVE_OBJECT_DETECTION_PREDICTION_KIND,
                    TENSOR_NATIVE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
                    TENSOR_NATIVE_KEYPOINT_DETECTION_PREDICTION_KIND,
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
        predictions: Batch[TensorNativePrediction],
        mask_opacity: float,
        background_color: Union[str, Tuple[int, int, int]],
    ) -> BlockResult:
        return [
            crop_image(
                image=image,
                predictions=detections,
                mask_opacity=mask_opacity,
                background_color=background_color,
            )
            for image, detections in zip(images, predictions)
        ]


def crop_image(
    image: WorkflowImageData,
    predictions: TensorNativePrediction,
    mask_opacity: float,
    background_color: Union[str, Tuple[int, int, int]],
    detection_id_key: str = DETECTION_ID_KEY,
) -> List[Dict[str, Any]]:
    bbox_detections = _bbox_carrier(predictions)
    if bbox_detections is None or len(bbox_detections) == 0:
        return []
    bboxes_metadata = bbox_detections.bboxes_metadata
    if bboxes_metadata is None or any(
        detection_id_key not in (entry or {}) for entry in bboxes_metadata
    ):
        raise ValueError(
            f"Detections object passed to crop step do not fulfill contract - lack of "
            f"{detection_id_key} key in per-detection metadata."
        )
    # Round to int with torch, then materialise the (n, 4) corner ints to host once.
    xyxy_int = bbox_detections.xyxy.round().to(torch.int64).detach().to("cpu").numpy()
    # tensor_image is CHW; clamp box corners to the image bounds so a box that
    # extends past an edge does not slice with a negative index (torch, like numpy,
    # would treat a negative start as "from the end") and so the (-x_min, -y_min)
    # translation offset stays consistent with the actually-cropped region.
    image_height = int(image.tensor_image.shape[1])
    image_width = int(image.tensor_image.shape[2])
    crops: List[Dict[str, Any]] = []
    for idx in range(len(bbox_detections)):
        x_min, y_min, x_max, y_max = (int(v) for v in xyxy_int[idx])
        x_min = max(0, min(x_min, image_width))
        y_min = max(0, min(y_min, image_height))
        x_max = max(0, min(x_max, image_width))
        y_max = max(0, min(y_max, image_height))
        detection_id = bboxes_metadata[idx][detection_id_key]
        # tensor_image is CHW; crop on-device and skip empties (out-of-bounds boxes).
        cropped_tensor_image = image.tensor_image[:, y_min:y_max, x_min:x_max]
        if cropped_tensor_image.numel() == 0:
            crops.append({"crops": None, "predictions": None})
            continue
        cropped_tensor_image = cropped_tensor_image.contiguous()
        if (
            mask_opacity > 0
            and isinstance(bbox_detections, InstanceDetections)
            and bbox_detections.mask is not None
        ):
            cropped_tensor_image = _overlay_tensor_crop_with_mask(
                crop=cropped_tensor_image,
                detection_mask_2d=_instance_mask_bool_tensor(
                    detections=bbox_detections,
                    index=idx,
                    device=cropped_tensor_image.device,
                )[y_min:y_max, x_min:x_max],
                mask_opacity=mask_opacity,
                background_color=background_color,
            )
        result = WorkflowImageData.create_crop_from_tensor(
            origin_image_data=image,
            crop_identifier=detection_id,
            cropped_tensor_image=cropped_tensor_image,
            offset_x=x_min,
            offset_y=y_min,
        )
        translated_prediction = _translate_single_prediction(
            prediction=predictions,
            index=idx,
            x_min=x_min,
            y_min=y_min,
            x_max=x_max,
            y_max=y_max,
        )
        crops.append(
            {
                "crops": result,
                # preserve all masks, keypoints, and metadata if present
                "predictions": translated_prediction,
            }
        )
    return crops


def _bbox_carrier(
    prediction: TensorNativePrediction,
) -> Optional[TensorNativeDetections]:
    """Return the bounding-box ``Detections`` / ``InstanceDetections`` carrier for any
    supported native input. For the keypoint tuple the bbox component supplies xyxy and
    the per-box ``detection_id`` / metadata."""
    if isinstance(prediction, tuple):
        _, detections = prediction
        return detections
    return prediction


def _translate_single_prediction(
    prediction: TensorNativePrediction,
    index: int,
    x_min: int,
    y_min: int,
    x_max: int,
    y_max: int,
) -> TensorNativePrediction:
    """Slice the prediction down to a single detection ``index`` and translate every
    geometry field into the crop's coordinate frame (offset ``(-x_min, -y_min)``).

    Returns the SAME shape as the input (``Detections`` / ``InstanceDetections`` /
    keypoint tuple), mirroring the numpy block's ``replace(...)`` behaviour: xyxy is
    shifted, dense / RLE masks are cropped to the box, and the flattened
    ``keypoints_xy`` / ``polygon`` / OBB metadata plus the ``KeyPoints.xy`` tensor are
    offset.
    """
    single = take_prediction_by_indices(prediction=prediction, indices=[index])
    is_tuple = isinstance(single, tuple)
    key_points = None
    if is_tuple:
        key_points, detections = single
    else:
        detections = single
    detections = deepcopy(detections)
    offset = torch.tensor(
        [-x_min, -y_min, -x_min, -y_min],
        dtype=detections.xyxy.dtype,
        device=detections.xyxy.device,
    )
    detections.xyxy = detections.xyxy + offset
    if isinstance(detections, InstanceDetections) and detections.mask is not None:
        detections.mask = _crop_native_mask(
            mask=detections.mask, x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max
        )
    _offset_metadata_geometry(
        bboxes_metadata=detections.bboxes_metadata, x_min=x_min, y_min=y_min
    )
    if not is_tuple:
        return detections
    key_points = deepcopy(key_points)
    if key_points.xy.numel() > 0:
        kp_offset = torch.tensor(
            [-x_min, -y_min], dtype=key_points.xy.dtype, device=key_points.xy.device
        )
        key_points.xy = key_points.xy + kp_offset
    return key_points, detections


def _offset_metadata_geometry(
    bboxes_metadata: Optional[List[dict]],
    x_min: int,
    y_min: int,
) -> None:
    """Translate the flattened geometry entries that ride in per-box metadata:
    ``keypoints_xy`` (list of [x, y]), ``polygon`` (list of [x, y]), and the OBB
    ``xyxyxyxy`` corners. Numpy is used purely for the elementwise subtraction; values
    are written back in their original list/array container so the serialiser keeps
    working. Mirrors the numpy block's ``data``-key offsets (lines 251-267)."""
    if not bboxes_metadata:
        return
    offset_xy = np.array([x_min, y_min])
    for entry in bboxes_metadata:
        if not entry:
            continue
        if KEYPOINTS_XY_KEY_IN_SV_DETECTIONS in entry:
            entry[KEYPOINTS_XY_KEY_IN_SV_DETECTIONS] = _subtract_offset(
                entry[KEYPOINTS_XY_KEY_IN_SV_DETECTIONS], offset_xy
            )
        if POLYGON_KEY_IN_SV_DETECTIONS in entry:
            entry[POLYGON_KEY_IN_SV_DETECTIONS] = _subtract_offset(
                entry[POLYGON_KEY_IN_SV_DETECTIONS], offset_xy
            )
        if ORIENTED_BOX_COORDINATES in entry:
            entry[ORIENTED_BOX_COORDINATES] = _subtract_offset(
                entry[ORIENTED_BOX_COORDINATES], offset_xy
            )


def _subtract_offset(
    value: Union[List, np.ndarray], offset_xy: np.ndarray
) -> Union[List, np.ndarray]:
    """Subtract ``offset_xy`` ([x, y]) from a coordinate container, preserving whether
    the caller stored a python list (keypoints flattened by the keypoint producer) or a
    numpy array (sv-origin OBB / polygon). Integer coordinates stay integers after the
    shift (the flattened ``keypoints_xy`` entries should not silently become floats)."""
    if value is None:
        return value
    was_list = isinstance(value, list)
    array = np.asarray(value)
    if array.size == 0:
        return value
    # Preserve integer coordinates (e.g. keypoints stored as ints) so the shift
    # does not coerce them to float; non-integer inputs keep float precision.
    is_integer = np.issubdtype(array.dtype, np.integer)
    if is_integer:
        shifted = array.astype(np.int64) - offset_xy.astype(np.int64)
    else:
        shifted = array.astype(float) - offset_xy
    return shifted.tolist() if was_list else shifted


def _crop_native_mask(
    mask: Union[torch.Tensor, InstancesRLEMasks],
    x_min: int,
    y_min: int,
    x_max: int,
    y_max: int,
) -> Union[torch.Tensor, InstancesRLEMasks]:
    """Crop a single-instance native mask to the detection box, keeping the same carrier
    (dense torch -> dense torch; RLE -> RLE). Mirrors the numpy block's
    ``selected_detection.mask[:, y_min:y_max, x_min:x_max]`` slice."""
    if isinstance(mask, InstancesRLEMasks):
        # Decode the single instance, crop, re-encode as RLE for the cropped size.
        numpy_masks = coco_rle_masks_to_numpy_mask(mask)  # (1, H, W)
        cropped = numpy_masks[:, y_min:y_max, x_min:x_max]
        target_height = cropped.shape[1]
        target_width = cropped.shape[2]
        rle_masks = [
            torch_mask_to_coco_rle(torch.as_tensor(single_mask, dtype=torch.bool))[
                "counts"
            ]
            for single_mask in cropped
        ]
        return InstancesRLEMasks(
            image_size=(target_height, target_width), masks=rle_masks
        )
    return mask[:, y_min:y_max, x_min:x_max].contiguous()


def _instance_mask_bool_tensor(
    detections: InstanceDetections,
    index: int,
    device: torch.device,
) -> torch.Tensor:
    """Materialise a single instance's full-image mask as a 2-D bool ``torch.Tensor``
    ``(H, W)`` on ``device`` for the background-removal overlay. A dense torch mask
    stays on-device (no host round trip); RLE is decoded one instance at a time and
    moved to ``device`` (the RLE codec is numpy-only)."""
    mask = detections.mask
    if isinstance(mask, InstancesRLEMasks):
        single = coco_rle_masks_to_numpy_mask(
            InstancesRLEMasks(image_size=mask.image_size, masks=[mask.masks[index]])
        )[0].astype(bool)
        return torch.as_tensor(single, dtype=torch.bool, device=device)
    return mask[index].to(device=device, dtype=torch.bool)


def _overlay_tensor_crop_with_mask(
    crop: torch.Tensor,
    detection_mask_2d: torch.Tensor,
    mask_opacity: float,
    background_color: Union[str, Tuple[int, int, int]],
) -> torch.Tensor:
    """Background-removal overlay for tensor crops, computed entirely on the crop's
    device. Equivalent to the numpy block's ``cv2.addWeighted`` blend, but expressed
    as a ``torch.where`` + lerp so a GPU pipeline never pays a per-crop host round trip.

    Inside the instance mask the crop is kept verbatim; outside it the pixel is faded
    toward ``background_color`` by ``mask_opacity`` (``mask_opacity * bg +
    (1 - mask_opacity) * crop``), matching the numpy result for both regions.

    ``crop`` is CHW RGB uint8; ``detection_mask_2d`` is a (H, W) bool tensor already
    sliced to the crop box, both on the same device."""
    device = crop.device
    # convert_color_to_bgr_tuple yields BGR; the crop tensor is RGB, so reverse it.
    bgr_color = convert_color_to_bgr_tuple(color=background_color)
    rgb_color = tuple(bgr_color[::-1])
    bg = torch.tensor(rgb_color, dtype=torch.float32, device=device).reshape(3, 1, 1)
    crop_float = crop.to(dtype=torch.float32)
    faded = mask_opacity * bg + (1.0 - mask_opacity) * crop_float
    mask_3c = detection_mask_2d.to(device=device, dtype=torch.bool).unsqueeze(0)
    overlaid = torch.where(mask_3c, crop_float, faded)
    return overlaid.round().clamp_(0, 255).to(dtype=crop.dtype).contiguous()


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
