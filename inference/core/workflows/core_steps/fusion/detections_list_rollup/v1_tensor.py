import uuid
from typing import Any, List, Literal, Optional, Tuple, Type, Union

import numpy as np
import torch
from pydantic import ConfigDict, Field

from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.object_detection import Detections

from inference.core.workflows.core_steps.common.tensor_native import (
    instance_mask_to_numpy,
    split_key_point_prediction,
)
from inference.core.workflows.execution_engine.constants import (
    CLASS_NAMES_KEY,
    IMAGE_DIMENSIONS_KEY,
    INFERENCE_ID_KEY,
    KEYPOINTS_CLASS_ID_KEY_IN_SV_DETECTIONS,
    KEYPOINTS_CLASS_NAME_KEY_IN_SV_DETECTIONS,
    KEYPOINTS_CONFIDENCE_KEY_IN_SV_DETECTIONS,
    KEYPOINTS_XY_KEY_IN_SV_DETECTIONS,
    PARENT_COORDINATES_KEY,
    PARENT_DIMENSIONS_KEY,
    PARENT_ID_KEY,
    PREDICTION_TYPE_KEY,
    ROOT_PARENT_COORDINATES_KEY,
    ROOT_PARENT_DIMENSIONS_KEY,
    ROOT_PARENT_ID_KEY,
)
from inference.core.workflows.execution_engine.entities.base import OutputDefinition
from inference.core.workflows.execution_engine.entities.tensor_native_types import (
    TENSOR_NATIVE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
    TENSOR_NATIVE_KEYPOINT_DETECTION_PREDICTION_KIND,
    TENSOR_NATIVE_OBJECT_DETECTION_PREDICTION_KIND,
)
from inference.core.workflows.execution_engine.entities.types import (
    FLOAT_KIND,
    FLOAT_ZERO_TO_ONE_KIND,
    LIST_OF_VALUES_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

LONG_DESCRIPTION = """
Rolls up dimensionality from children to parent detections

Useful in scenarios like:
* rolling up results from a secondary model run on crops back to parent images
* rolling up OCR results for dynamically cropped images
"""

SHORT_DESCRIPTION = (
    "Roll up multiple levels of dimensionality back to a single dimension."
)

# Per-image metadata fields. In tensor-native predictions these live in
# ``image_metadata`` (one value per prediction) rather than being duplicated
# across every detection in ``bboxes_metadata``. They are read from / written to
# ``image_metadata`` while the rollup machinery keeps treating them as scalar
# "data" values for backward-compatible merge behaviour.
_PER_IMAGE_METADATA_KEYS = (
    PREDICTION_TYPE_KEY,
    PARENT_ID_KEY,
    INFERENCE_ID_KEY,
    ROOT_PARENT_ID_KEY,
    IMAGE_DIMENSIONS_KEY,
    PARENT_DIMENSIONS_KEY,
    PARENT_COORDINATES_KEY,
    ROOT_PARENT_DIMENSIONS_KEY,
    ROOT_PARENT_COORDINATES_KEY,
)

# Per-detection keypoint fields live in ``bboxes_metadata`` natively (same keys
# the serialiser reads back) — mirrors the legacy ``sv.Detections.data`` keys.
_KEYPOINT_DATA_KEYS = (
    KEYPOINTS_XY_KEY_IN_SV_DETECTIONS,
    KEYPOINTS_CLASS_NAME_KEY_IN_SV_DETECTIONS,
    KEYPOINTS_CLASS_ID_KEY_IN_SV_DETECTIONS,
    KEYPOINTS_CONFIDENCE_KEY_IN_SV_DETECTIONS,
)


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Detections List Roll-Up",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "fusion",
            "ui_manifest": {
                "section": "advanced",
                "icon": "fal fa-bring-front",
                "blockPriority": 6,
            },
        }
    )
    type: Literal["roboflow_core/detections_list_rollup@v1", "DetectionsListRollUp"]
    parent_detection: Selector(
        kind=[
            TENSOR_NATIVE_OBJECT_DETECTION_PREDICTION_KIND,
            TENSOR_NATIVE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
            TENSOR_NATIVE_KEYPOINT_DETECTION_PREDICTION_KIND,
        ]
    ) = Field(
        description="The parent detection the dimensionality inherits from.",
    )

    child_detections: Selector(kind=[LIST_OF_VALUES_KIND]) = Field(
        description="A list of child detections resulting from higher dimensionality,"
        ' such as predictions made on dynamic crops. Use the "Dimension Collapse" to '
        " reduce the higher dimensionality result to one that can be used with this."
        " Example: Prediction -> Dimension Collapse -> Detections List Roll-Up",
    )

    confidence_strategy: Union[
        Selector(kind=[LIST_OF_VALUES_KIND]), Literal["max", "mean", "min"]
    ] = Field(
        default="max",
        title="Confidence Strategy",
        description=(
            "Strategy to use when merging confidence scores from child detections. "
            "Options are 'max', 'mean', or 'min'."
        ),
        examples=["min", "mean", "max"],
    )

    overlap_threshold: Union[Selector(kind=[FLOAT_ZERO_TO_ONE_KIND]), float] = Field(
        default=0.0,
        title="Overlap Threshold",
        description=(
            "Minimum overlap ratio (IoU) to consider when merging overlapping "
            "detections from child crops. "
            "A value of 0.0 merges any overlapping detections, while higher values "
            "require greater overlap to merge. Specify between 0.0 and 1.0. A value of 1.0 "
            "only merges completely overlapping detections."
        ),
        examples=[0.0, 0.5],
        ge=0.0,
        le=1.0,
    )

    keypoint_merge_threshold: Union[Selector(kind=[FLOAT_KIND]), float] = Field(
        default=10.0,
        title="Keypoint Merge Threshold",
        description=(
            "Keypoint distance (in pixels) to merge keypoint detections if the child detections contain keypoint data."
        ),
        examples=[0.0, 20.0],
        ge=0.0,
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name="rolled_up_detections",
                kind=[
                    TENSOR_NATIVE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
                    TENSOR_NATIVE_OBJECT_DETECTION_PREDICTION_KIND,
                    TENSOR_NATIVE_KEYPOINT_DETECTION_PREDICTION_KIND,
                ],
            ),
            OutputDefinition(
                name="crop_zones",
                kind=[LIST_OF_VALUES_KIND],
            ),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class DetectionsListRollUpBlockV1(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        parent_detection: Any,
        child_detections: Any,
        confidence_strategy: str = "max",
        overlap_threshold: float = 0.0,
        keypoint_merge_threshold: float = 10.0,
    ) -> BlockResult:

        detections, zones = merge_crop_predictions(
            parent_detection,
            child_detections,
            confidence_strategy,
            overlap_threshold,
            keypoint_merge_threshold,
        )

        return {"rolled_up_detections": detections, "crop_zones": zones}


def _native_box_metadata(prediction) -> List[dict]:
    """Per-detection metadata list for a tensor-native bbox prediction, always
    of length equal to the number of detections (``{}`` where absent)."""
    number_of_detections = int(prediction.xyxy.shape[0])
    bboxes_metadata = prediction.bboxes_metadata
    if bboxes_metadata is None:
        return [{} for _ in range(number_of_detections)]
    return [
        dict(box_metadata) if box_metadata is not None else {}
        for box_metadata in bboxes_metadata
    ]


def _merge_keypoint_detections(
    preds: List[dict], confidence_strategy: str, keypoint_threshold: float
) -> List[dict]:
    """
    Merge keypoint detections based on keypoint proximity.

    Args:
        preds: List of prediction dicts with 'bbox', 'confidence', 'class_id', 'keypoint_data'
        confidence_strategy: How to combine confidences ('max', 'mean', 'min')
        keypoint_threshold: Maximum average keypoint distance (in pixels) to merge detections

    Returns:
        List of merged prediction dicts
    """
    if not preds:
        return []

    # Filter predictions that have keypoint data
    preds_with_keypoints = [
        p
        for p in preds
        if p.get("keypoint_data") and "keypoints_xy" in p["keypoint_data"]
    ]
    preds_without_keypoints = [
        p
        for p in preds
        if not (p.get("keypoint_data") and "keypoints_xy" in p["keypoint_data"])
    ]

    if not preds_with_keypoints:
        return preds

    merged = []
    used = set()

    for i, pred1 in enumerate(preds_with_keypoints):
        if i in used:
            continue

        # Start a new merged group with this prediction
        group = [pred1]
        used.add(i)

        kp1 = np.array(pred1["keypoint_data"]["keypoints_xy"])

        # Find all predictions that should merge with this one
        for j, pred2 in enumerate(preds_with_keypoints[i + 1 :], start=i + 1):
            if j in used:
                continue

            kp2 = np.array(pred2["keypoint_data"]["keypoints_xy"])

            # Calculate average distance between corresponding keypoints
            if len(kp1) == len(kp2):
                distances = np.linalg.norm(kp1 - kp2, axis=1)
                avg_distance = np.mean(distances)

                if avg_distance < keypoint_threshold:
                    group.append(pred2)
                    used.add(j)

        # Merge the group
        if len(group) == 1:
            merged.append(group[0])
        else:
            # Merge multiple predictions
            if confidence_strategy == "max":
                best_idx = np.argmax([p["confidence"] for p in group])
                confidence = group[best_idx]["confidence"]
            elif confidence_strategy == "mean":
                confidence = np.mean([p["confidence"] for p in group])
            else:  # 'min'
                confidence = np.min([p["confidence"] for p in group])

            # Average keypoint coordinates
            all_kp_xy = [np.array(p["keypoint_data"]["keypoints_xy"]) for p in group]
            merged_kp_xy = np.mean(all_kp_xy, axis=0).tolist()

            # Average keypoint confidences if available
            merged_kp_data = {
                "keypoints_xy": merged_kp_xy,
                "keypoints_class_name": group[0]["keypoint_data"].get(
                    "keypoints_class_name"
                ),
                "keypoints_class_id": group[0]["keypoint_data"].get(
                    "keypoints_class_id"
                ),
            }

            if "keypoints_confidence" in group[0]["keypoint_data"]:
                all_kp_conf = [
                    np.array(p["keypoint_data"]["keypoints_confidence"]) for p in group
                ]
                merged_kp_conf = np.mean(all_kp_conf, axis=0).tolist()
                merged_kp_data["keypoints_confidence"] = merged_kp_conf

            # Average bbox coordinates
            all_bboxes = np.array([p["bbox"] for p in group])
            merged_bbox = np.mean(all_bboxes, axis=0)

            merged.append(
                {
                    "bbox": merged_bbox,
                    "confidence": confidence,
                    "class_id": group[0]["class_id"],
                    "mask": None,
                    "keypoint_data": merged_kp_data,
                    "detection_data": group[0].get(
                        "detection_data", {}
                    ),  # Preserve first detection's metadata
                }
            )

    # Add back predictions without keypoints
    merged.extend(preds_without_keypoints)

    return merged


def merge_crop_predictions(
    parent_prediction,
    child_predictions: List,
    confidence_strategy: str = "max",
    overlap_threshold: float = 0.0,
    keypoint_merge_threshold: float = 10.0,
) -> Tuple:
    """
    Merge predictions from multiple crops back to parent image coordinates.

    Args:
        parent_prediction: Supervision Detections object that defines the crop locations.
                          Each detection in this prediction represents one crop region.
        child_predictions: List of Supervision Detections objects from crops.
                          Order matches the detection order in parent_prediction.
        confidence_strategy: How to handle confidence when merging overlaps.
                           Options: "max", "mean", "min"
        overlap_threshold: Minimum IoU/overlap ratio to merge detections (0.0 to 1.0).
                         - 0.0: Only merge if detections touch or overlap at all (default)
                         - >0.0: Only merge if overlap ratio exceeds this threshold
                         - 1.0: Only merge completely overlapping detections
        keypoint_merge_threshold: Maximum distance in pixels to merge keypoints (default: 10).
                                For keypoint detections, merges detections if their average
                                keypoint distance is below this threshold.

    Returns:
        Tuple of (detections, crop_zones):
        - detections: Detections object with merged predictions in parent image coordinates.
                     Works for both instance segmentation (with masks) and object detection (without masks).
        - crop_zones: List of lists of (x, y) tuples. Each inner list defines the rectangular
                     zone boundary of a crop in parent image coordinates as 4 corner points.
    """
    # Keypoint predictions arrive as a (KeyPoints, Detections) tuple; the rollup
    # only needs the bbox component (parent crop boxes / child boxes + masks +
    # per-detection metadata). Keypoints themselves are carried in the child
    # bboxes_metadata (keypoints_* keys), exactly like the legacy data dict.
    _parent_key_points, parent_prediction = split_key_point_prediction(
        parent_prediction
    )
    child_predictions = [
        split_key_point_prediction(child_pred)[1] for child_pred in child_predictions
    ]

    if len(parent_prediction) != len(child_predictions):
        raise ValueError(
            f"Number of detections in parent_prediction ({len(parent_prediction)}) "
            f"must match number of child predictions ({len(child_predictions)})"
        )

    # Extract parent image shape from parent prediction's image_metadata.
    # root_parent_dimensions is a single per-image value (height, width) for the
    # tensor-native prediction (numpy stored it once per detection).
    parent_image_metadata = parent_prediction.image_metadata or {}
    root_parent_dims = parent_image_metadata.get(ROOT_PARENT_DIMENSIONS_KEY)

    if root_parent_dims is None or len(root_parent_dims) == 0:
        raise ValueError(
            "parent_prediction must have 'root_parent_dimensions' in its data attribute"
        )

    # The per-image value is already the (height, width) for the parent image.
    parent_image_shape = tuple(root_parent_dims)

    # Pre-read per-detection metadata + per-image metadata for every child once.
    child_box_metadata = [
        _native_box_metadata(child_pred) for child_pred in child_predictions
    ]
    child_image_metadata = [
        (child_pred.image_metadata or {}) for child_pred in child_predictions
    ]
    parent_xyxy = parent_prediction.xyxy.detach().to("cpu").numpy()

    # Merge the class_id -> name maps across all children so the rolled-up
    # native prediction can carry a single image_metadata["class_names"] map
    # (numpy stored class_name per detection instead). Keys are normalized to
    # python int so map lookups never miss on a numpy-scalar class_id.
    merged_class_names: dict = {}
    for image_metadata in child_image_metadata:
        child_class_names = image_metadata.get(CLASS_NAMES_KEY)
        if child_class_names:
            for class_id_key, class_name in child_class_names.items():
                merged_class_names[int(class_id_key)] = class_name

    # Build crop zones list - one zone per crop/child prediction
    crop_zones = []
    for i in range(len(parent_prediction)):
        crop_bbox = parent_xyxy[i]  # [x_min, y_min, x_max, y_max]
        x_min, y_min, x_max, y_max = (
            crop_bbox[0],
            crop_bbox[1],
            crop_bbox[2],
            crop_bbox[3],
        )

        # Create zone as list of 4 corner points: top-left, top-right, bottom-right, bottom-left
        zone = [
            (float(x_min), float(y_min)),  # top-left
            (float(x_max), float(y_min)),  # top-right
            (float(x_max), float(y_max)),  # bottom-right
            (float(x_min), float(y_max)),  # bottom-left
        ]
        crop_zones.append(zone)

    # Check if we have instance segmentation (with masks) or object detection (without masks)
    has_masks = False
    is_keypoint_detection = False
    for child_pred in child_predictions:
        if (
            isinstance(child_pred, InstanceDetections)
            and child_pred.mask is not None
            and len(child_pred) > 0
        ):
            has_masks = True
            break

    for image_metadata in child_image_metadata:
        # Check for keypoint detection. Native image_metadata stores
        # prediction_type as a scalar string (build_native_image_metadata),
        # so no np.ndarray legacy form is possible here.
        if PREDICTION_TYPE_KEY in image_metadata:
            if image_metadata[PREDICTION_TYPE_KEY] == "keypoint-detection":
                is_keypoint_detection = True
                break

    # Group predictions by class
    class_predictions = {}

    # Iterate through each crop region and its corresponding child predictions
    for i, child_pred in enumerate(child_predictions):
        box_metadata = child_box_metadata[i]
        # Get crop location from parent prediction
        crop_bbox = parent_xyxy[i]  # [x_min, y_min, x_max, y_max]
        x_min, y_min = int(crop_bbox[0]), int(crop_bbox[1])

        child_class_ids = child_pred.class_id.detach().to("cpu").numpy()
        child_confidences = child_pred.confidence.detach().to("cpu").numpy()
        child_xyxy = child_pred.xyxy.detach().to("cpu").numpy()
        child_has_masks = (
            isinstance(child_pred, InstanceDetections) and child_pred.mask is not None
        )

        # Process each detection in the child prediction
        for j in range(len(child_pred)):
            detection_metadata = box_metadata[j]
            class_id = child_class_ids[j]
            confidence = child_confidences[j]

            # Prepare keypoint data if present
            keypoint_data = {}
            if (
                is_keypoint_detection
                and KEYPOINTS_XY_KEY_IN_SV_DETECTIONS in detection_metadata
            ):
                # Transform keypoint coordinates from crop to parent space
                keypoints_xy = detection_metadata[
                    KEYPOINTS_XY_KEY_IN_SV_DETECTIONS
                ]  # Shape: (num_keypoints, 2)

                # Vectorised offset — avoids a per-keypoint Python loop.
                # copy=True prevents mutating the source array; reshape handles
                # edge cases where keypoints_xy arrives as a flat/empty array.
                kp_array = np.array(keypoints_xy, dtype=np.float64, copy=True)
                if kp_array.size == 0:
                    keypoint_data["keypoints_xy"] = []
                else:
                    kp_array = kp_array.reshape(-1, 2)
                    kp_array = kp_array + np.array([x_min, y_min], dtype=np.float64)
                    keypoint_data["keypoints_xy"] = kp_array.tolist()

                # Copy other keypoint data
                if KEYPOINTS_CLASS_NAME_KEY_IN_SV_DETECTIONS in detection_metadata:
                    keypoint_data["keypoints_class_name"] = detection_metadata[
                        KEYPOINTS_CLASS_NAME_KEY_IN_SV_DETECTIONS
                    ]
                if KEYPOINTS_CLASS_ID_KEY_IN_SV_DETECTIONS in detection_metadata:
                    keypoint_data["keypoints_class_id"] = detection_metadata[
                        KEYPOINTS_CLASS_ID_KEY_IN_SV_DETECTIONS
                    ]
                if KEYPOINTS_CONFIDENCE_KEY_IN_SV_DETECTIONS in detection_metadata:
                    keypoint_data["keypoints_confidence"] = detection_metadata[
                        KEYPOINTS_CONFIDENCE_KEY_IN_SV_DETECTIONS
                    ]

            # Collect per-detection data fields to preserve individual detection metadata
            # This is crucial for preserving class_name and other fields when multiple
            # detections have the same class_id but different values
            detection_data = {}
            for key in detection_metadata.keys():
                if key not in [
                    "detection_id",
                    "parent_id",
                    "inference_id",
                    "keypoints_xy",
                    "keypoints_class_name",
                    "keypoints_class_id",
                    "keypoints_confidence",
                ]:
                    detection_data[key] = detection_metadata[key]

            if has_masks and child_has_masks:
                # Instance segmentation - transform mask
                mask = instance_mask_to_numpy(child_pred, j)
                transformed_mask = _transform_mask_to_parent(
                    mask, x_min, y_min, parent_image_shape
                )

                # Also store the transformed bbox for cheap pre-filtering
                raw_bbox = child_xyxy[j]
                transformed_bbox = np.array(
                    [
                        raw_bbox[0] + x_min,
                        raw_bbox[1] + y_min,
                        raw_bbox[2] + x_min,
                        raw_bbox[3] + y_min,
                    ]
                )

                # Store prediction with transformed mask
                if class_id not in class_predictions:
                    class_predictions[class_id] = []

                class_predictions[class_id].append(
                    {
                        "mask": transformed_mask,
                        "confidence": confidence,
                        "class_id": class_id,
                        "bbox": transformed_bbox,
                        "keypoint_data": keypoint_data,
                        "detection_data": detection_data,  # Store per-detection metadata
                    }
                )
            else:
                # Object detection - transform bounding box
                bbox = child_xyxy[j]  # [x_min, y_min, x_max, y_max]
                transformed_bbox = np.array(
                    [bbox[0] + x_min, bbox[1] + y_min, bbox[2] + x_min, bbox[3] + y_min]
                )

                # Store prediction with transformed bbox
                if class_id not in class_predictions:
                    class_predictions[class_id] = []

                class_predictions[class_id].append(
                    {
                        "bbox": transformed_bbox,
                        "confidence": confidence,
                        "class_id": class_id,
                        "mask": None,
                        "keypoint_data": keypoint_data,
                        "detection_data": detection_data,  # Store per-detection metadata
                    }
                )

    # Merge overlapping predictions for each class
    merged_masks = []
    merged_bboxes = []
    merged_confidences = []
    merged_class_ids = []

    # Collect all data field names from child predictions. Tensor-native
    # per-detection keys live in bboxes_metadata; the per-image fields below
    # (prediction_type, parent/root ids+coords+dims, inference_id) are added
    # explicitly so the rollup re-creates them on the output exactly as the
    # numpy block did from sv.Detections.data.
    all_data_keys = set()
    for box_metadata in child_box_metadata:
        for detection_metadata in box_metadata:
            all_data_keys.update(detection_metadata.keys())
    for image_metadata in child_image_metadata:
        for key in _PER_IMAGE_METADATA_KEYS:
            if key in image_metadata:
                all_data_keys.add(key)

    # Initialize lists for each data field
    merged_data = {
        key: []
        for key in all_data_keys
        if key
        not in [
            "keypoints_xy",
            "keypoints_class_name",
            "keypoints_class_id",
            "keypoints_confidence",
        ]
    }

    # Collect keypoint data separately
    all_keypoints_data = {
        "keypoints_xy": [],
        "keypoints_class_name": [],
        "keypoints_class_id": [],
        "keypoints_confidence": [],
    }

    # Build mapping from class_id to typical data values
    class_id_to_data = {}
    for i, child_pred in enumerate(child_predictions):
        box_metadata = child_box_metadata[i]
        image_metadata = child_image_metadata[i]
        child_class_ids = child_pred.class_id.detach().to("cpu").numpy()
        for index in range(len(child_pred)):
            detection_metadata = box_metadata[index]
            class_id = child_class_ids[index]
            if class_id not in class_id_to_data:
                class_id_to_data[class_id] = {}
                # Store sample values for this class_id (except ID fields and keypoint fields)
                for key in detection_metadata.keys():
                    if key not in [
                        "detection_id",
                        "parent_id",
                        "inference_id",
                        "keypoints_xy",
                        "keypoints_class_name",
                        "keypoints_class_id",
                        "keypoints_confidence",
                    ]:
                        class_id_to_data[class_id][key] = detection_metadata[key]
                # Per-image fields are shared across all detections of this child
                for key in _PER_IMAGE_METADATA_KEYS:
                    if key in image_metadata and key not in [
                        "parent_id",
                        "inference_id",
                    ]:
                        class_id_to_data[class_id][key] = image_metadata[key]

    # Get a sample inference_id and parent_id from the first child prediction if available
    sample_inference_id = None
    sample_parent_id = None
    if len(child_predictions) > 0 and len(child_predictions[0]) > 0:
        first_image_metadata = child_image_metadata[0]
        if INFERENCE_ID_KEY in first_image_metadata:
            sample_inference_id = first_image_metadata[INFERENCE_ID_KEY]
        if PARENT_ID_KEY in first_image_metadata:
            sample_parent_id = first_image_metadata[PARENT_ID_KEY]

    for class_id, preds in class_predictions.items():
        if is_keypoint_detection:
            # For keypoint detection, merge based on keypoint proximity
            merged_preds = _merge_keypoint_detections(
                preds, confidence_strategy, keypoint_merge_threshold
            )
        elif has_masks:
            merged_preds = _merge_overlapping_masks(
                preds, confidence_strategy, overlap_threshold
            )
        else:
            merged_preds = _merge_overlapping_bboxes(
                preds, confidence_strategy, overlap_threshold
            )

        for pred in merged_preds:
            if has_masks:
                merged_masks.append(pred["mask"])
            else:
                # For non-mask detections, collect bboxes
                if "bbox" in pred and pred["bbox"] is not None:
                    merged_bboxes.append(pred["bbox"])
            merged_confidences.append(pred["confidence"])
            merged_class_ids.append(pred["class_id"])

            # Collect keypoint data if present
            if "keypoint_data" in pred and pred["keypoint_data"]:
                kp_data = pred["keypoint_data"]
                all_keypoints_data["keypoints_xy"].append(kp_data.get("keypoints_xy"))
                all_keypoints_data["keypoints_class_name"].append(
                    kp_data.get("keypoints_class_name")
                )
                all_keypoints_data["keypoints_class_id"].append(
                    kp_data.get("keypoints_class_id")
                )
                all_keypoints_data["keypoints_confidence"].append(
                    kp_data.get("keypoints_confidence")
                )

            # Add data fields for this detection
            for key in all_data_keys:
                # Skip keypoint fields as they're handled separately
                if key in [
                    "keypoints_xy",
                    "keypoints_class_name",
                    "keypoints_class_id",
                    "keypoints_confidence",
                ]:
                    continue

                if key == "detection_id":
                    # Generate new UUID for merged detection
                    merged_data[key].append(str(uuid.uuid4()))
                elif key == "parent_id":
                    # Use sample parent_id or generate new one
                    merged_data[key].append(
                        sample_parent_id if sample_parent_id else str(uuid.uuid4())
                    )
                elif key == "inference_id":
                    # Use the same inference_id as inputs (they're from same inference batch)
                    merged_data[key].append(
                        sample_inference_id
                        if sample_inference_id
                        else str(uuid.uuid4())
                    )
                elif key == "root_parent_dimensions":
                    # Add the parent image shape as a list [height, width]
                    merged_data[key].append(list(parent_image_shape))
                elif key == "parent_dimensions":
                    # Parent dimensions should be same as root_parent_dimensions for merged results
                    merged_data[key].append(list(parent_image_shape))
                elif key == "image_dimensions":
                    # Image dimensions for this detection
                    merged_data[key].append(list(parent_image_shape))
                elif key == "root_parent_coordinates":
                    # Root parent coordinates [y, x] - should be [0, 0] for the root
                    if (
                        pred["class_id"] in class_id_to_data
                        and key in class_id_to_data[pred["class_id"]]
                    ):
                        merged_data[key].append(class_id_to_data[pred["class_id"]][key])
                    else:
                        merged_data[key].append([0, 0])
                elif key == "parent_coordinates":
                    # Parent coordinates [y, x]
                    if (
                        pred["class_id"] in class_id_to_data
                        and key in class_id_to_data[pred["class_id"]]
                    ):
                        merged_data[key].append(class_id_to_data[pred["class_id"]][key])
                    else:
                        merged_data[key].append([0, 0])
                elif key == "root_parent_id":
                    # Root parent ID
                    if (
                        pred["class_id"] in class_id_to_data
                        and key in class_id_to_data[pred["class_id"]]
                    ):
                        merged_data[key].append(class_id_to_data[pred["class_id"]][key])
                    else:
                        merged_data[key].append("image")
                elif key == "prediction_type":
                    # Prediction type should be 'instance-segmentation'
                    merged_data[key].append("instance-segmentation")
                else:
                    # For other fields like class_name, check pred dict first (per-detection data)
                    # then fall back to class_id_to_data (class-level defaults)
                    if key in pred.get("detection_data", {}):
                        merged_data[key].append(pred["detection_data"][key])
                    elif (
                        pred["class_id"] in class_id_to_data
                        and key in class_id_to_data[pred["class_id"]]
                    ):
                        merged_data[key].append(class_id_to_data[pred["class_id"]][key])
                    else:
                        merged_data[key].append(None)

    if not merged_confidences:
        # Return empty detections if no detections
        return _empty_native_detections(has_masks, merged_class_names), crop_zones

    # Convert to numpy arrays
    merged_confidences_array = np.array(merged_confidences, dtype=np.float32)
    merged_class_ids_array = np.array(merged_class_ids, dtype=int)

    if has_masks:
        # Instance segmentation - stack masks and compute bounding boxes
        merged_masks_array = np.stack(merged_masks, axis=0)

        # Compute bounding boxes from masks
        xyxy = []
        for mask in merged_masks_array:
            rows, cols = np.where(mask)
            if len(rows) > 0:
                x_min, x_max = cols.min(), cols.max()
                y_min, y_max = rows.min(), rows.max()
                xyxy.append([x_min, y_min, x_max + 1, y_max + 1])
            else:
                xyxy.append([0, 0, 0, 0])

        xyxy_array = np.array(xyxy, dtype=np.float32)
    else:
        # Object detection - use bounding boxes directly
        if merged_bboxes:
            xyxy_array = np.array(merged_bboxes, dtype=np.float32)
        else:
            # Shouldn't happen, but handle edge case
            xyxy_array = np.zeros((len(merged_confidences), 4), dtype=np.float32)

    # Route the merged data fields into native per-image (image_metadata) and
    # per-detection (bboxes_metadata) state instead of a flat sv.Detections.data
    # dict. Per-image fields are written once (taken from the first merged value,
    # which is shared across all rolled-up detections); per-detection fields are
    # written per row. class_name is not stored per-detection natively — names
    # are resolved from class_id via image_metadata["class_names"].
    number_of_detections = len(merged_confidences)
    image_metadata: dict = {}
    # Always carry a class_names map covering every merged class_id so the
    # serializer's per-row class_id lookup never fails (e.g. the OCR / map-less
    # rollup path, where children never declared a class_names map). Names come
    # from the unioned child maps; any class_id without an entry falls back to
    # f"class_{id}". Keys are normalized to python int to match map lookups.
    output_class_names: dict = {}
    for class_id_value in merged_class_ids:
        class_id_int = int(class_id_value)
        output_class_names[class_id_int] = merged_class_names.get(
            class_id_int, f"class_{class_id_int}"
        )
    image_metadata[CLASS_NAMES_KEY] = output_class_names
    bboxes_metadata: List[dict] = [{} for _ in range(number_of_detections)]

    for key, values in merged_data.items():
        if key in _PER_IMAGE_METADATA_KEYS:
            # Per-image field: store once in image_metadata (values are identical
            # across rows by construction above).
            if values:
                image_metadata[key] = _coerce_metadata_value(key, values[0])
        else:
            # Per-detection field: store per row in bboxes_metadata.
            for index in range(number_of_detections):
                bboxes_metadata[index][key] = _coerce_metadata_value(key, values[index])

    # Add keypoint data if it exists (per-detection, into bboxes_metadata).
    if is_keypoint_detection:
        for key in _KEYPOINT_DATA_KEYS:
            values = all_keypoints_data[key]
            if values:
                for index in range(number_of_detections):
                    bboxes_metadata[index][key] = values[index]

    if has_masks:
        result = InstanceDetections(
            xyxy=torch.as_tensor(xyxy_array, dtype=torch.float32).reshape(-1, 4),
            mask=torch.from_numpy(merged_masks_array).to(torch.bool),
            confidence=torch.as_tensor(merged_confidences_array, dtype=torch.float32),
            class_id=torch.as_tensor(merged_class_ids_array, dtype=torch.long),
            image_metadata=image_metadata,
            bboxes_metadata=bboxes_metadata,
        )
    else:
        result = Detections(
            xyxy=torch.as_tensor(xyxy_array, dtype=torch.float32).reshape(-1, 4),
            confidence=torch.as_tensor(merged_confidences_array, dtype=torch.float32),
            class_id=torch.as_tensor(merged_class_ids_array, dtype=torch.long),
            image_metadata=image_metadata,
            bboxes_metadata=bboxes_metadata,
        )

    return result, crop_zones


def _coerce_metadata_value(key: str, value: Any) -> Any:
    """Coerce a merged metadata value into the dtype the numpy block used when
    writing into sv.Detections.data, but as a plain python scalar/list (native
    per-detection / per-image state is python, not numpy arrays)."""
    if value is None:
        return value
    if key in [
        "prediction_type",
        "detection_id",
        "parent_id",
        "inference_id",
        "root_parent_id",
    ]:
        # String fields. class_name is intentionally omitted: names are never
        # stored per-detection natively (resolved from class_id via the
        # image_metadata["class_names"] map).
        return str(value)
    if key in [
        "root_parent_dimensions",
        "parent_dimensions",
        "image_dimensions",
        "root_parent_coordinates",
        "parent_coordinates",
    ]:
        # Array/coordinate fields - integers
        return np.array(value, dtype=int).tolist()
    return value


def _empty_native_detections(
    has_masks: bool, class_names: dict
) -> Union[Detections, InstanceDetections]:
    """Build an empty tensor-native prediction (mirrors sv.Detections.empty())."""
    image_metadata = {CLASS_NAMES_KEY: class_names} if class_names else {}
    if has_masks:
        return InstanceDetections(
            xyxy=torch.zeros((0, 4), dtype=torch.float32),
            class_id=torch.zeros((0,), dtype=torch.long),
            confidence=torch.zeros((0,), dtype=torch.float32),
            mask=torch.zeros((0, 0, 0), dtype=torch.bool),
            image_metadata=image_metadata,
            bboxes_metadata=None,
        )
    return Detections(
        xyxy=torch.zeros((0, 4), dtype=torch.float32),
        class_id=torch.zeros((0,), dtype=torch.long),
        confidence=torch.zeros((0,), dtype=torch.float32),
        image_metadata=image_metadata,
        bboxes_metadata=None,
    )


def _transform_mask_to_parent(
    mask: np.ndarray, x_offset: int, y_offset: int, parent_shape: Tuple[int, int]
) -> np.ndarray:
    """
    Transform a mask from crop coordinates to parent image coordinates.

    Args:
        mask: Boolean mask array from crop (H, W)
        x_offset: X offset of crop in parent image
        y_offset: Y offset of crop in parent image
        parent_shape: (height, width) of parent image

    Returns:
        Boolean mask in parent image coordinates
    """
    parent_mask = np.zeros(parent_shape, dtype=bool)

    crop_h, crop_w = mask.shape
    parent_h, parent_w = parent_shape

    # Calculate valid region to paste (handle edge cases)
    y_start = max(0, y_offset)
    y_end = min(parent_h, y_offset + crop_h)
    x_start = max(0, x_offset)
    x_end = min(parent_w, x_offset + crop_w)

    # Calculate corresponding crop region
    crop_y_start = y_start - y_offset
    crop_y_end = crop_y_start + (y_end - y_start)
    crop_x_start = x_start - x_offset
    crop_x_end = crop_x_start + (x_end - x_start)

    # Paste the mask
    parent_mask[y_start:y_end, x_start:x_end] = mask[
        crop_y_start:crop_y_end, crop_x_start:crop_x_end
    ]

    return parent_mask


def _merge_overlapping_masks(
    predictions: List[dict], confidence_strategy: str, overlap_threshold: float = 0.0
) -> List[dict]:
    """
    Merge overlapping masks of the same class using union operations.

    Args:
        predictions: List of dictionaries with 'mask', 'confidence', and 'class_id' keys
        confidence_strategy: How to combine confidence scores
        overlap_threshold: Minimum overlap ratio (IoU) to merge (0.0 to 1.0)

    Returns:
        List of merged prediction dictionaries
    """
    if not predictions:
        return []

    n = len(predictions)
    masks = [p["mask"] for p in predictions]

    # Pre-compute bbox arrays for cheap spatial pre-filtering.
    # Bboxes are stored alongside masks when predictions are collected.
    bboxes_present = all(p.get("bbox") is not None for p in predictions)
    if bboxes_present:
        bboxes = np.array([p["bbox"] for p in predictions], dtype=np.float64)
        bx1, by1, bx2, by2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]

    parent = list(range(n))

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            next_x = parent[x]
            parent[x] = root
            x = next_x
        return root

    def union(x: int, y: int) -> None:
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    # Precompute per-mask pixel counts so IoU union can be derived arithmetically
    # (area_i + area_j - intersection) instead of materialising a full OR array.
    mask_areas = [int(np.count_nonzero(m)) for m in masks]

    for i in range(n):
        mask_i = masks[i]
        for j in range(i + 1, n):
            # Cheap bbox pre-filter: skip pixel-level AND if bboxes don't overlap
            if bboxes_present:
                if (
                    bx2[i] <= bx1[j]
                    or bx2[j] <= bx1[i]
                    or by2[i] <= by1[j]
                    or by2[j] <= by1[i]
                ):
                    continue

            mask_j = masks[j]
            intersection_count = int(np.count_nonzero(mask_i & mask_j))
            if overlap_threshold <= 0.0:
                if intersection_count > 0:
                    union(i, j)
            else:
                # Compute union via arithmetic to avoid allocating a temporary OR array
                union_count = mask_areas[i] + mask_areas[j] - intersection_count
                iou = intersection_count / union_count if union_count > 0 else 0.0
                if iou >= overlap_threshold:
                    union(i, j)

    groups_dict: dict = {}
    for i in range(n):
        root = find(i)
        if root not in groups_dict:
            groups_dict[root] = []
        groups_dict[root].append(i)

    merged_results = []
    for group_indices in groups_dict.values():
        group = [predictions[idx] for idx in group_indices]
        confidences = [p["confidence"] for p in group]
        if confidence_strategy == "max":
            merged_confidence = max(confidences)
        elif confidence_strategy == "mean":
            merged_confidence = float(np.mean(confidences))
        else:  # min
            merged_confidence = min(confidences)

        merged_mask = masks[group_indices[0]].copy()
        for idx in group_indices[1:]:
            merged_mask |= masks[idx]

        if merged_mask.any():
            merged_results.append(
                {
                    "mask": merged_mask,
                    "confidence": merged_confidence,
                    "class_id": group[0]["class_id"],
                    "detection_data": group[0].get("detection_data", {}),
                }
            )

    return merged_results


def _merge_overlapping_bboxes(
    predictions: List[dict], confidence_strategy: str, overlap_threshold: float = 0.0
) -> List[dict]:
    """
    Merge overlapping bounding boxes of the same class.

    Args:
        predictions: List of dictionaries with 'bbox', 'confidence', and 'class_id' keys
        confidence_strategy: How to combine confidence scores
        overlap_threshold: Minimum overlap ratio (IoU) to merge (0.0 to 1.0)

    Returns:
        List of merged prediction dictionaries
    """
    if not predictions:
        return []

    # Find connected components (groups of overlapping bboxes)
    groups = _find_overlapping_bbox_groups(predictions, overlap_threshold)

    # Merge each group
    merged_results = []
    for group in groups:
        # Calculate merged confidence
        confidences = [item["confidence"] for item in group]
        if confidence_strategy == "max":
            merged_confidence = max(confidences)
        elif confidence_strategy == "mean":
            merged_confidence = np.mean(confidences)
        elif confidence_strategy == "min":
            merged_confidence = min(confidences)
        else:
            merged_confidence = max(confidences)

        class_id = group[0]["class_id"]

        # Merge bounding boxes - take the union (min/max coordinates)
        bboxes_arr = np.array([item["bbox"] for item in group])
        merged_bbox = np.array(
            [
                bboxes_arr[:, 0].min(),
                bboxes_arr[:, 1].min(),
                bboxes_arr[:, 2].max(),
                bboxes_arr[:, 3].max(),
            ]
        )

        merged_results.append(
            {
                "bbox": merged_bbox,
                "confidence": merged_confidence,
                "class_id": class_id,
                "detection_data": group[0].get(
                    "detection_data", {}
                ),  # Preserve first detection's metadata
            }
        )

    return merged_results


def _find_overlapping_bbox_groups(
    predictions: List[dict], overlap_threshold: float = 0.0
) -> List[List[dict]]:
    """
    Find groups of overlapping bounding boxes using union-find with vectorised numpy ops.

    Performs all intersection/IoU computations as batched numpy operations, avoiding
    per-pair Python overhead and eliminating the need for Shapely geometry objects.

    This is a vectorised all-pairs broadphase: for each box it computes intersections
    against all subsequent boxes in one numpy call. This removes per-pair Python
    overhead and avoids Shapely geometry objects, but worst-case runtime is O(n²) in
    the number of predictions. For typical rollup workloads (tens of detections per
    class) this is faster than an STRtree due to lower constant overhead, but for
    very large detection sets a spatial index would be preferable.

    Args:
        predictions: List of dictionaries with 'bbox' key
        overlap_threshold: Minimum overlap ratio (IoU) to consider overlap

    Returns:
        List of groups, where each group is a list of prediction dicts
    """
    n = len(predictions)
    if n == 0:
        return []

    bboxes = np.array([p["bbox"] for p in predictions], dtype=np.float64)
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    parent = list(range(n))

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            next_x = parent[x]
            parent[x] = root
            x = next_x
        return root

    def union(x: int, y: int) -> None:
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for i in range(n - 1):
        # Vectorised intersection against all j > i in one shot
        inter_x1 = np.maximum(x1[i], x1[i + 1 :])
        inter_y1 = np.maximum(y1[i], y1[i + 1 :])
        inter_x2 = np.minimum(x2[i], x2[i + 1 :])
        inter_y2 = np.minimum(y2[i], y2[i + 1 :])
        inter_w = np.maximum(0.0, inter_x2 - inter_x1)
        inter_h = np.maximum(0.0, inter_y2 - inter_y1)
        intersection = inter_w * inter_h

        if overlap_threshold <= 0.0:
            overlapping = np.where(intersection > 0)[0]
        else:
            union_areas = areas[i] + areas[i + 1 :] - intersection
            iou = np.where(union_areas > 0, intersection / union_areas, 0.0)
            overlapping = np.where(iou >= overlap_threshold)[0]

        for j_offset in overlapping:
            union(i, i + 1 + int(j_offset))

    groups_dict: dict = {}
    for i in range(n):
        root = find(i)
        if root not in groups_dict:
            groups_dict[root] = []
        groups_dict[root].append(predictions[i])

    return list(groups_dict.values())
