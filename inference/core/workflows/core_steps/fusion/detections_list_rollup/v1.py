import uuid
from typing import Any, List, Literal, Optional, Tuple, Type, Union

import numpy as np
from pydantic import ConfigDict, Field
from supervision import Detections

from inference.core.workflows.execution_engine.entities.base import OutputDefinition
from inference.core.workflows.execution_engine.entities.types import (
    FLOAT_KIND,
    FLOAT_ZERO_TO_ONE_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    KEYPOINT_DETECTION_PREDICTION_KIND,
    LIST_OF_VALUES_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
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
            OBJECT_DETECTION_PREDICTION_KIND,
            INSTANCE_SEGMENTATION_PREDICTION_KIND,
            KEYPOINT_DETECTION_PREDICTION_KIND,
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
                    INSTANCE_SEGMENTATION_PREDICTION_KIND,
                    OBJECT_DETECTION_PREDICTION_KIND,
                    KEYPOINT_DETECTION_PREDICTION_KIND,
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
    if len(parent_prediction) != len(child_predictions):
        raise ValueError(
            f"Number of detections in parent_prediction ({len(parent_prediction)}) "
            f"must match number of child predictions ({len(child_predictions)})"
        )

    # Extract parent image shape from parent prediction's data
    # root_parent_dimensions is a list of tuples, one per detection (all should be the same)
    root_parent_dims = parent_prediction.data.get("root_parent_dimensions")

    if root_parent_dims is None or len(root_parent_dims) == 0:
        raise ValueError(
            "parent_prediction must have 'root_parent_dimensions' in its data attribute"
        )

    # Get the first tuple (all should be identical for the same parent image)
    parent_image_shape = root_parent_dims[0]

    # Build crop zones list - one zone per crop/child prediction
    crop_zones = []
    for i in range(len(parent_prediction)):
        crop_bbox = parent_prediction.xyxy[i]  # [x_min, y_min, x_max, y_max]
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
        if child_pred.mask is not None and len(child_pred.mask) > 0:
            has_masks = True
            break

    for child_pred in child_predictions:
        # Check for keypoint detection
        if "prediction_type" in child_pred.data:
            pred_type = child_pred.data["prediction_type"]
            if isinstance(pred_type, np.ndarray):
                if len(pred_type) > 0 and pred_type[0] == "keypoint-detection":
                    is_keypoint_detection = True
                    break
            elif pred_type == "keypoint-detection":
                is_keypoint_detection = True
                break

    # Group predictions by class
    class_predictions = {}

    # Iterate through each crop region and its corresponding child predictions
    for i, child_pred in enumerate(child_predictions):
        # Get crop location from parent prediction
        crop_bbox = parent_prediction.xyxy[i]  # [x_min, y_min, x_max, y_max]
        x_min, y_min = int(crop_bbox[0]), int(crop_bbox[1])

        # Process each detection in the child prediction
        for j in range(len(child_pred)):
            class_id = child_pred.class_id[j]
            confidence = child_pred.confidence[j]

            # Prepare keypoint data if present
            keypoint_data = {}
            if is_keypoint_detection and "keypoints_xy" in child_pred.data:
                # Transform keypoint coordinates from crop to parent space
                keypoints_xy = child_pred.data["keypoints_xy"][
                    j
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
                if "keypoints_class_name" in child_pred.data:
                    keypoint_data["keypoints_class_name"] = child_pred.data[
                        "keypoints_class_name"
                    ][j]
                if "keypoints_class_id" in child_pred.data:
                    keypoint_data["keypoints_class_id"] = child_pred.data[
                        "keypoints_class_id"
                    ][j]
                if "keypoints_confidence" in child_pred.data:
                    keypoint_data["keypoints_confidence"] = child_pred.data[
                        "keypoints_confidence"
                    ][j]

            # Collect per-detection data fields to preserve individual detection metadata
            # This is crucial for preserving class_name and other fields when multiple
            # detections have the same class_id but different values
            detection_data = {}
            for key in child_pred.data.keys():
                if key not in [
                    "detection_id",
                    "parent_id",
                    "inference_id",
                    "keypoints_xy",
                    "keypoints_class_name",
                    "keypoints_class_id",
                    "keypoints_confidence",
                ]:
                    if j < len(child_pred.data[key]):
                        detection_data[key] = child_pred.data[key][j]

            if has_masks and child_pred.mask is not None:
                # Instance segmentation - transform mask
                mask = child_pred.mask[j]
                transformed_mask = _transform_mask_to_parent(
                    mask, x_min, y_min, parent_image_shape
                )

                # Also store the transformed bbox for cheap pre-filtering
                raw_bbox = child_pred.xyxy[j]
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
                bbox = child_pred.xyxy[j]  # [x_min, y_min, x_max, y_max]
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

    # Collect all data field names from child predictions
    all_data_keys = set()
    for child_pred in child_predictions:
        all_data_keys.update(child_pred.data.keys())

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
    for child_pred in child_predictions:
        for i in range(len(child_pred)):
            class_id = child_pred.class_id[i]
            if class_id not in class_id_to_data:
                class_id_to_data[class_id] = {}
                # Store sample values for this class_id (except ID fields and keypoint fields)
                for key in child_pred.data.keys():
                    if key not in [
                        "detection_id",
                        "parent_id",
                        "inference_id",
                        "keypoints_xy",
                        "keypoints_class_name",
                        "keypoints_class_id",
                        "keypoints_confidence",
                    ]:
                        if key in child_pred.data and i < len(child_pred.data[key]):
                            class_id_to_data[class_id][key] = child_pred.data[key][i]

    # Get a sample inference_id and parent_id from the first child prediction if available
    sample_inference_id = None
    sample_parent_id = None
    if len(child_predictions) > 0 and len(child_predictions[0]) > 0:
        if "inference_id" in child_predictions[0].data:
            sample_inference_id = child_predictions[0].data["inference_id"][0]
        if "parent_id" in child_predictions[0].data:
            sample_parent_id = child_predictions[0].data["parent_id"][0]

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
        return Detections.empty(), crop_zones

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

        # Create Detections object with masks
        result = Detections(
            xyxy=xyxy_array,
            mask=merged_masks_array,
            confidence=merged_confidences_array,
            class_id=merged_class_ids_array,
        )
    else:
        # Object detection - use bounding boxes directly
        if merged_bboxes:
            xyxy_array = np.array(merged_bboxes, dtype=np.float32)
        else:
            # Shouldn't happen, but handle edge case
            xyxy_array = np.zeros((len(merged_confidences), 4), dtype=np.float32)

        # Create Detections object without masks
        result = Detections(
            xyxy=xyxy_array,
            confidence=merged_confidences_array,
            class_id=merged_class_ids_array,
        )

    # Convert data fields to numpy arrays with proper dtypes
    for key, values in merged_data.items():
        if key in [
            "class_name",
            "prediction_type",
            "detection_id",
            "parent_id",
            "inference_id",
            "root_parent_id",
        ]:
            # String fields - use 'U' dtype (Unicode strings), not np.str_
            result.data[key] = np.array(values, dtype=str)
        elif key in [
            "root_parent_dimensions",
            "parent_dimensions",
            "image_dimensions",
            "root_parent_coordinates",
            "parent_coordinates",
        ]:
            # Array/coordinate fields - convert to numpy arrays of integers
            result.data[key] = np.array(values, dtype=int)
        else:
            # Other fields - store as is
            result.data[key] = np.array(values)

    # Add keypoint data if it exists
    if is_keypoint_detection:
        if all_keypoints_data["keypoints_xy"]:
            result.data["keypoints_xy"] = np.array(
                all_keypoints_data["keypoints_xy"], dtype=object
            )
        if all_keypoints_data["keypoints_class_name"]:
            result.data["keypoints_class_name"] = np.array(
                all_keypoints_data["keypoints_class_name"], dtype=object
            )
        if all_keypoints_data["keypoints_class_id"]:
            result.data["keypoints_class_id"] = np.array(
                all_keypoints_data["keypoints_class_id"], dtype=object
            )
        if all_keypoints_data["keypoints_confidence"]:
            result.data["keypoints_confidence"] = np.array(
                all_keypoints_data["keypoints_confidence"], dtype=object
            )

    return result, crop_zones


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
