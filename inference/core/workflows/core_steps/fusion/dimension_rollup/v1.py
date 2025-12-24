from typing import Any, List, Literal, Optional, Type, Union

from pydantic import ConfigDict, Field

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union
from typing import List, Tuple

from inference.core.workflows.execution_engine.entities.base import (
    OutputDefinition,
)
from inference.core.workflows.execution_engine.entities.types import (
    LIST_OF_VALUES_KIND,
    FLOAT_ZERO_TO_ONE_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

LONG_DESCRIPTION = """
Rolls up dimensionality from from children to parent detections

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
            "name": "Dimension Roll Up",
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
    type: Literal["roboflow_core/dimension_rollup@v1", "DimensionRollUp"]
    parent_detection: Selector(kind=[OBJECT_DETECTION_PREDICTION_KIND, INSTANCE_SEGMENTATION_PREDICTION_KIND]) = Field(
        description="The parent detection the dimensionality inherits from.",
    )

    child_detections: Selector(
        kind=[
            LIST_OF_VALUES_KIND
        ]
    ) = Field(
        description="A list of child detections resulting from inferences on dynamic crops. This list can be constructed by running the \"Dimension Collapse\" block on a higher dimensionality result (ex. a prediction after a dynamic crop).",
    )

    confidence_strategy: Union[Selector(kind=[LIST_OF_VALUES_KIND]), str] = Field(
        default="max",
        title="Confidence Strategy",
        description=(
            "Strategy to use when merging confidence scores from child detections. "
            "Options are 'max', 'mean', or 'min'."
        ),
        examples=["min","mean","max"],
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
    )

    @classmethod
    def get_dimensionality_reference_property(cls) -> Optional[str]:
        return "parent_detection"

    #@classmethod
    #def get_output_dimensionality_offset(
    #    cls,
    #) -> int:
    #    return -1

    #@classmethod
    #def get_parameters_enforcing_auto_batch_casting(cls) -> List[str]:
    #    return ["child_detections"]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name="rolled_up_detections",
                kind=[INSTANCE_SEGMENTATION_PREDICTION_KIND, OBJECT_DETECTION_PREDICTION_KIND],
            )
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class DimensionRollUpBlockV1(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(self,
            parent_detection: Union[OBJECT_DETECTION_PREDICTION_KIND,INSTANCE_SEGMENTATION_PREDICTION_KIND],
            child_detections: Any, # TBD List[Union[OBJECT_DETECTION_PREDICTION_KIND,INSTANCE_SEGMENTATION_PREDICTION_KIND]]],
            confidence_strategy: str = "max",
            overlap_threshold: float = 0.0,
            ) -> BlockResult:

        print("Running Dimension Rollup Block V1")
        print(type(parent_detection))
        print(type(child_detections))
        print(parent_detection)
        print(child_detections)

        return {"rolled_up_detections": merge_crop_predictions(
            parent_detection,
            child_detections,
            confidence_strategy,
            overlap_threshold
        )}


def merge_crop_predictions(
    parent_prediction,
    child_predictions: List,
    confidence_strategy: str = "max",
    overlap_threshold: float = 0.0
):

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

    Returns:
        Detections object with merged predictions in parent image coordinates.
        Works for both instance segmentation (with masks) and object detection (without masks).
    """
    if len(parent_prediction) != len(child_predictions):
        raise ValueError(
            f"Number of detections in parent_prediction ({len(parent_prediction)}) "
            f"must match number of child predictions ({len(child_predictions)})"
        )

    # Extract parent image shape from parent prediction's data
    # root_parent_dimensions is a list of tuples, one per detection (all should be the same)
    root_parent_dims = parent_prediction.data.get('root_parent_dimensions')

    if root_parent_dims is None or len(root_parent_dims) == 0:
        raise ValueError(
            "parent_prediction must have 'root_parent_dimensions' in its data attribute"
        )

    # Get the first tuple (all should be identical for the same parent image)
    parent_image_shape = root_parent_dims[0]

    # Check if we have instance segmentation (with masks) or object detection (without masks)
    has_masks = False
    for child_pred in child_predictions:
        if child_pred.mask is not None and len(child_pred.mask) > 0:
            has_masks = True
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

            if has_masks and child_pred.mask is not None:
                # Instance segmentation - transform mask
                mask = child_pred.mask[j]
                transformed_mask = _transform_mask_to_parent(
                    mask, x_min, y_min, parent_image_shape
                )

                # Store prediction with transformed mask
                if class_id not in class_predictions:
                    class_predictions[class_id] = []

                class_predictions[class_id].append({
                    'mask': transformed_mask,
                    'confidence': confidence,
                    'class_id': class_id,
                    'bbox': None  # Will compute from mask
                })
            else:
                # Object detection - transform bounding box
                bbox = child_pred.xyxy[j]  # [x_min, y_min, x_max, y_max]
                transformed_bbox = np.array([
                    bbox[0] + x_min,
                    bbox[1] + y_min,
                    bbox[2] + x_min,
                    bbox[3] + y_min
                ])

                # Store prediction with transformed bbox
                if class_id not in class_predictions:
                    class_predictions[class_id] = []

                class_predictions[class_id].append({
                    'bbox': transformed_bbox,
                    'confidence': confidence,
                    'class_id': class_id,
                    'mask': None
                })

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
    merged_data = {key: [] for key in all_data_keys}

    # Build mapping from class_id to typical data values
    class_id_to_data = {}
    for child_pred in child_predictions:
        for i in range(len(child_pred)):
            class_id = child_pred.class_id[i]
            if class_id not in class_id_to_data:
                class_id_to_data[class_id] = {}
                # Store sample values for this class_id (except ID fields)
                for key in child_pred.data.keys():
                    if key not in ['detection_id', 'parent_id', 'inference_id']:
                        if key in child_pred.data and i < len(child_pred.data[key]):
                            class_id_to_data[class_id][key] = child_pred.data[key][i]

    # Get a sample inference_id and parent_id from the first child prediction if available
    sample_inference_id = None
    sample_parent_id = None
    if len(child_predictions) > 0 and len(child_predictions[0]) > 0:
        if 'inference_id' in child_predictions[0].data:
            sample_inference_id = child_predictions[0].data['inference_id'][0]
        if 'parent_id' in child_predictions[0].data:
            sample_parent_id = child_predictions[0].data['parent_id'][0]

    # Counter for generating new detection IDs
    import uuid

    for class_id, preds in class_predictions.items():
        if has_masks:
            merged_preds = _merge_overlapping_masks(preds, confidence_strategy, overlap_threshold)
        else:
            merged_preds = _merge_overlapping_bboxes(preds, confidence_strategy, overlap_threshold)

        for pred in merged_preds:
            if has_masks:
                merged_masks.append(pred['mask'])
            merged_confidences.append(pred['confidence'])
            merged_class_ids.append(pred['class_id'])

            # Store bbox for later use
            if 'bbox' in pred and pred['bbox'] is not None:
                merged_bboxes.append(pred['bbox'])

            # Add data fields for this detection
            for key in all_data_keys:
                if key == 'detection_id':
                    # Generate new UUID for merged detection
                    merged_data[key].append(str(uuid.uuid4()))
                elif key == 'parent_id':
                    # Use sample parent_id or generate new one
                    merged_data[key].append(sample_parent_id if sample_parent_id else str(uuid.uuid4()))
                elif key == 'inference_id':
                    # Use the same inference_id as inputs (they're from same inference batch)
                    merged_data[key].append(sample_inference_id if sample_inference_id else str(uuid.uuid4()))
                elif key == 'root_parent_dimensions':
                    # Add the parent image shape as a list [height, width]
                    merged_data[key].append(list(parent_image_shape))
                elif key == 'parent_dimensions':
                    # Parent dimensions should be same as root_parent_dimensions for merged results
                    merged_data[key].append(list(parent_image_shape))
                elif key == 'image_dimensions':
                    # Image dimensions for this detection
                    merged_data[key].append(list(parent_image_shape))
                elif key == 'root_parent_coordinates':
                    # Root parent coordinates [y, x] - should be [0, 0] for the root
                    if pred['class_id'] in class_id_to_data and key in class_id_to_data[pred['class_id']]:
                        merged_data[key].append(class_id_to_data[pred['class_id']][key])
                    else:
                        merged_data[key].append([0, 0])
                elif key == 'parent_coordinates':
                    # Parent coordinates [y, x]
                    if pred['class_id'] in class_id_to_data and key in class_id_to_data[pred['class_id']]:
                        merged_data[key].append(class_id_to_data[pred['class_id']][key])
                    else:
                        merged_data[key].append([0, 0])
                elif key == 'root_parent_id':
                    # Root parent ID
                    if pred['class_id'] in class_id_to_data and key in class_id_to_data[pred['class_id']]:
                        merged_data[key].append(class_id_to_data[pred['class_id']][key])
                    else:
                        merged_data[key].append('image')
                elif key == 'prediction_type':
                    # Prediction type should be 'instance-segmentation'
                    merged_data[key].append('instance-segmentation')
                else:
                    # For other fields like class_name, use the value associated with this class_id
                    if pred['class_id'] in class_id_to_data and key in class_id_to_data[pred['class_id']]:
                        merged_data[key].append(class_id_to_data[pred['class_id']][key])
                    else:
                        merged_data[key].append(None)

    if not merged_confidences:
        # Return empty detections if no detections
        from supervision import Detections
        return Detections.empty()

    # Convert to numpy arrays
    merged_confidences_array = np.array(merged_confidences, dtype=np.float32)
    merged_class_ids_array = np.array(merged_class_ids, dtype=int)

    from supervision import Detections

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
            class_id=merged_class_ids_array
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
            class_id=merged_class_ids_array
        )

    # Convert data fields to numpy arrays with proper dtypes
    for key, values in merged_data.items():
        if key in ['class_name', 'prediction_type', 'detection_id', 'parent_id',
                   'inference_id', 'root_parent_id']:
            # String fields - use 'U' dtype (Unicode strings), not np.str_
            result.data[key] = np.array(values, dtype=str)
        elif key in ['root_parent_dimensions', 'parent_dimensions', 'image_dimensions',
                     'root_parent_coordinates', 'parent_coordinates']:
            # Array/coordinate fields - convert to numpy arrays of integers
            result.data[key] = np.array(values, dtype=int)
        else:
            # Other fields - store as is
            result.data[key] = np.array(values)

    return result


def _transform_mask_to_parent(
    mask: np.ndarray,
    x_offset: int,
    y_offset: int,
    parent_shape: Tuple[int, int]
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
    predictions: List[dict],
    confidence_strategy: str,
    overlap_threshold: float = 0.0
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

    # Convert masks to polygons for merging
    polygons_with_data = []
    for pred in predictions:
        mask = pred['mask']
        polygons = _mask_to_polygons(mask)

        for poly in polygons:
            if poly.is_valid and not poly.is_empty:
                polygons_with_data.append({
                    'polygon': poly,
                    'confidence': pred['confidence'],
                    'class_id': pred['class_id']
                })

    if not polygons_with_data:
        return []

    # Find connected components (groups of overlapping polygons)
    groups = _find_overlapping_groups(polygons_with_data, overlap_threshold)

    # Merge each group
    merged_results = []
    for group in groups:
        merged_poly = unary_union([item['polygon'] for item in group])

        # Calculate merged confidence
        confidences = [item['confidence'] for item in group]
        if confidence_strategy == "max":
            merged_confidence = max(confidences)
        elif confidence_strategy == "mean":
            merged_confidence = np.mean(confidences)
        elif confidence_strategy == "min":
            merged_confidence = min(confidences)
        else:
            merged_confidence = max(confidences)

        class_id = group[0]['class_id']

        # Get image shape from first mask
        image_shape = group[0].get('image_shape', predictions[0]['mask'].shape)

        # Handle MultiPolygon results
        if merged_poly.geom_type == 'MultiPolygon':
            for poly in merged_poly.geoms:
                mask = _polygon_to_mask(poly, image_shape)
                if mask.any():
                    merged_results.append({
                        'mask': mask,
                        'confidence': merged_confidence,
                        'class_id': class_id
                    })
        else:
            mask = _polygon_to_mask(merged_poly, image_shape)
            if mask.any():
                merged_results.append({
                    'mask': mask,
                    'confidence': merged_confidence,
                    'class_id': class_id
                })

    return merged_results


def _merge_overlapping_bboxes(
    predictions: List[dict],
    confidence_strategy: str,
    overlap_threshold: float = 0.0
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
        confidences = [item['confidence'] for item in group]
        if confidence_strategy == "max":
            merged_confidence = max(confidences)
        elif confidence_strategy == "mean":
            merged_confidence = np.mean(confidences)
        elif confidence_strategy == "min":
            merged_confidence = min(confidences)
        else:
            merged_confidence = max(confidences)

        class_id = group[0]['class_id']

        # Merge bounding boxes - take the union (min/max coordinates)
        bboxes = [item['bbox'] for item in group]
        x_mins = [bbox[0] for bbox in bboxes]
        y_mins = [bbox[1] for bbox in bboxes]
        x_maxs = [bbox[2] for bbox in bboxes]
        y_maxs = [bbox[3] for bbox in bboxes]

        merged_bbox = np.array([
            min(x_mins),
            min(y_mins),
            max(x_maxs),
            max(y_maxs)
        ])

        merged_results.append({
            'bbox': merged_bbox,
            'confidence': merged_confidence,
            'class_id': class_id
        })

    return merged_results


def _find_overlapping_bbox_groups(predictions: List[dict], overlap_threshold: float = 0.0) -> List[List[dict]]:
    """
    Find groups of overlapping bounding boxes using union-find.

    Args:
        predictions: List of dictionaries with 'bbox' key
        overlap_threshold: Minimum overlap ratio (IoU) to consider overlap

    Returns:
        List of groups, where each group is a list of prediction dicts
    """
    n = len(predictions)
    parent = list(range(n))

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    def bbox_iou(bbox1, bbox2):
        """Calculate IoU between two bboxes [x_min, y_min, x_max, y_max]"""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2

        # Calculate intersection
        x_min = max(x1_min, x2_min)
        y_min = max(y1_min, y2_min)
        x_max = min(x1_max, x2_max)
        y_max = min(y1_max, y2_max)

        if x_max <= x_min or y_max <= y_min:
            return 0.0

        intersection = (x_max - x_min) * (y_max - y_min)

        # Calculate union
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    # Check all pairs for overlap
    for i in range(n):
        for j in range(i + 1, n):
            bbox1 = predictions[i]['bbox']
            bbox2 = predictions[j]['bbox']

            iou = bbox_iou(bbox1, bbox2)

            if overlap_threshold <= 0.0:
                # Merge if they overlap at all
                if iou > 0:
                    union(i, j)
            else:
                # Merge only if IoU exceeds threshold
                if iou >= overlap_threshold:
                    union(i, j)

    # Group by root parent
    groups_dict = {}
    for i in range(n):
        root = find(i)
        if root not in groups_dict:
            groups_dict[root] = []
        groups_dict[root].append(predictions[i])

    return list(groups_dict.values())


def _mask_to_polygons(mask: np.ndarray) -> List[Polygon]:
    """
    Convert a binary mask to a list of Shapely polygons.

    Args:
        mask: Boolean mask array (H, W)

    Returns:
        List of Polygon objects
    """
    from skimage import measure

    # Find contours
    contours = measure.find_contours(mask.astype(np.uint8), 0.5)

    polygons = []
    for contour in contours:
        # Convert from (row, col) to (x, y)
        contour = np.flip(contour, axis=1)

        # Simplify and create polygon
        if len(contour) >= 3:
            try:
                poly = Polygon(contour)
                if poly.is_valid:
                    polygons.append(poly)
            except Exception:
                continue

    return polygons


def _polygon_to_mask(polygon: Polygon, shape: Tuple[int, int]) -> np.ndarray:
    """
    Convert a Shapely polygon to a binary mask.

    Args:
        polygon: Shapely Polygon object
        shape: (height, width) of output mask

    Returns:
        Boolean mask array
    """
    from skimage import draw

    mask = np.zeros(shape, dtype=bool)

    # Get exterior coordinates
    coords = np.array(polygon.exterior.coords)

    if len(coords) < 3:
        return mask

    # Convert from (x, y) to (row, col)
    rows = coords[:, 1]
    cols = coords[:, 0]

    # Fill polygon
    try:
        rr, cc = draw.polygon(rows, cols, shape)
        mask[rr, cc] = True
    except Exception:
        pass

    return mask


def _find_overlapping_groups(polygons_with_data: List[dict], overlap_threshold: float = 0.0) -> List[List[dict]]:
    """
    Find groups of overlapping/touching polygons using union-find.

    Args:
        polygons_with_data: List of dictionaries with 'polygon' key.
        overlap_threshold: Minimum overlap ratio (IoU) to consider overlap (0.0 to 1.0)

    Returns:
        List of groups, where each group is a list of polygon data dicts.
    """
    n = len(polygons_with_data)
    parent = list(range(n))

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    # Check all pairs for overlap
    for i in range(n):
        for j in range(i + 1, n):
            poly1 = polygons_with_data[i]['polygon']
            poly2 = polygons_with_data[j]['polygon']

            # Check if polygons overlap based on threshold
            if overlap_threshold <= 0.0:
                # Merge if they touch or overlap at all
                if poly1.intersects(poly2) or poly1.touches(poly2):
                    union(i, j)
            else:
                # Merge only if overlap ratio exceeds threshold
                if poly1.intersects(poly2):
                    intersection_area = poly1.intersection(poly2).area
                    union_area = poly1.union(poly2).area
                    iou = intersection_area / union_area if union_area > 0 else 0
                    if iou >= overlap_threshold:
                        union(i, j)

    # Group by root parent
    groups_dict = {}
    for i in range(n):
        root = find(i)
        if root not in groups_dict:
            groups_dict[root] = []
        groups_dict[root].append(polygons_with_data[i])

    return list(groups_dict.values())