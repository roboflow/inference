from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

import numpy as np
import supervision as sv
import torch
from pydantic import ConfigDict, Field
from shapely.geometry import Polygon, box

from inference.core.workflows.execution_engine.constants import (
    CLASS_NAMES_KEY,
    DETECTION_ID_KEY,
)
from inference.core.workflows.execution_engine.entities.base import OutputDefinition
from inference.core.workflows.execution_engine.entities.tensor_native_types import (
    TENSOR_NATIVE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
    TENSOR_NATIVE_OBJECT_DETECTION_PREDICTION_KIND,
)
from inference.core.workflows.execution_engine.entities.types import (
    DETECTIONS_OVERLAPS_KIND,
    FLOAT_ZERO_TO_ONE_KIND,
    FloatZeroToOne,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)
from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.object_detection import Detections
from inference_models.models.base.types import InstancesRLEMasks
from inference_models.models.common.rle_utils import coco_rle_masks_to_numpy_mask

LONG_DESCRIPTION = """
Compute pairwise geometric overlap between two sets of detections.

For each pair (reference, candidate) drawn from `reference_predictions` x
`candidate_predictions`, the block computes
`intersection_area / reference_polygon.area` and emits one record per
pair whose ratio reaches `min_overlap`.

When a detection carries a mask, the precise polygon is the longest
contour of that mask (validated via shapely); otherwise the bounding
box polygon is used. A vectorised bbox-IoU prefilter
(`supervision.box_iou_batch`) eliminates non-touching pairs before any
shapely intersection is computed.

The relation is intentionally **not symmetric** across the two inputs:
the denominator is always the reference detection's area. Swap the two
selectors if you want overlap reported relative to the other set.

The output is a flat list of dicts (one per accepted pair) attached to
the same dimensionality as the inputs — the block does not increase
dimensionality. See the `detections_overlaps` kind docs for the
per-record schema.
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Overlap Analysis",
            "version": "v1",
            "short_description": (
                "Compute pairwise overlap between two sets of detections."
            ),
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "fusion",
        },
        protected_namespaces=(),
    )
    type: Literal["roboflow_core/overlap_analysis@v1", "OverlapAnalysis"]
    reference_predictions: Selector(
        kind=[
            TENSOR_NATIVE_OBJECT_DETECTION_PREDICTION_KIND,
            TENSOR_NATIVE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
        ]
    ) = Field(
        description=(
            "Detections whose area is the denominator of the overlap ratio. "
            "For each reference detection, overlap with every candidate is "
            "computed; pairs above `min_overlap` appear in the output."
        ),
        examples=["$steps.model_a.predictions"],
    )
    candidate_predictions: Selector(
        kind=[
            TENSOR_NATIVE_OBJECT_DETECTION_PREDICTION_KIND,
            TENSOR_NATIVE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
        ]
    ) = Field(
        description="Detections checked against each reference.",
        examples=["$steps.model_b.predictions"],
    )
    min_overlap: Union[FloatZeroToOne, Selector(kind=[FLOAT_ZERO_TO_ONE_KIND])] = Field(
        default=0.1,
        description=(
            "Minimum (intersection / reference_area) ratio for a pair to be "
            "included in the output."
        ),
        examples=[0.1, "$inputs.min_overlap"],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="overlaps", kind=[DETECTIONS_OVERLAPS_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class OverlapAnalysisBlockV1(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        reference_predictions: Union[Detections, InstanceDetections],
        candidate_predictions: Union[Detections, InstanceDetections],
        min_overlap: float,
    ) -> BlockResult:
        if len(reference_predictions) == 0 or len(candidate_predictions) == 0:
            return {"overlaps": []}

        ref_xyxy = reference_predictions.xyxy.detach().to("cpu").numpy()
        cand_xyxy = candidate_predictions.xyxy.detach().to("cpu").numpy()

        iou_matrix = sv.box_iou_batch(ref_xyxy, cand_xyxy)

        ref_ids = _detection_ids(reference_predictions)
        cand_ids = _detection_ids(candidate_predictions)
        ref_class_names = _class_names(reference_predictions)
        cand_class_names = _class_names(candidate_predictions)
        ref_confidences = _confidence_values(reference_predictions)
        cand_confidences = _confidence_values(candidate_predictions)

        if _is_bbox_only(reference_predictions) and _is_bbox_only(
            candidate_predictions
        ):
            ref_boxes = ref_xyxy.astype(np.float64, copy=False)
            cand_boxes = cand_xyxy.astype(np.float64, copy=False)

            ref_x1 = np.minimum(ref_boxes[:, 0], ref_boxes[:, 2])
            ref_y1 = np.minimum(ref_boxes[:, 1], ref_boxes[:, 3])
            ref_x2 = np.maximum(ref_boxes[:, 0], ref_boxes[:, 2])
            ref_y2 = np.maximum(ref_boxes[:, 1], ref_boxes[:, 3])
            cand_x1 = np.minimum(cand_boxes[:, 0], cand_boxes[:, 2])
            cand_y1 = np.minimum(cand_boxes[:, 1], cand_boxes[:, 3])
            cand_x2 = np.maximum(cand_boxes[:, 0], cand_boxes[:, 2])
            cand_y2 = np.maximum(cand_boxes[:, 1], cand_boxes[:, 3])

            intersection_width = np.clip(
                np.minimum(ref_x2[:, None], cand_x2[None, :])
                - np.maximum(ref_x1[:, None], cand_x1[None, :]),
                0.0,
                None,
            )
            intersection_height = np.clip(
                np.minimum(ref_y2[:, None], cand_y2[None, :])
                - np.maximum(ref_y1[:, None], cand_y1[None, :]),
                0.0,
                None,
            )
            intersection = intersection_width * intersection_height
            ref_area = (ref_x2 - ref_x1) * (ref_y2 - ref_y1)
            overlap_ratios = np.zeros_like(intersection, dtype=np.float64)
            np.divide(
                intersection,
                ref_area[:, None],
                out=overlap_ratios,
                where=ref_area[:, None] > 0.0,
            )
            emitted_pairs = np.argwhere(
                (iou_matrix > 0.0)
                & (ref_area[:, None] > 0.0)
                & (overlap_ratios >= min_overlap)
            )

            results: List[Dict[str, Any]] = []
            for i, j in emitted_pairs:
                results.append(
                    _build_record(
                        i=int(i),
                        j=int(j),
                        overlap_ratio=overlap_ratios[i, j],
                        ref_class_names=ref_class_names,
                        cand_class_names=cand_class_names,
                        ref_confidences=ref_confidences,
                        cand_confidences=cand_confidences,
                        ref_ids=ref_ids,
                        cand_ids=cand_ids,
                    )
                )
            return {"overlaps": results}

        eligible_pairs = iou_matrix > 0.0
        ref_mask_rows = _materialize_mask_rows(
            reference_predictions,
            np.flatnonzero(np.any(eligible_pairs, axis=1)),
        )
        cand_mask_rows = _materialize_mask_rows(
            candidate_predictions,
            np.flatnonzero(np.any(eligible_pairs, axis=0)),
        )

        results: List[Dict[str, Any]] = []
        ref_polys: Dict[int, Polygon] = {}
        cand_polys: Dict[int, Polygon] = {}

        for i in range(len(reference_predictions)):
            for j in range(len(candidate_predictions)):
                if not eligible_pairs[i, j]:
                    continue
                if i not in ref_polys:
                    ref_polys[i] = _detection_to_shapely(
                        reference_predictions, ref_xyxy, i, ref_mask_rows
                    )
                if j not in cand_polys:
                    cand_polys[j] = _detection_to_shapely(
                        candidate_predictions, cand_xyxy, j, cand_mask_rows
                    )
                ref_poly = ref_polys[i]
                cand_poly = cand_polys[j]
                if ref_poly.area <= 0:
                    continue
                intersection_area = ref_poly.intersection(cand_poly).area
                overlap_ratio = intersection_area / ref_poly.area
                if overlap_ratio < min_overlap:
                    continue
                results.append(
                    _build_record(
                        i=i,
                        j=j,
                        overlap_ratio=overlap_ratio,
                        ref_class_names=ref_class_names,
                        cand_class_names=cand_class_names,
                        ref_confidences=ref_confidences,
                        cand_confidences=cand_confidences,
                        ref_ids=ref_ids,
                        cand_ids=cand_ids,
                    )
                )
        return {"overlaps": results}


def _is_bbox_only(detections: Union[Detections, InstanceDetections]) -> bool:
    return not isinstance(detections, InstanceDetections) or detections.mask is None


def _confidence_values(
    detections: Union[Detections, InstanceDetections],
) -> Optional[np.ndarray]:
    if detections.confidence is None:
        return None
    return detections.confidence.detach().cpu().numpy()


def _materialize_mask_rows(
    detections: Union[Detections, InstanceDetections], indices: np.ndarray
) -> Dict[int, np.ndarray]:
    if (
        not isinstance(detections, InstanceDetections)
        or detections.mask is None
        or len(indices) == 0
    ):
        return {}
    mask = detections.mask
    if isinstance(mask, InstancesRLEMasks):
        return {
            int(index): coco_rle_masks_to_numpy_mask(
                InstancesRLEMasks(
                    image_size=mask.image_size,
                    masks=[mask.masks[int(index)]],
                )
            )[0]
            for index in indices
        }
    indices_tensor = torch.as_tensor(indices, dtype=torch.long, device=mask.device)
    materialized = mask[indices_tensor].detach().to("cpu").numpy()
    return {
        int(index): materialized[position] for position, index in enumerate(indices)
    }


def _build_record(
    i: int,
    j: int,
    overlap_ratio: float,
    ref_class_names: List[Optional[str]],
    cand_class_names: List[Optional[str]],
    ref_confidences: Optional[np.ndarray],
    cand_confidences: Optional[np.ndarray],
    ref_ids: Optional[List[Any]],
    cand_ids: Optional[List[Any]],
) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "reference_class": _safe_get(ref_class_names, i),
        "reference_confidence": (
            float(ref_confidences[i]) if ref_confidences is not None else None
        ),
        "candidate_class": _safe_get(cand_class_names, j),
        "candidate_confidence": (
            float(cand_confidences[j]) if cand_confidences is not None else None
        ),
        "overlap_ratio": float(overlap_ratio),
    }
    if ref_ids is not None:
        record["reference_detection_id"] = _safe_get(ref_ids, i)
    if cand_ids is not None:
        record["candidate_detection_id"] = _safe_get(cand_ids, j)
    return record


def _detection_ids(
    detections: Union[Detections, InstanceDetections],
) -> Optional[List[Any]]:
    """Return the per-detection `detection_id` list (read from `bboxes_metadata`)
    or `None` when no detection carries one — mirroring the numpy block's
    `detections.data.get(DETECTION_ID_KEY)`."""
    bboxes_metadata = detections.bboxes_metadata
    if bboxes_metadata is None:
        return None
    ids = [m.get(DETECTION_ID_KEY) for m in bboxes_metadata]
    if all(value is None for value in ids):
        return None
    return ids


def _class_names(
    detections: Union[Detections, InstanceDetections],
) -> List[Optional[str]]:
    """Resolve the per-detection class name from the `image_metadata` class-names
    map (the tensor-native equivalent of the numpy block's
    `detections.data.get(CLASS_NAME_DATA_FIELD)`)."""
    class_names_map = (detections.image_metadata or {}).get(CLASS_NAMES_KEY) or {}
    class_ids = detections.class_id.detach().to("cpu").numpy()
    return [
        class_names_map.get(int(class_id), f"class_{int(class_id)}")
        for class_id in class_ids
    ]


def _detection_to_shapely(
    detections: Union[Detections, InstanceDetections],
    xyxy: np.ndarray,
    idx: int,
    materialized_masks: Dict[int, np.ndarray],
) -> Polygon:
    """Return the precise polygon for the detection at `idx`.

    When `detections` carries a mask and it is non-empty for this row, the
    polygon is the longest contour of the mask (validated via shapely).
    Otherwise — and on any invalidity / emptiness — the 4-corner bbox polygon is
    returned. Object detection (`inference_models.Detections`) has no mask, so it
    always falls back to the bbox polygon.
    """
    x1, y1, x2, y2 = xyxy[idx]
    bbox_poly = box(float(x1), float(y1), float(x2), float(y2))
    if isinstance(detections, InstanceDetections) and detections.mask is not None:
        mask = materialized_masks[idx]
        if np.any(mask):
            polygons = sv.mask_to_polygons(mask=mask.astype(np.uint8))
            if polygons:
                longest = max(polygons, key=len)
                if len(longest) >= 3:
                    candidate = Polygon(
                        [(float(pt[0]), float(pt[1])) for pt in longest]
                    )
                    if candidate.is_valid and not candidate.is_empty:
                        return candidate
    return bbox_poly


def _safe_get(arr: Any, idx: int) -> Optional[Any]:
    """Return `arr[idx]` when `arr` is not None and `idx` is within range,
    else `None`. Works on both numpy arrays and plain Python sequences."""
    if arr is None:
        return None
    try:
        if len(arr) <= idx:
            return None
        value = arr[idx]
    except (TypeError, IndexError):
        return None
    # numpy scalar -> Python scalar where applicable; keep strings as-is.
    if hasattr(value, "item") and not isinstance(value, (bytes, str)):
        try:
            return value.item()
        except (ValueError, TypeError):
            return value
    return value
