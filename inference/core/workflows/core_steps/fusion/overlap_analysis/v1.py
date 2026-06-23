from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

import numpy as np
import supervision as sv
from pydantic import ConfigDict, Field
from shapely.geometry import Polygon, box
from supervision.config import CLASS_NAME_DATA_FIELD

from inference.core.workflows.execution_engine.constants import DETECTION_ID_KEY
from inference.core.workflows.execution_engine.entities.base import OutputDefinition
from inference.core.workflows.execution_engine.entities.types import (
    DETECTIONS_OVERLAPS_KIND,
    FLOAT_ZERO_TO_ONE_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    FloatZeroToOne,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

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
            OBJECT_DETECTION_PREDICTION_KIND,
            INSTANCE_SEGMENTATION_PREDICTION_KIND,
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
            OBJECT_DETECTION_PREDICTION_KIND,
            INSTANCE_SEGMENTATION_PREDICTION_KIND,
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
        reference_predictions: sv.Detections,
        candidate_predictions: sv.Detections,
        min_overlap: float,
    ) -> BlockResult:
        if len(reference_predictions) == 0 or len(candidate_predictions) == 0:
            return {"overlaps": []}

        iou_matrix = sv.box_iou_batch(
            reference_predictions.xyxy, candidate_predictions.xyxy
        )

        ref_ids = reference_predictions.data.get(DETECTION_ID_KEY)
        cand_ids = candidate_predictions.data.get(DETECTION_ID_KEY)
        ref_classes = reference_predictions.data.get(CLASS_NAME_DATA_FIELD)
        cand_classes = candidate_predictions.data.get(CLASS_NAME_DATA_FIELD)

        results: List[Dict[str, Any]] = []
        ref_polys: Dict[int, Polygon] = {}
        cand_polys: Dict[int, Polygon] = {}

        for i in range(len(reference_predictions)):
            for j in range(len(candidate_predictions)):
                if iou_matrix[i, j] <= 0.0:
                    continue
                if i not in ref_polys:
                    ref_polys[i] = _detection_to_shapely(reference_predictions, i)
                if j not in cand_polys:
                    cand_polys[j] = _detection_to_shapely(candidate_predictions, j)
                ref_poly = ref_polys[i]
                cand_poly = cand_polys[j]
                if ref_poly.area <= 0:
                    continue
                intersection_area = ref_poly.intersection(cand_poly).area
                overlap_ratio = intersection_area / ref_poly.area
                if overlap_ratio < min_overlap:
                    continue
                record: Dict[str, Any] = {
                    "reference_class": _safe_get(ref_classes, i),
                    "reference_confidence": (
                        float(reference_predictions.confidence[i])
                        if reference_predictions.confidence is not None
                        else None
                    ),
                    "candidate_class": _safe_get(cand_classes, j),
                    "candidate_confidence": (
                        float(candidate_predictions.confidence[j])
                        if candidate_predictions.confidence is not None
                        else None
                    ),
                    "overlap_ratio": float(overlap_ratio),
                }
                if ref_ids is not None:
                    record["reference_detection_id"] = _safe_get(ref_ids, i)
                if cand_ids is not None:
                    record["candidate_detection_id"] = _safe_get(cand_ids, j)
                results.append(record)
        return {"overlaps": results}


def _detection_to_shapely(detections: sv.Detections, idx: int) -> Polygon:
    """Return the precise polygon for the detection at `idx`.

    When `detections.mask` is set and non-empty for this row, the polygon is
    the longest contour of the mask (validated via shapely). Otherwise — and
    on any invalidity / emptiness — the 4-corner bbox polygon is returned.
    `sv.Detections` always exposes a `mask` attribute, but it may be `None`
    for bbox-only sources (e.g. plain object detection).
    """
    x1, y1, x2, y2 = detections.xyxy[idx]
    bbox_poly = box(float(x1), float(y1), float(x2), float(y2))
    if detections.mask is not None and np.any(detections.mask[idx]):
        polygons = sv.mask_to_polygons(mask=detections.mask[idx].astype(np.uint8))
        if polygons:
            longest = max(polygons, key=len)
            if len(longest) >= 3:
                candidate = Polygon([(float(pt[0]), float(pt[1])) for pt in longest])
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
