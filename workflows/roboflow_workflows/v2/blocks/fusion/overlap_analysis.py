"""Asymmetric detection overlap, with a CPU reference boundary for polygons.

Boxes use native tensor intersection/reference-area arithmetic. Masks use V1's
longest contour (by vertex count), validated by Shapely; empty or invalid masks
fall back to their box. Only rows with a positive box IoU are materialized, at
reference fidelity. Coordinates are compared as supplied; no frame reprojection
is performed. The cleared IoU prefilter casts boxes to float32. V1 retains the input dtype
for box areas (supervision uses float32 intersection buffers), so higher
precision input can decide differently near numeric boundaries. Box ratio
arithmetic itself remains float64.
"""

import math
from typing import Annotated, Any, Dict

import numpy as np
import supervision as sv
import torch
from inference_models.models.base.instance_segmentation import InstanceDetections
from pydantic import Field
from roboflow_workflows.v2.blocks.fusion.pairwise import box_iou_between
from roboflow_workflows.v2.data.kinds import FLOAT_KIND, Kind
from roboflow_workflows.v2.data.masks.grid import (
    INSTANCE_SEGMENTATION_VIEW_KIND,
    SegmentationView,
)
from roboflow_workflows.v2.data.masks.providers import mask_provider_of
from roboflow_workflows.v2.data.prediction_ops import CLASS_NAMES_KEY
from roboflow_workflows.v2.data.predictions import (
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
)
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref
from shapely.geometry import Polygon, box

DETECTION_KINDS = (
    OBJECT_DETECTION_PREDICTION_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
    INSTANCE_SEGMENTATION_VIEW_KIND,
)
UnitFloat = Annotated[float, Field(ge=0.0, le=1.0)]


def _valid_overlaps(payload: Any) -> bool:
    required = {
        "reference_class",
        "reference_confidence",
        "candidate_class",
        "candidate_confidence",
        "overlap_ratio",
    }
    optional = {"reference_detection_id", "candidate_detection_id"}
    if not isinstance(payload, list):
        return False

    for record in payload:
        if not isinstance(record, dict) or not required <= record.keys():
            return False

        if record.keys() - required - optional:
            return False

        ratio = record["overlap_ratio"]
        if isinstance(ratio, bool) or not isinstance(ratio, (int, float)):
            return False

        if not math.isfinite(ratio) or not 0 <= ratio <= 1:
            return False

        for key in ("reference_class", "candidate_class"):
            if record[key] is not None and not isinstance(record[key], str):
                return False

        for key in ("reference_confidence", "candidate_confidence"):
            value = record[key]
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, (int, float))
            ):
                return False

    return True


DETECTIONS_OVERLAPS_KIND = Kind(
    name="detections_overlaps",
    description="Flat reference-major pair records with intersection/reference-area ratios.",
    validate=_valid_overlaps,
)

__all__ = ["OverlapAnalysisBlock", "DETECTIONS_OVERLAPS_KIND"]


class OverlapAnalysisBlock(Block):
    """Inspect overlap relative to each reference detection's area.

    Swap the inputs to change the denominator. Class names come from each
    input's image-level class map, with ``class_<id>`` fallback. IDs come from
    aligned bbox metadata; an ID key is omitted when that input has no IDs.
    Output entry metadata follows the reference. The block is stateless and
    never changes either prediction or its masks.
    """

    type = "v2/overlap_analysis@v1"
    aliases = ("OverlapAnalysis",)
    prunable = True
    metadata = {
        "name": "Overlap Analysis",
        "port_of": "roboflow_core/overlap_analysis@v1",
        "section": "advanced",
    }
    outputs = {
        "overlaps": Output(
            DETECTIONS_OVERLAPS_KIND,
            source="reference_predictions",
            description="Flat pair records whose intersection/reference-area reaches the threshold.",
        ),
    }

    class Params(BlockParams):
        reference_predictions: Ref(*DETECTION_KINDS) = Field(
            description="Native reference detections; their area is the denominator.",
            examples=["$steps.model_a.predictions"],
        )
        candidate_predictions: Ref(*DETECTION_KINDS) = Field(
            description="Native candidate detections in comparable coordinates.",
            examples=["$steps.model_b.predictions"],
        )
        min_overlap: UnitFloat | Ref(FLOAT_KIND) = Field(
            default=0.1,
            description="Inclusive minimum intersection/reference-area ratio in [0,1].",
            examples=[0.1, "$inputs.min_overlap"],
        )

    def run(
        self,
        *,
        reference_predictions: Any,
        candidate_predictions: Any,
        min_overlap: float,
    ) -> Dict[str, Any]:
        """Compute the accepted reference/candidate pairs in reference-major order.

        Args:
            reference_predictions: Native detections or a SegmentationView.
            candidate_predictions: Native detections or a SegmentationView.
            min_overlap: Inclusive ratio threshold from zero to one.

        Returns:
            ``overlaps`` as a flat list of class, confidence, ratio and optional
            detection ID records, without introducing a batch dimension.

        Raises:
            ValueError: If a selected threshold falls outside [0,1].
        """
        if not 0 <= min_overlap <= 1:
            raise ValueError("min_overlap must be in [0,1]")

        reference = _detections(reference_predictions)
        candidate = _detections(candidate_predictions)
        if not len(reference) or not len(candidate):
            return {"overlaps": []}

        a = reference.xyxy.detach()
        b = candidate.xyxy.detach().to(device=a.device)
        eligible = box_iou_between(a, b) > 0
        if _has_masks(reference_predictions) or _has_masks(candidate_predictions):
            pairs, ratios = _polygon_reference_overlaps(
                reference_predictions,
                candidate_predictions,
                eligible=eligible,
                min_overlap=min_overlap,
            )
        else:
            a, b = a.to(torch.float64), b.to(torch.float64)
            a_min, a_max = torch.minimum(a[:, :2], a[:, 2:]), torch.maximum(
                a[:, :2], a[:, 2:]
            )
            b_min, b_max = torch.minimum(b[:, :2], b[:, 2:]), torch.maximum(
                b[:, :2], b[:, 2:]
            )
            intersection = (
                (
                    torch.minimum(a_max[:, None], b_max[None])
                    - torch.maximum(a_min[:, None], b_min[None])
                )
                .clamp(min=0)
                .prod(dim=2)
            )
            area = (a_max - a_min).prod(dim=1)
            ratios_tensor = intersection / area[:, None].clamp(
                min=torch.finfo(torch.float64).tiny
            )
            accepted = eligible & (area[:, None] > 0) & (ratios_tensor >= min_overlap)
            pair_tensor = accepted.nonzero()
            pairs = pair_tensor.cpu().tolist()
            ratios = ratios_tensor[accepted].cpu().tolist()

        ref_fields, cand_fields = _record_fields(reference), _record_fields(candidate)
        records = []
        for (i, j), ratio in zip(pairs, ratios):
            record = {
                "reference_class": ref_fields["classes"][i],
                "reference_confidence": ref_fields["confidence"][i],
                "candidate_class": cand_fields["classes"][j],
                "candidate_confidence": cand_fields["confidence"][j],
                "overlap_ratio": float(ratio),
            }
            for prefix, fields, index in (
                ("reference", ref_fields, i),
                ("candidate", cand_fields, j),
            ):
                if fields["ids"] is not None:
                    record[f"{prefix}_detection_id"] = _python_scalar(
                        fields["ids"][index]
                    )
            records.append(record)

        return {"overlaps": records}


def _detections(payload):
    return payload.detections if isinstance(payload, SegmentationView) else payload


def _has_masks(payload):
    return isinstance(payload, SegmentationView) or (
        isinstance(payload, InstanceDetections) and payload.mask is not None
    )


def _record_fields(detections):
    mapping = (detections.image_metadata or {}).get(CLASS_NAMES_KEY) or {}
    classes = [
        mapping.get(int(cid), f"class_{int(cid)}")
        for cid in detections.class_id.detach().cpu().tolist()
    ]
    confidence = (
        [None] * len(detections)
        if detections.confidence is None
        else detections.confidence.detach().cpu().tolist()
    )
    metadata = detections.bboxes_metadata
    ids = None if metadata is None else [row.get("detection_id") for row in metadata]
    if ids is not None and all(value is None for value in ids):
        ids = None

    return {"classes": classes, "confidence": confidence, "ids": ids}


def _polygons_at_cpu_reference(payload, *, indices):
    # Selection precedes reference materialization, including low-resolution
    # views and RLE: disjoint rows must never demand full-resolution masks.
    if not indices:
        return {}

    boxes = _detections(payload).xyxy[indices].detach().cpu().tolist()
    masks = None
    if _has_masks(payload):
        provider = mask_provider_of(payload).select(indices=indices)
        masks = (
            provider.binary(fidelity="reference")
            .predictions.mask.detach()
            .cpu()
            .numpy()
        )

    polygons = {}
    for position, (index, coordinates) in enumerate(zip(indices, boxes)):
        polygon = box(*coordinates)
        if masks is not None and np.any(masks[position]):
            contours = sv.mask_to_polygons(mask=masks[position].astype(np.uint8))
            if contours:
                longest = max(contours, key=len)
                if len(longest) >= 3:
                    precise = Polygon(longest)
                    if precise.is_valid and not precise.is_empty:
                        polygon = precise
        polygons[index] = polygon

    return polygons


def _polygon_reference_overlaps(reference, candidate, *, eligible, min_overlap):
    pairs = eligible.nonzero().cpu().tolist()
    ref_indices = sorted({i for i, _ in pairs})
    cand_indices = sorted({j for _, j in pairs})
    ref_polygons = _polygons_at_cpu_reference(reference, indices=ref_indices)
    cand_polygons = _polygons_at_cpu_reference(candidate, indices=cand_indices)
    accepted, ratios = [], []
    for i, j in pairs:
        ref, cand = ref_polygons[i], cand_polygons[j]
        if ref.area <= 0:
            continue

        ratio = ref.intersection(cand).area / ref.area
        if ratio >= min_overlap:
            accepted.append([i, j])
            ratios.append(ratio)

    return accepted, ratios


def _python_scalar(value):
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass

    return value
